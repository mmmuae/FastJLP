#ifndef GPU_METAL_MATH_H
#define GPU_METAL_MATH_H

#ifdef GPU_BACKEND_METAL
#ifdef __METAL_VERSION__

/*
  MetalMath.h — Metal backend (final, do-while scoped chains)

  Key points:
  - No function-scope statics.
  - Explicit address spaces for refs (`thread &`).
  - Carry/borrow/MAD chains use do{...}while(0) to create a fresh scope;
    this prevents variable redeclaration errors and guarantees braces
    are balanced, eliminating “function definition is not allowed here”.
  - Call sites (UADDO→UADDC→UADD, USUBO→USUBC→USUB, MADDO→MADDC→MADD/MADDS)
    remain unchanged.
*/

#include <metal_stdlib>
using namespace metal;

#include "MetalConstants.h"

#ifndef NBBLOCK
#define NBBLOCK 5
#endif

#ifndef BIFULLSIZE
#define BIFULLSIZE 40
#endif

namespace metal_math {

// ---------- Types ----------
struct uint256_t { ulong d[4]; };

// ---------- 64-bit add/sub helpers ----------
inline ulong add64_c(ulong a, ulong b, thread bool &carry) {
  ulong sum = a + b;
  bool overflow = (sum < a);
  if (carry) {
    ulong sum2 = sum + 1ul;
    bool overflow2 = (sum2 < sum);
    sum = sum2;
    carry = overflow || overflow2;
  } else {
    carry = overflow;
  }
  return sum;
}

inline ulong add64_nocc(ulong a, ulong b, thread bool &carry) {
  ulong sum = a + b;
  carry = (sum < a);
  return sum;
}

inline ulong sub64_b(ulong a, ulong b, thread bool &borrow) {
  ulong diff = a - b;
  bool under = (a < b);
  if (borrow) {
    bool under2 = (diff == 0ul);
    diff -= 1ul;
    borrow = under || under2;
  } else {
    borrow = under;
  }
  return diff;
}

// Numeric carry/borrow (0/1), constant-time friendly
inline ulong add64_raw(ulong a, ulong b, thread ulong &carry) {
  ulong sum = a + b;
  bool c1 = (sum < a);
  if (carry != 0ul) {
    ulong sum2 = sum + carry;
    bool c2 = (sum2 < sum);
    carry = (c1 || c2) ? 1ul : 0ul;
    return sum2;
  }
  carry = c1 ? 1ul : 0ul;
  return sum;
}

inline ulong sub64_raw(ulong a, ulong b, thread ulong &borrow) {
  ulong diff = a - b;
  bool b1 = (a < b);
  if (borrow != 0ul) {
    ulong diff2 = diff - borrow;
    bool b2 = (diff < borrow);
    borrow = (b1 || b2) ? 1ul : 0ul;
    return diff2;
  }
  borrow = b1 ? 1ul : 0ul;
  return diff;
}

// Signed add used by MADDS: carry becomes 1 if signed sum < 0
inline ulong add64_signed_raw(ulong lhs, ulong rhs, thread ulong &carry) {
  long sum = static_cast<long>(lhs) + static_cast<long>(rhs);
  if (carry != 0ul) sum += static_cast<long>(carry);
  carry = (sum < 0) ? 1ul : 0ul;
  return static_cast<ulong>(sum);
}

// ---------- Multiplication helpers ----------
inline ulong mul64_hi(ulong a, ulong b) { return mulhi(a, b); }

inline ulong2 mul64_wide_emulated(ulong a, ulong b) {
  uint a0 = static_cast<uint>(a & 0xFFFFFFFFul);
  uint a1 = static_cast<uint>(a >> 32);
  uint b0 = static_cast<uint>(b & 0xFFFFFFFFul);
  uint b1 = static_cast<uint>(b >> 32);
  ulong p00 = static_cast<ulong>(a0) * static_cast<ulong>(b0);
  ulong p01 = static_cast<ulong>(a0) * static_cast<ulong>(b1);
  ulong p10 = static_cast<ulong>(a1) * static_cast<ulong>(b0);
  ulong p11 = static_cast<ulong>(a1) * static_cast<ulong>(b1);
  ulong mid = (p00 >> 32) + (p01 & 0xFFFFFFFFul) + (p10 & 0xFFFFFFFFul);
  ulong lo  = (p00 & 0xFFFFFFFFul) | (mid << 32);
  ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
  return ulong2(lo, hi);
}

inline ulong2 mul64_wide(ulong a, ulong b) {
#if (__METAL_VERSION__ >= 200)
  return ulong2(a * b, mul64_hi(a, b));
#else
  return mul64_wide_emulated(a, b);
#endif
}

inline ulong mul_lo(ulong a, ulong b)   { return a * b; }
inline ulong mul_hi64(ulong a, ulong b) { return mulhi(a, b); }

// ---------- 256-bit helpers ----------
inline uint256_t u256_add(uint256_t a, uint256_t b) {
  bool carry = false;
  uint256_t r;
  r.d[0] = add64_nocc(a.d[0], b.d[0], carry);
  r.d[1] = add64_c(a.d[1], b.d[1], carry);
  r.d[2] = add64_c(a.d[2], b.d[2], carry);
  r.d[3] = add64_c(a.d[3], b.d[3], carry);
  return r;
}

inline uint256_t u256_sub(uint256_t a, uint256_t b) {
  bool borrow = false;
  uint256_t r;
  r.d[0] = sub64_b(a.d[0], b.d[0], borrow);
  r.d[1] = sub64_b(a.d[1], b.d[1], borrow);
  r.d[2] = sub64_b(a.d[2], b.d[2], borrow);
  r.d[3] = sub64_b(a.d[3], b.d[3], borrow);
  return r;
}

inline bool u256_ge(uint256_t x, uint256_t y) {
  if (x.d[3] != y.d[3]) return x.d[3] > y.d[3];
  if (x.d[2] != y.d[2]) return x.d[2] > y.d[2];
  if (x.d[1] != y.d[1]) return x.d[1] > y.d[1];
  return x.d[0] >= y.d[0];
}

} // namespace metal_math

// ======================================================================
// Scoped-chain macros (do { ... } while (0)) — ALWAYS balanced.
// ======================================================================

// ---- Unsigned Add chain ----
#define UADDO(res, a, b) \
  do { thread ulong __mm_add = 0ul; \
       (res) = metal_math::add64_raw((ulong)(a), (ulong)(b), __mm_add)

#define UADDC(res, a, b) \
     ; (res) = metal_math::add64_raw((ulong)(a), (ulong)(b), __mm_add)

#define UADD(res, a, b) \
     ; (res) = metal_math::add64_raw((ulong)(a), (ulong)(b), __mm_add); \
  } while (0)

#define UADDO1(res, a) \
  do { thread ulong __mm_add = 0ul; \
       (res) = metal_math::add64_raw((ulong)(res), (ulong)(a), __mm_add)

#define UADDC1(res, a) \
     ; (res) = metal_math::add64_raw((ulong)(res), (ulong)(a), __mm_add)

#define UADD1(res, a) \
     ; (res) = metal_math::add64_raw((ulong)(res), (ulong)(a), __mm_add); \
  } while (0)

// ---- Unsigned Sub chain ----
#define USUBO(res, a, b) \
  do { thread ulong __mm_sub = 0ul; \
       (res) = metal_math::sub64_raw((ulong)(a), (ulong)(b), __mm_sub)

#define USUBC(res, a, b) \
     ; (res) = metal_math::sub64_raw((ulong)(a), (ulong)(b), __mm_sub)

#define USUB(res, a, b) \
     ; (res) = metal_math::sub64_raw((ulong)(a), (ulong)(b), __mm_sub); \
  } while (0)

#define USUBO1(res, a) \
  do { thread ulong __mm_sub = 0ul; \
       (res) = metal_math::sub64_raw((ulong)(res), (ulong)(a), __mm_sub)

#define USUBC1(res, a) \
     ; (res) = metal_math::sub64_raw((ulong)(res), (ulong)(a), __mm_sub)

#define USUB1(res, a) \
     ; (res) = metal_math::sub64_raw((ulong)(res), (ulong)(a), __mm_sub); \
  } while (0)

// ---- Multiply helpers ----
#define UMULLO(lo, a, b) \
  (lo) = metal_math::mul_lo((ulong)(a), (ulong)(b))

#define UMULHI(hi, a, b) \
  (hi) = metal_math::mul_hi64((ulong)(a), (ulong)(b))

// ---- Multiply-Add chain (uses HI product) ----
#define MADDO(res, a, b, c) \
  do { thread ulong __mm_mad = 0ul; \
       ulong __mhi = metal_math::mul_hi64((ulong)(a), (ulong)(b)); \
       (res) = metal_math::add64_raw(__mhi, (ulong)(c), __mm_mad)

#define MADDC(res, a, b, c) \
     ; { ulong __mhi = metal_math::mul_hi64((ulong)(a), (ulong)(b)); \
         (res) = metal_math::add64_raw(__mhi, (ulong)(c), __mm_mad); }

#define MADD(res, a, b, c) \
     ; { ulong __mhi = metal_math::mul_hi64((ulong)(a), (ulong)(b)); \
         (res) = metal_math::add64_raw(__mhi, (ulong)(c), __mm_mad); } \
  } while (0)

#define MADDS(res, a, b, c) \
     ; { ulong __mhi = metal_math::mul_hi64((ulong)(a), (ulong)(b)); \
         (res) = metal_math::add64_signed_raw(__mhi, (ulong)(c), __mm_mad); } \
  } while (0)

// CUDA-style barrier alias
#define __syncthreads() threadgroup_barrier(mem_flags::mem_threadgroup)

#endif  // __METAL_VERSION__
#endif  // GPU_BACKEND_METAL
#endif  // GPU_METAL_MATH_H
