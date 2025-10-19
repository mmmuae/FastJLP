#include <metal_stdlib>
using namespace metal;

#include "MetalConstants.h"
#include "MetalMath.h"
#include "../GPUMath.h"

using metal_math::uint256_t;

struct gpu_kpoint_t {
  ulong px[4];
  ulong py[4];
  ulong dist[2];
#if defined(USE_SYMMETRY)
  ulong lastJump;
#endif
};

namespace {
constant uint kDpWords = 14u;

inline void Add128Vec(thread ulong (&dst)[2], ulong2 delta) {
  thread ulong carry = 0;
  dst[0] = metal_math::add64_raw(dst[0], static_cast<ulong>(delta.x), carry);
  dst[1] = metal_math::add64_raw(dst[1], static_cast<ulong>(delta.y), carry);
}

inline void LoadUint256(const ulong4 value, thread ulong (&out)[4]) {
  out[0] = static_cast<ulong>(value.x);
  out[1] = static_cast<ulong>(value.y);
  out[2] = static_cast<ulong>(value.z);
  out[3] = static_cast<ulong>(value.w);
}

inline void StoreDpItem(device uint* dp_items,
                        uint32_t pos,
                        const thread ulong (&x)[4],
                        const thread ulong (&dist)[2],
                        ulong index) {
  device uint* dest = dp_items + static_cast<ulong>(pos) * kDpWords;
  for (int limb = 0; limb < 4; ++limb) {
    ulong value = x[limb];
    dest[limb * 2 + 0] = static_cast<uint>(value & 0xFFFFFFFFULL);
    dest[limb * 2 + 1] = static_cast<uint>((value >> 32) & 0xFFFFFFFFULL);
  }
  for (int limb = 0; limb < 2; ++limb) {
    ulong value = dist[limb];
    dest[8 + limb * 2 + 0] = static_cast<uint>(value & 0xFFFFFFFFULL);
    dest[8 + limb * 2 + 1] = static_cast<uint>((value >> 32) & 0xFFFFFFFFULL);
  }
  dest[12] = static_cast<uint>(index & 0xFFFFFFFFULL);
  dest[13] = static_cast<uint>((index >> 32) & 0xFFFFFFFFULL);
}

}  // namespace

kernel void comp_kangaroos(device gpu_kpoint_t* kangaroos [[buffer(0)]],
                           device atomic_uint* dp_count [[buffer(1)]],
                           device uint* dp_items [[buffer(2)]],
                           device const ulong2* jump_dist [[buffer(3)]],
                           device const ulong4* jump_px [[buffer(4)]],
                           device const ulong4* jump_py [[buffer(5)]],
                           constant uint& max_found [[buffer(6)]],
                           constant ulong& dp_mask [[buffer(7)]],
                           constant uint& nb_run [[buffer(8)]],
                           constant uint& threads_per_group [[buffer(9)]],
                           constant uint& total_threads [[buffer(10)]],
                           constant uint& total_kangaroos [[buffer(11)]],
                           constant uint256_t& prime [[buffer(12)]],
                           constant uint& jump_count [[buffer(13)]],
                           uint tid [[thread_position_in_grid]]) {
  if (tid >= total_threads) {
    return;
  }

  const uint32_t groupThreads = threads_per_group;
  const uint32_t localThread = tid % groupThreads;
  const uint32_t groupIndex = tid / groupThreads;
  const ulong blockBase = static_cast<ulong>(groupIndex) *
                          static_cast<ulong>(groupThreads) *
                          static_cast<ulong>(GPU_GRP_SIZE);

  thread ulong pxGroup[GPU_GRP_SIZE][4];
  thread ulong pyGroup[GPU_GRP_SIZE][4];
  thread ulong distGroup[GPU_GRP_SIZE][2];
#if defined(USE_SYMMETRY)
  thread ulong lastJumpGroup[GPU_GRP_SIZE];
#endif

  for (uint32_t g = 0; g < GPU_GRP_SIZE; ++g) {
    ulong idx = blockBase + static_cast<ulong>(g) * static_cast<ulong>(groupThreads) +
                static_cast<ulong>(localThread);
    if (idx >= total_kangaroos) {
      break;
    }
    const gpu_kpoint_t point = kangaroos[idx];
    for (int limb = 0; limb < 4; ++limb) {
      pxGroup[g][limb] = point.px[limb];
      pyGroup[g][limb] = point.py[limb];
    }
    distGroup[g][0] = point.dist[0];
    distGroup[g][1] = point.dist[1];
#if defined(USE_SYMMETRY)
    lastJumpGroup[g] = point.lastJump;
#endif
  }

  thread ulong dx[GPU_GRP_SIZE][4];
  thread ulong dy[4];
  thread ulong rx[4];
  thread ulong ry[4];
  thread ulong slope[4];
  thread ulong tmp[4];
  (void)prime;

  for (uint run = 0; run < nb_run; ++run) {
    for (uint32_t g = 0; g < GPU_GRP_SIZE; ++g) {
      ulong xLimb0 = pxGroup[g][0];
      uint32_t jmp = static_cast<uint32_t>(xLimb0 % static_cast<ulong>(jump_count));
#if defined(USE_SYMMETRY)
      if (jmp == lastJumpGroup[g]) {
        jmp = (lastJumpGroup[g] + 1u) % jump_count;
      }
      lastJumpGroup[g] = jmp;
#endif
      ulong4 jumpPxVec = jump_px[jmp];
      thread ulong jumpPx[4];
      LoadUint256(jumpPxVec, jumpPx);
      ModSub256(dx[g], pxGroup[g], jumpPx);
    }

    _ModInvGrouped(dx);

    for (uint32_t g = 0; g < GPU_GRP_SIZE; ++g) {
#if defined(USE_SYMMETRY)
      uint32_t jmp = lastJumpGroup[g];
#else
      ulong xLimb0 = pxGroup[g][0];
      uint32_t jmp = static_cast<uint32_t>(xLimb0 % static_cast<ulong>(jump_count));
#endif
      ulong4 jumpPxVec = jump_px[jmp];
      ulong4 jumpPyVec = jump_py[jmp];
      thread ulong jumpPx[4];
      thread ulong jumpPy[4];
      LoadUint256(jumpPxVec, jumpPx);
      LoadUint256(jumpPyVec, jumpPy);

      ModSub256(dy, pyGroup[g], jumpPy);
      _ModMult(slope, dy, dx[g]);
      _ModSqr(tmp, slope);
      ModSub256(rx, tmp, jumpPx);
      ModSub256(rx, pxGroup[g]);

      ModSub256(ry, pxGroup[g], rx);
      _ModMult(ry, slope);
      ModSub256(ry, pyGroup[g]);

      Load256(pxGroup[g], rx);
      Load256(pyGroup[g], ry);

      ulong2 jumpDistVec = jump_dist[jmp];
      Add128Vec(distGroup[g], jumpDistVec);

#if defined(USE_SYMMETRY)
      if (ModPositive256(pyGroup[g])) {
        ModNeg256Order(distGroup[g]);
      }
#endif

      if ((pxGroup[g][3] & dp_mask) == 0u) {
        uint32_t pos = atomic_fetch_add_explicit(dp_count, 1u, memory_order_relaxed);
        if (pos < max_found) {
          ulong globalIdx = blockBase + static_cast<ulong>(g) * static_cast<ulong>(groupThreads) +
                            static_cast<ulong>(localThread);
          StoreDpItem(dp_items, pos, pxGroup[g], distGroup[g], globalIdx);
        }
      }
    }
  }

  for (uint32_t g = 0; g < GPU_GRP_SIZE; ++g) {
    ulong idx = blockBase + static_cast<ulong>(g) * static_cast<ulong>(groupThreads) +
                static_cast<ulong>(localThread);
    if (idx >= total_kangaroos) {
      continue;
    }
    gpu_kpoint_t updated;
    for (int limb = 0; limb < 4; ++limb) {
      updated.px[limb] = pxGroup[g][limb];
      updated.py[limb] = pyGroup[g][limb];
    }
    updated.dist[0] = distGroup[g][0];
    updated.dist[1] = distGroup[g][1];
#if defined(USE_SYMMETRY)
    updated.lastJump = lastJumpGroup[g];
#endif
    kangaroos[idx] = updated;
  }
}
