#pragma once

#ifdef GPU_BACKEND_METAL

#include <cstdint>
#include <cstring>

#include "SECPK1/Int.h"

struct MetalDpItem {
  Int x;
  Int dist;
  uint64_t index;
};

inline void MetalEncodeDistinguishedPoint(const MetalDpItem& src,
                                          uint32_t* destWords) {
  if (!destWords) {
    return;
  }
  std::memcpy(destWords, src.x.bits64, 4 * sizeof(uint64_t));
  std::memcpy(destWords + 8, src.dist.bits64, 2 * sizeof(uint64_t));
  destWords[12] = static_cast<uint32_t>(src.index & 0xFFFFFFFFULL);
  destWords[13] = static_cast<uint32_t>((src.index >> 32) & 0xFFFFFFFFULL);
}

inline void MetalDecodeDistinguishedPoint(const uint32_t* srcWords,
                                          MetalDpItem& out) {
  if (!srcWords) {
    return;
  }
  out.x.SetInt32(0);
  out.dist.SetInt32(0);
  const uint64_t* xWords = reinterpret_cast<const uint64_t*>(srcWords);
  out.x.bits64[0] = xWords[0];
  out.x.bits64[1] = xWords[1];
  out.x.bits64[2] = xWords[2];
  out.x.bits64[3] = xWords[3];
  const uint64_t* distWords = reinterpret_cast<const uint64_t*>(srcWords + 8);
  out.dist.bits64[0] = distWords[0];
  out.dist.bits64[1] = distWords[1];
  out.index = (static_cast<uint64_t>(srcWords[13]) << 32) |
              static_cast<uint64_t>(srcWords[12]);
}

#endif  // GPU_BACKEND_METAL
