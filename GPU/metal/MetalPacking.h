#pragma once

#ifdef GPU_BACKEND_METAL

#include <cstdint>

#include "MetalConstants.h"
#include "SECPK1/Int.h"

struct MetalKangaroo {
  uint64_t px[4];
  uint64_t py[4];
  uint64_t dist[2];
#ifdef USE_SYMMETRY
  uint64_t lastJump;
#endif
};

inline void PackKangaroo(const Int& px,
                         const Int& py,
                         const Int& dist,
                         uint64_t index,
                         const Int& wildOffset,
                         MetalKangaroo& out) {
  for (int i = 0; i < 4; ++i) {
    out.px[i] = px.bits64[i];
    out.py[i] = py.bits64[i];
  }

  Int adjusted;
  adjusted.Set(const_cast<Int*>(&dist));
  if ((index % 2) == WILD) {
    adjusted.ModAddK1order(const_cast<Int*>(&wildOffset));
  }
  out.dist[0] = adjusted.bits64[0];
  out.dist[1] = adjusted.bits64[1];
#ifdef USE_SYMMETRY
  out.lastJump = NB_JUMP;
#endif
}

inline void UnpackKangaroo(const MetalKangaroo& src,
                           uint64_t index,
                           const Int& wildOffset,
                           Int& px,
                           Int& py,
                           Int& dist) {
  px.SetInt32(0);
  py.SetInt32(0);
  dist.SetInt32(0);
  for (int i = 0; i < 4; ++i) {
    px.bits64[i] = src.px[i];
    py.bits64[i] = src.py[i];
  }
  Int adjusted;
  adjusted.SetInt32(0);
  adjusted.bits64[0] = src.dist[0];
  adjusted.bits64[1] = src.dist[1];
  if ((index % 2) == WILD) {
    adjusted.ModSubK1order(const_cast<Int*>(&wildOffset));
  }
  dist.Set(&adjusted);
}

#endif  // GPU_BACKEND_METAL
