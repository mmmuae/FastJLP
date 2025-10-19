#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>

#define GPU_BACKEND_METAL 1

#include "Constants.h"
#include "GPU/metal/MetalPacking.h"

namespace {

Int RandomCoordinate(std::mt19937_64& rng) {
  Int value;
  value.SetInt32(0);
  for (int i = 0; i < 4; ++i) {
    value.bits64[i] = rng();
  }
  value.bits64[4] = 0;
  return value;
}

Int RandomDistance(std::mt19937_64& rng) {
  Int value;
  value.SetInt32(0);
  for (int i = 0; i < 2; ++i) {
    value.bits64[i] = rng();
  }
  value.bits64[1] &= 0x0FFFFFFFFFFFFFFFULL;
  for (int i = 2; i < NB64BLOCK; ++i) {
    value.bits64[i] = 0;
  }
  return value;
}

Int BuildOrder() {
  Int order;
  order.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
  return order;
}

}  // namespace

int main() {
  std::mt19937_64 rng(0xC0FFEEu);

  static Int order = BuildOrder();
  Int::InitK1(&order);

  Int wildOffset = RandomDistance(rng);

  for (uint64_t idx = 0; idx < 64; ++idx) {
    Int px = RandomCoordinate(rng);
    Int py = RandomCoordinate(rng);
    Int dist = RandomDistance(rng);
    Int originalDist(dist);

    MetalKangaroo packed{};
    PackKangaroo(px, py, dist, idx, wildOffset, packed);

    for (int limb = 0; limb < 4; ++limb) {
      assert(packed.px[limb] == px.bits64[limb]);
      assert(packed.py[limb] == py.bits64[limb]);
    }

    Int adjusted(originalDist);
    if ((idx % 2) == WILD) {
      adjusted.ModAddK1order(&wildOffset);
    }
    assert(packed.dist[0] == adjusted.bits64[0]);
    assert(packed.dist[1] == adjusted.bits64[1]);

#ifdef USE_SYMMETRY
    assert(packed.lastJump == NB_JUMP);
#endif

    Int outPx;
    Int outPy;
    Int outDist;
    UnpackKangaroo(packed, idx, wildOffset, outPx, outPy, outDist);

    for (int limb = 0; limb < NB64BLOCK; ++limb) {
      assert(outPx.bits64[limb] == px.bits64[limb]);
      assert(outPy.bits64[limb] == py.bits64[limb]);
      assert(outDist.bits64[limb] == originalDist.bits64[limb]);
    }
  }

  std::cout << "MetalPackingTests: all pack/unpack checks passed (" << 64
            << " kangaroos)." << std::endl;
  return 0;
}
