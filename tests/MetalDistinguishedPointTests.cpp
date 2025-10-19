#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>

#include "GPU/GPUEngine.h"
#include "GPU/metal/MetalDistinguishedPoint.h"
#include "SECPK1/Int.h"

namespace {

bool CheckFixedLayout() {
  MetalDpItem item;
  item.x.SetInt32(0);
  item.dist.SetInt32(0);
  item.index = 0x1122334455667788ULL;

  item.x.bits64[0] = 0x0123456789ABCDEFULL;
  item.x.bits64[1] = 0x0FEDCBA987654321ULL;
  item.x.bits64[2] = 0x0011223344556677ULL;
  item.x.bits64[3] = 0x8899AABBCCDDEEFFULL;

  item.dist.bits64[0] = 0x1234567890ABCDEFULL;
  item.dist.bits64[1] = 0x0FEDCBA098765432ULL;

  std::array<uint32_t, ITEM_SIZE32> words{};
  MetalEncodeDistinguishedPoint(item, words.data());

  const uint32_t* xWords = reinterpret_cast<const uint32_t*>(item.x.bits64);
  for (size_t i = 0; i < 8; ++i) {
    if (words[i] != xWords[i]) {
      std::fprintf(stderr, "Encode mismatch at x[%zu]\n", i);
      return false;
    }
  }

  const uint32_t* distWords = reinterpret_cast<const uint32_t*>(item.dist.bits64);
  for (size_t i = 0; i < 4; ++i) {
    if (words[8 + i] != distWords[i]) {
      std::fprintf(stderr, "Encode mismatch at dist[%zu]\n", i);
      return false;
    }
  }

  if (words[12] != static_cast<uint32_t>(item.index & 0xFFFFFFFFULL) ||
      words[13] != static_cast<uint32_t>((item.index >> 32) & 0xFFFFFFFFULL)) {
    std::fprintf(stderr, "Encode mismatch at index words\n");
    return false;
  }

  MetalDpItem decoded;
  MetalDecodeDistinguishedPoint(words.data(), decoded);

  for (size_t i = 0; i < 4; ++i) {
    if (decoded.x.bits64[i] != item.x.bits64[i]) {
      std::fprintf(stderr, "Decode mismatch at x limb %zu\n", i);
      return false;
    }
  }

  for (size_t i = 0; i < 2; ++i) {
    if (decoded.dist.bits64[i] != item.dist.bits64[i]) {
      std::fprintf(stderr, "Decode mismatch at dist limb %zu\n", i);
      return false;
    }
  }

  if (decoded.index != item.index) {
    std::fprintf(stderr, "Decode mismatch at index\n");
    return false;
  }

  for (size_t i = 2; i < 5; ++i) {
    if (decoded.dist.bits64[i] != 0) {
      std::fprintf(stderr, "Unexpected dist limb data at %zu\n", i);
      return false;
    }
  }
  if (decoded.x.bits64[4] != 0) {
    std::fprintf(stderr, "Unexpected x high limb data\n");
    return false;
  }

  return true;
}

bool CheckRandomRoundTrips() {
  std::mt19937_64 rng(1337);
  std::array<uint32_t, ITEM_SIZE32> words{};
  for (int caseIdx = 0; caseIdx < 32; ++caseIdx) {
    MetalDpItem input;
    input.x.SetInt32(0);
    input.dist.SetInt32(0);
    for (int limb = 0; limb < 4; ++limb) {
      input.x.bits64[limb] = rng();
    }
    for (int limb = 0; limb < 2; ++limb) {
      input.dist.bits64[limb] = rng();
    }
    input.index = (static_cast<uint64_t>(static_cast<uint32_t>(rng())) << 32) |
                  static_cast<uint32_t>(rng());

    MetalEncodeDistinguishedPoint(input, words.data());
    MetalDpItem decoded;
    MetalDecodeDistinguishedPoint(words.data(), decoded);

    for (int limb = 0; limb < 4; ++limb) {
      if (decoded.x.bits64[limb] != input.x.bits64[limb]) {
        std::fprintf(stderr, "Round trip x mismatch at case %d limb %d\n", caseIdx, limb);
        return false;
      }
    }
    for (int limb = 0; limb < 2; ++limb) {
      if (decoded.dist.bits64[limb] != input.dist.bits64[limb]) {
        std::fprintf(stderr, "Round trip dist mismatch at case %d limb %d\n", caseIdx, limb);
        return false;
      }
    }
    if (decoded.index != input.index) {
      std::fprintf(stderr, "Round trip index mismatch at case %d\n", caseIdx);
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckFixedLayout()) {
    return 1;
  }
  if (!CheckRandomRoundTrips()) {
    return 1;
  }
  std::cout << "MetalDistinguishedPointTests: layout and round-trip checks "
               "passed." << std::endl;
  return 0;
}
