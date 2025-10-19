#include <cstdio>
#include <cstring>
#include <vector>

#ifdef __APPLE__
#define Point OSXPoint
#import <Metal/Metal.h>
#undef Point
#include "../GPU/GPUEngine.h"
#include "../GPU/metal/MetalBackend.hpp"

namespace {
constexpr uint32_t kThreadsPerGroup = 32;
constexpr uint32_t kGroups = 1;
constexpr uint32_t kIterations = 1;
constexpr uint32_t kJumpCount = 4;
constexpr uint32_t kMaxFound = 8;
constexpr uint32_t kTotalKangaroos = kThreadsPerGroup * kGroups;

struct DummyKangaroo {
  uint64_t px[4];
  uint64_t py[4];
  uint64_t dist[2];
};

void FillSequence(uint64_t* data, size_t count, uint64_t base) {
  for (size_t i = 0; i < count; ++i) {
    data[i] = base + static_cast<uint64_t>(i);
  }
}

bool RunSmoke() {
  MetalBackend backend;
  if (!backend.init()) {
    std::fprintf(stderr, "MetalBackendSmoke: init failed\n");
    return false;
  }

  std::vector<DummyKangaroo> herd(kTotalKangaroos);
  for (size_t i = 0; i < herd.size(); ++i) {
    FillSequence(herd[i].px, 4, static_cast<uint64_t>(i) * 10ULL);
    FillSequence(herd[i].py, 4, static_cast<uint64_t>(i) * 10ULL + 100ULL);
    FillSequence(herd[i].dist, 2, static_cast<uint64_t>(i) * 10ULL + 200ULL);
  }

  std::vector<uint64_t> jumpDist(kJumpCount * 2);
  std::vector<uint64_t> jumpPx(kJumpCount * 4);
  std::vector<uint64_t> jumpPy(kJumpCount * 4);
  FillSequence(jumpDist.data(), jumpDist.size(), 500ULL);
  FillSequence(jumpPx.data(), jumpPx.size(), 1000ULL);
  FillSequence(jumpPy.data(), jumpPy.size(), 2000ULL);

  alignas(16) uint64_t prime[4] = {0xFFFFFFFFULL, 0, 0, 0};

  Buffers buffers{};
  buffers.kangaroos = herd.data();
  buffers.jumpDist = jumpDist.data();
  buffers.jumpPx = jumpPx.data();
  buffers.jumpPy = jumpPy.data();
  buffers.dpItems = nullptr;
  buffers.prime = prime;
  buffers.dpCount = nullptr;
  buffers.totalKangaroos = kTotalKangaroos;

  GpuConfig config{};
  config.threadsPerGroup = kThreadsPerGroup;
  config.groups = kGroups;
  config.iterationsPerDispatch = kIterations;
  config.jumpCount = kJumpCount;
  config.dpMask = 0;
  config.maxFound = kMaxFound;

  if (!backend.allocate(buffers, config)) {
    std::fprintf(stderr, "MetalBackendSmoke: allocate failed\n");
    return false;
  }

  if (!backend.uploadJumps(jumpDist.data(), jumpPx.data(), jumpPy.data(), kJumpCount)) {
    std::fprintf(stderr, "MetalBackendSmoke: uploadJumps failed\n");
    return false;
  }

  if (!backend.uploadKangaroos(herd.data(), herd.size() * sizeof(DummyKangaroo))) {
    std::fprintf(stderr, "MetalBackendSmoke: uploadKangaroos failed\n");
    return false;
  }

  backend.resetDPCount();

  if (!backend.runOnce()) {
    std::fprintf(stderr, "MetalBackendSmoke: runOnce failed\n");
    return false;
  }

  std::vector<uint32_t> dpRing((kMaxFound + 1) * (ITEM_SIZE / sizeof(uint32_t)));
  uint32_t dpCount = 0;
  if (!backend.readDP(dpRing.data(), dpRing.size() * sizeof(uint32_t), dpCount)) {
    std::fprintf(stderr, "MetalBackendSmoke: readDP failed\n");
    return false;
  }

  if (dpCount != 0) {
    std::fprintf(stderr, "MetalBackendSmoke: expected dpCount=0, got %u\n", dpCount);
    return false;
  }

  std::vector<DummyKangaroo> download(herd.size());
  if (!backend.downloadKangaroos(download.data(), download.size() * sizeof(DummyKangaroo))) {
    std::fprintf(stderr, "MetalBackendSmoke: downloadKangaroos failed\n");
    return false;
  }

  for (size_t i = 0; i < herd.size(); ++i) {
    if (std::memcmp(&herd[i], &download[i], sizeof(DummyKangaroo)) != 0) {
      std::fprintf(stderr, "MetalBackendSmoke: kangaroo mismatch at %zu\n", i);
      return false;
    }
  }

  backend.shutdown();
  return true;
}
}  // namespace

int main() {
  return RunSmoke() ? 0 : 1;
}

#else  // __APPLE__

int main() {
  std::fprintf(stderr, "MetalBackendSmoke: requires macOS with Metal support\n");
  return 0;
}

#endif
