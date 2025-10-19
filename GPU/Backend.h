#pragma once
#include <cstddef>
#include <cstdint>

struct GpuConfig {
  uint32_t threadsPerGroup;
  uint32_t groups;
  uint32_t iterationsPerDispatch;
  uint32_t jumpCount;  // n = floor(log2(sqrt(range))) + 1
  uint64_t dpMask;     // 0 => store all steps
  uint32_t maxFound;   // capacity of DP ring (items)
};

struct Buffers {
  void* kangaroos;  // device buffer (struct matches JLP's GPU state)
  void* jumpDist;   // device buffer: ulong[4]*n
  void* jumpPx;     // device buffer: uint256_t*n
  void* jumpPy;     // device buffer: uint256_t*n
  void* dpItems;    // device buffer: uint32_t[ITEM_SIZE32*maxFound+1]
  void* prime;      // device buffer: uint256_t
  void* dpCount;    // device buffer: atomic uint
  uint32_t totalKangaroos;
};

class IGpuBackend {
public:
  virtual ~IGpuBackend() {}
  virtual bool init() = 0;
  virtual bool allocate(const Buffers& h, const GpuConfig& c) = 0;
  virtual bool uploadJumps(const void* d, const void* px, const void* py, uint32_t n) = 0;
  virtual bool uploadKangaroos(const void* host, size_t bytes) = 0;
  virtual bool runOnce() = 0;
  virtual bool readDP(void* hostDp, size_t bytes, uint32_t& outCount) = 0;
  virtual bool downloadKangaroos(void* host, size_t bytes) = 0;
  virtual void resetDPCount() = 0;
  virtual void shutdown() = 0;
};
