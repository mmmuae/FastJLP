#pragma once

#include "../Backend.h"

struct MetalImpl;

class MetalBackend : public IGpuBackend {
public:
  MetalBackend();
  ~MetalBackend() override;

  bool init() override;
  bool allocate(const Buffers& buffers, const GpuConfig& config) override;
  bool uploadJumps(const void* distances, const void* px, const void* py, uint32_t count) override;
  bool uploadKangaroos(const void* host, size_t bytes) override;
  bool runOnce() override;
  bool readDP(void* hostDp, size_t bytes, uint32_t& outCount) override;
  bool downloadKangaroos(void* host, size_t bytes) override;
  void resetDPCount() override;
  void shutdown() override;

private:
  MetalImpl* impl_;
};
