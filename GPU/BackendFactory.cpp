#include "BackendFactory.h"

#include <cstdio>

#ifdef GPU_BACKEND_METAL
#include "metal/MetalBackend.hpp"
#endif

namespace {

class NullBackend : public IGpuBackend {
public:
  explicit NullBackend(const char* label) : label_(label) {}
  bool init() override { warn("init"); return false; }
  bool allocate(const Buffers&, const GpuConfig&) override { warn("allocate"); return false; }
  bool uploadJumps(const void*, const void*, const void*, uint32_t) override { warn("uploadJumps"); return false; }
  bool uploadKangaroos(const void*, size_t) override { warn("uploadKangaroos"); return false; }
  bool runOnce() override { warn("runOnce"); return false; }
  bool readDP(void*, size_t, uint32_t&) override { warn("readDP"); return false; }
  bool downloadKangaroos(void*, size_t) override { warn("downloadKangaroos"); return false; }
  void resetDPCount() override { warn("resetDPCount"); }
  void shutdown() override { warn("shutdown"); }
private:
  void warn(const char* action) const {
    std::printf("%s backend stub: %s not implemented\n", label_, action);
  }
  const char* label_;
};

#ifdef GPU_BACKEND_CUDA
IGpuBackend* CreateCudaBackend() {
  return new NullBackend("CUDA");
}
#endif

#ifdef GPU_BACKEND_METAL
IGpuBackend* CreateMetalBackend() {
  return new MetalBackend();
}
#endif

}  // namespace

IGpuBackend* CreateBackend(BackendKind kind) {
  switch (kind) {
#ifdef GPU_BACKEND_CUDA
    case BackendKind::CUDA:
      return CreateCudaBackend();
#endif
#ifdef GPU_BACKEND_METAL
    case BackendKind::METAL:
      return CreateMetalBackend();
#endif
    default:
      break;
  }
  return nullptr;
}

const char* BackendName(BackendKind kind) {
  switch (kind) {
    case BackendKind::CUDA:
      return "cuda";
    case BackendKind::METAL:
      return "metal";
    default:
      break;
  }
  return "unknown";
}

bool IsBackendAvailable(BackendKind kind) {
  switch (kind) {
#ifdef GPU_BACKEND_CUDA
    case BackendKind::CUDA:
      return true;
#endif
#ifdef GPU_BACKEND_METAL
    case BackendKind::METAL:
      return true;
#endif
    default:
      break;
  }
  return false;
}

BackendKind GetDefaultBackend() {
#if defined(GPU_BACKEND_CUDA)
  return BackendKind::CUDA;
#elif defined(GPU_BACKEND_METAL)
  return BackendKind::METAL;
#else
  return BackendKind::CUDA;
#endif
}
