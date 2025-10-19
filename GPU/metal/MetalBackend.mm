#include "MetalBackend.hpp"

#ifdef GPU_BACKEND_METAL

#ifdef __APPLE__
#define Point OSXPoint
#endif

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#ifdef __APPLE__
#undef Point
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../GPUEngine.h"
#include "MetalConstants.h"

namespace {

template <typename T>
T RetainIfNeeded(T object) {
#if !__has_feature(objc_arc)
  if (object) {
    return [object retain];
  }
#endif
  return object;
}

template <typename T>
void ReleaseIfNeeded(T object) {
#if !__has_feature(objc_arc)
  if (object) {
    [object release];
  }
#else
  (void)object;
#endif
}

std::string LoadFileContents(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) {
    return {};
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

void RemoveIncludeLine(std::string& text, const std::string& token) {
  const size_t pos = text.find(token);
  if (pos == std::string::npos) {
    return;
  }
  size_t lineStart = text.rfind('\n', pos);
  if (lineStart == std::string::npos) {
    lineStart = 0;
  } else {
    lineStart += 1;
  }
  size_t lineEnd = text.find('\n', pos);
  if (lineEnd == std::string::npos) {
    text.erase(lineStart);
  } else {
    text.erase(lineStart, lineEnd - lineStart + 1);
  }
}

void RemovePragmaOnce(std::string& text) {
  const std::string token = "#pragma once";
  size_t searchPos = 0;
  while ((searchPos = text.find(token, searchPos)) != std::string::npos) {
    size_t lineStart = text.rfind('\n', searchPos);
    if (lineStart == std::string::npos) {
      lineStart = 0;
    } else {
      lineStart += 1;
    }
    size_t lineEnd = text.find('\n', searchPos);
    if (lineEnd == std::string::npos) {
      text.erase(lineStart);
      break;
    }
    text.erase(lineStart, lineEnd - lineStart + 1);
  }
}

std::string LoadKernelSource() {
  namespace fs = std::filesystem;
  std::vector<fs::path> candidates;
  candidates.push_back(fs::path(__FILE__).parent_path() / "kernels.metal");
  if (NSBundle* bundle = [NSBundle mainBundle]) {
    if (NSString* resource = [bundle resourcePath]) {
      candidates.push_back(fs::path([resource UTF8String]) / "GPU/metal/kernels.metal");
      candidates.push_back(fs::path([resource UTF8String]) / "kernels.metal");
    }
  }
  candidates.push_back(fs::current_path() / "GPU/metal/kernels.metal");

  fs::path kernelPath;
  std::string kernelSource;
  for (const auto& path : candidates) {
    std::string contents = LoadFileContents(path);
    if (contents.empty()) {
      continue;
    }
    kernelPath = path;
    kernelSource = std::move(contents);
    break;
  }

  if (kernelSource.empty()) {
    std::fprintf(stderr, "Metal backend: kernel source not found\n");
    return {};
  }

  const fs::path baseDir = kernelPath.parent_path();

  auto resolveHeader = [&](const fs::path& relative) -> std::pair<fs::path, std::string> {
    std::vector<fs::path> headerCandidates;
    headerCandidates.push_back(baseDir / relative);
    headerCandidates.push_back(fs::path(__FILE__).parent_path() / relative);
    headerCandidates.push_back(fs::current_path() / relative);
    for (const auto& candidate : headerCandidates) {
      std::string contents = LoadFileContents(candidate);
      if (!contents.empty()) {
        return {candidate, std::move(contents)};
      }
    }
    return {};
  };

  auto [constantsPath, constantsSource] = resolveHeader("MetalConstants.h");
  if (constantsSource.empty()) {
    std::fprintf(stderr, "Metal backend: MetalConstants.h not found\n");
    return {};
  }
  RemoveIncludeLine(kernelSource, "#include \"MetalConstants.h\"");
  RemovePragmaOnce(constantsSource);

  auto [metalMathPath, metalMathSource] = resolveHeader("MetalMath.h");
  if (metalMathSource.empty()) {
    std::fprintf(stderr, "Metal backend: MetalMath.h not found\n");
    return {};
  }
  RemoveIncludeLine(kernelSource, "#include \"MetalMath.h\"");
  RemoveIncludeLine(metalMathSource, "#include \"MetalConstants.h\"");
  RemovePragmaOnce(metalMathSource);

  auto [gpuMathPath, gpuMathSource] = resolveHeader("../GPUMath.h");
  if (gpuMathSource.empty()) {
    std::fprintf(stderr, "Metal backend: GPUMath.h not found for Metal compilation\n");
    return {};
  }
  RemoveIncludeLine(kernelSource, "#include \"../GPUMath.h\"");
  RemoveIncludeLine(gpuMathSource, "#include \"metal/MetalMath.h\"");
  RemovePragmaOnce(gpuMathSource);

  std::ostringstream merged;
  merged << "#line 1 \"" << constantsPath.string() << "\"\n" << constantsSource << "\n";
  merged << "#line 1 \"" << metalMathPath.string() << "\"\n" << metalMathSource << "\n";
  merged << "#line 1 \"" << gpuMathPath.string() << "\"\n" << gpuMathSource << "\n";
  merged << "#line 1 \"" << kernelPath.string() << "\"\n" << kernelSource;

  return merged.str();
}

constexpr size_t KangarooStrideBytes() {
  return static_cast<size_t>(KSIZE) * sizeof(uint64_t);
}

constexpr size_t DistanceStrideBytes() {
  return sizeof(uint64_t) * 2ULL;
}

constexpr size_t PointStrideBytes() {
  return sizeof(uint64_t) * 4ULL;
}

constexpr size_t PrimeBytes() {
  return sizeof(uint64_t) * 4ULL;
}

constexpr size_t DpBufferBytes(uint32_t maxFound) {
  return static_cast<size_t>(maxFound) * ITEM_SIZE + sizeof(uint32_t);
}

}  // namespace

struct MetalImpl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  id<MTLFunction> function = nil;
  id<MTLComputePipelineState> pipeline = nil;
  id<MTLBuffer> kangaroos = nil;
  id<MTLBuffer> jumpDistances = nil;
  id<MTLBuffer> jumpPx = nil;
  id<MTLBuffer> jumpPy = nil;
  id<MTLBuffer> dpItems = nil;
  id<MTLBuffer> dpCount = nil;
  id<MTLBuffer> prime = nil;

  uint32_t totalKangaroos = 0;
  uint32_t totalThreads = 0;
  size_t kangarooBytes = 0;
  size_t dpItemsBytes = 0;

  GpuConfig config{};
};

MetalBackend::MetalBackend() : impl_(nullptr) {}

MetalBackend::~MetalBackend() { shutdown(); }

bool MetalBackend::init() {
  if (impl_) {
    return true;
  }

  auto* impl = new MetalImpl();
  impl->device = RetainIfNeeded(MTLCreateSystemDefaultDevice());
  if (!impl->device) {
    std::fprintf(stderr, "Metal backend: no compatible GPU found\n");
    delete impl;
    return false;
  }

  impl->queue = RetainIfNeeded([impl->device newCommandQueue]);
  if (!impl->queue) {
    std::fprintf(stderr, "Metal backend: failed to create command queue\n");
    shutdown();
    delete impl;
    return false;
  }

  std::string source = LoadKernelSource();
  if (source.empty()) {
    ReleaseIfNeeded(impl->queue);
    ReleaseIfNeeded(impl->device);
    delete impl;
    return false;
  }

  NSString* nsSource = [[NSString alloc] initWithBytes:source.data()
                                                length:source.size()
                                              encoding:NSUTF8StringEncoding];
  if (!nsSource) {
    std::fprintf(stderr, "Metal backend: kernel source encoding failure\n");
    ReleaseIfNeeded(impl->queue);
    ReleaseIfNeeded(impl->device);
    delete impl;
    return false;
  }

  MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
#if defined(GPU_BACKEND_METAL)
  NSMutableDictionary<NSString*, NSNumber*>* macros = [NSMutableDictionary dictionary];
  macros[@"GPU_BACKEND_METAL"] = @1;
#if defined(USE_SYMMETRY)
  macros[@"USE_SYMMETRY"] = @1;
#endif
  options.preprocessorMacros = macros;
#endif
  NSError* error = nil;
  impl->library = RetainIfNeeded([impl->device newLibraryWithSource:nsSource
                                                             options:options
                                                               error:&error]);
  ReleaseIfNeeded(options);
  ReleaseIfNeeded(nsSource);

  if (!impl->library) {
    const char* desc = error ? [[error localizedDescription] UTF8String] : "unknown error";
    std::fprintf(stderr, "Metal backend: failed to compile library (%s)\n", desc);
    ReleaseIfNeeded(impl->queue);
    ReleaseIfNeeded(impl->device);
    delete impl;
    return false;
  }

  impl->function = RetainIfNeeded([impl->library newFunctionWithName:@"comp_kangaroos"]);
  if (!impl->function) {
    std::fprintf(stderr, "Metal backend: kernel 'comp_kangaroos' not found\n");
    shutdown();
    ReleaseIfNeeded(impl->library);
    ReleaseIfNeeded(impl->queue);
    ReleaseIfNeeded(impl->device);
    delete impl;
    return false;
  }

  NSError* pipelineError = nil;
  impl->pipeline = RetainIfNeeded([impl->device newComputePipelineStateWithFunction:impl->function
                                                                              error:&pipelineError]);
  if (!impl->pipeline) {
    const char* desc = pipelineError ? [[pipelineError localizedDescription] UTF8String] : "unknown error";
    std::fprintf(stderr, "Metal backend: failed to create pipeline (%s)\n", desc);
    ReleaseIfNeeded(impl->function);
    ReleaseIfNeeded(impl->library);
    ReleaseIfNeeded(impl->queue);
    ReleaseIfNeeded(impl->device);
    delete impl;
    return false;
  }

  impl_ = impl;
  return true;
}

bool MetalBackend::allocate(const Buffers& buffers, const GpuConfig& config) {
  if (!impl_ && !init()) {
    return false;
  }
  if (!impl_ || !impl_->device) {
    return false;
  }

  impl_->config = config;
  impl_->totalKangaroos = buffers.totalKangaroos;
  impl_->totalThreads = config.groups * config.threadsPerGroup;
  impl_->kangarooBytes = static_cast<size_t>(buffers.totalKangaroos) * KangarooStrideBytes();
  impl_->dpItemsBytes = DpBufferBytes(config.maxFound);

  ReleaseIfNeeded(impl_->kangaroos);
  ReleaseIfNeeded(impl_->jumpDistances);
  ReleaseIfNeeded(impl_->jumpPx);
  ReleaseIfNeeded(impl_->jumpPy);
  ReleaseIfNeeded(impl_->dpItems);
  ReleaseIfNeeded(impl_->dpCount);
  ReleaseIfNeeded(impl_->prime);

  impl_->kangaroos = RetainIfNeeded([impl_->device newBufferWithLength:impl_->kangarooBytes
                                                               options:MTLResourceStorageModeShared]);
  impl_->jumpDistances = RetainIfNeeded([impl_->device newBufferWithLength:static_cast<NSUInteger>(config.jumpCount) * DistanceStrideBytes()
                                                                       options:MTLResourceStorageModeShared]);
  impl_->jumpPx = RetainIfNeeded([impl_->device newBufferWithLength:static_cast<NSUInteger>(config.jumpCount) * PointStrideBytes()
                                                                options:MTLResourceStorageModeShared]);
  impl_->jumpPy = RetainIfNeeded([impl_->device newBufferWithLength:static_cast<NSUInteger>(config.jumpCount) * PointStrideBytes()
                                                                options:MTLResourceStorageModeShared]);
  impl_->dpItems = RetainIfNeeded([impl_->device newBufferWithLength:impl_->dpItemsBytes
                                                             options:MTLResourceStorageModeShared]);
  impl_->dpCount = RetainIfNeeded([impl_->device newBufferWithLength:sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared]);
  impl_->prime = RetainIfNeeded([impl_->device newBufferWithLength:PrimeBytes()
                                                            options:MTLResourceStorageModeShared]);

  if (!impl_->kangaroos || !impl_->jumpDistances || !impl_->jumpPx || !impl_->jumpPy || !impl_->dpItems || !impl_->dpCount || !impl_->prime) {
    std::fprintf(stderr, "Metal backend: failed to allocate buffers\n");
    return false;
  }

  if (buffers.prime) {
    std::memcpy([impl_->prime contents], buffers.prime, PrimeBytes());
  }

  resetDPCount();
  return true;
}

bool MetalBackend::uploadJumps(const void* distances, const void* px, const void* py, uint32_t count) {
  if (!impl_ || !impl_->jumpDistances || !impl_->jumpPx || !impl_->jumpPy) {
    return false;
  }
  if (count > impl_->config.jumpCount) {
    std::fprintf(stderr, "Metal backend: uploadJumps count exceeds allocation (%u>%u)\n", count, impl_->config.jumpCount);
    return false;
  }

  size_t distBytes = static_cast<size_t>(count) * DistanceStrideBytes();
  size_t pointBytes = static_cast<size_t>(count) * PointStrideBytes();

  if (distances) {
    std::memcpy([impl_->jumpDistances contents], distances, distBytes);
  }
  if (px) {
    std::memcpy([impl_->jumpPx contents], px, pointBytes);
  }
  if (py) {
    std::memcpy([impl_->jumpPy contents], py, pointBytes);
  }

  return true;
}

bool MetalBackend::uploadKangaroos(const void* host, size_t bytes) {
  if (!impl_ || !impl_->kangaroos) {
    return false;
  }
  if (!host || bytes > impl_->kangarooBytes) {
    std::fprintf(stderr, "Metal backend: uploadKangaroos invalid size (%zu>%zu)\n", bytes, impl_->kangarooBytes);
    return false;
  }

  std::memcpy([impl_->kangaroos contents], host, bytes);
  return true;
}

bool MetalBackend::runOnce() {
  if (!impl_ || !impl_->pipeline || !impl_->queue) {
    return false;
  }

  id<MTLCommandBuffer> command = [impl_->queue commandBuffer];
  if (!command) {
    return false;
  }

  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (!encoder) {
    return false;
  }

  [encoder setComputePipelineState:impl_->pipeline];
  [encoder setBuffer:impl_->kangaroos offset:0 atIndex:0];
  [encoder setBuffer:impl_->dpCount offset:0 atIndex:1];
  [encoder setBuffer:impl_->dpItems offset:0 atIndex:2];
  [encoder setBuffer:impl_->jumpDistances offset:0 atIndex:3];
  [encoder setBuffer:impl_->jumpPx offset:0 atIndex:4];
  [encoder setBuffer:impl_->jumpPy offset:0 atIndex:5];
  [encoder setBuffer:impl_->prime offset:0 atIndex:12];

  uint32_t maxFound = impl_->config.maxFound;
  uint64_t dpMask = impl_->config.dpMask;
  uint32_t nbRun = impl_->config.iterationsPerDispatch;
  uint32_t threadsPerGroup = impl_->config.threadsPerGroup;
  uint32_t totalThreads = impl_->totalThreads;
  uint32_t totalKangaroos = impl_->totalKangaroos;
  uint32_t jumpCount = impl_->config.jumpCount;

  [encoder setBytes:&maxFound length:sizeof(maxFound) atIndex:6];
  [encoder setBytes:&dpMask length:sizeof(dpMask) atIndex:7];
  [encoder setBytes:&nbRun length:sizeof(nbRun) atIndex:8];
  [encoder setBytes:&threadsPerGroup length:sizeof(threadsPerGroup) atIndex:9];
  [encoder setBytes:&totalThreads length:sizeof(totalThreads) atIndex:10];
  [encoder setBytes:&totalKangaroos length:sizeof(totalKangaroos) atIndex:11];
  [encoder setBytes:&jumpCount length:sizeof(jumpCount) atIndex:13];

  MTLSize threadsPerThreadgroup = MTLSizeMake(impl_->config.threadsPerGroup, 1, 1);
  MTLSize threadgrid = MTLSizeMake(impl_->totalThreads, 1, 1);
  [encoder dispatchThreads:threadgrid threadsPerThreadgroup:threadsPerThreadgroup];
  [encoder endEncoding];

  [command commit];
  [command waitUntilCompleted];

  return true;
}

bool MetalBackend::readDP(void* hostDp, size_t bytes, uint32_t& outCount) {
  outCount = 0;
  if (!impl_ || !impl_->dpItems || !impl_->dpCount) {
    return false;
  }

  const uint32_t* countPtr = static_cast<const uint32_t*>([impl_->dpCount contents]);
  if (!countPtr) {
    return false;
  }
  uint32_t count = *countPtr;
  if (count > impl_->config.maxFound) {
    count = impl_->config.maxFound;
  }
  size_t copyBytes = std::min(bytes, impl_->dpItemsBytes);
  if (hostDp && copyBytes > 0) {
    std::memcpy(hostDp, [impl_->dpItems contents], copyBytes);
  }
  outCount = count;
  return true;
}

bool MetalBackend::downloadKangaroos(void* host, size_t bytes) {
  if (!impl_ || !impl_->kangaroos || !host) {
    return false;
  }
  if (bytes > impl_->kangarooBytes) {
    return false;
  }
  std::memcpy(host, [impl_->kangaroos contents], bytes);
  return true;
}

void MetalBackend::resetDPCount() {
  if (!impl_ || !impl_->dpCount) {
    return;
  }
  uint32_t zero = 0;
  std::memcpy([impl_->dpCount contents], &zero, sizeof(zero));
  if (impl_->dpItems) {
    std::memset([impl_->dpItems contents], 0, std::min(static_cast<size_t>(sizeof(uint32_t)), impl_->dpItemsBytes));
  }
}

void MetalBackend::shutdown() {
  if (!impl_) {
    return;
  }

  ReleaseIfNeeded(impl_->prime);
  ReleaseIfNeeded(impl_->dpCount);
  ReleaseIfNeeded(impl_->dpItems);
  ReleaseIfNeeded(impl_->jumpPy);
  ReleaseIfNeeded(impl_->jumpPx);
  ReleaseIfNeeded(impl_->jumpDistances);
  ReleaseIfNeeded(impl_->kangaroos);
  ReleaseIfNeeded(impl_->pipeline);
  ReleaseIfNeeded(impl_->function);
  ReleaseIfNeeded(impl_->library);
  ReleaseIfNeeded(impl_->queue);
  ReleaseIfNeeded(impl_->device);

  delete impl_;
  impl_ = nullptr;
}

#else

MetalBackend::MetalBackend() : impl_(nullptr) {}
MetalBackend::~MetalBackend() = default;
bool MetalBackend::init() { return false; }
bool MetalBackend::allocate(const Buffers&, const GpuConfig&) { return false; }
bool MetalBackend::uploadJumps(const void*, const void*, const void*, uint32_t) { return false; }
bool MetalBackend::uploadKangaroos(const void*, size_t) { return false; }
bool MetalBackend::runOnce() { return false; }
bool MetalBackend::readDP(void*, size_t, uint32_t&) { return false; }
bool MetalBackend::downloadKangaroos(void*, size_t) { return false; }
void MetalBackend::resetDPCount() {}
void MetalBackend::shutdown() {}

#endif  // GPU_BACKEND_METAL
