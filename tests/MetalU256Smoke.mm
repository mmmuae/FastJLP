#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <Metal/Metal.h>

namespace {

std::optional<std::string> LoadFile(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return std::nullopt;
  }
  std::string contents((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
  return contents;
}

std::string BuildKernelSource(const std::string& metalMathSource) {
  std::string src;
  src.reserve(metalMathSource.size() + 512);
  src.append("#include <metal_stdlib>\nusing namespace metal;\n");
  src.append(metalMathSource);
  src.append(R"(
kernel void u256_addsub(device ulong4* out [[buffer(0)]]) {
  uint256_t a = { { 0x0123456789ABCDEFull, 0x0FEDCBA987654321ull,
                    0x1111111111111111ull, 0x2222222222222222ull } };
  uint256_t b = { { 0x9999999999999999ull, 0xAAAAAAAAAAAAAAAAull,
                    0xBBBBBBBBBBBBBBBBull, 0xCCCCCCCCCCCCCCCCull } };
  uint256_t sum = u256_add(a, b);
  uint256_t diff = u256_sub(a, b);
  bool ge = u256_ge(a, b);
  out[0] = ulong4(sum.d[0], sum.d[1], sum.d[2], sum.d[3]);
  out[1] = ulong4(diff.d[0], diff.d[1], diff.d[2], diff.d[3]);
  out[2] = ulong4(ge ? 1ull : 0ull, 0ull, 0ull, 0ull);
}
)");
  return src;
}

std::array<uint64_t, 4> CpuAdd(const std::array<uint64_t, 4>& a,
                               const std::array<uint64_t, 4>& b) {
  std::array<uint64_t, 4> r{};
  unsigned __int128 carry = 0;
  for (size_t i = 0; i < 4; ++i) {
    unsigned __int128 sum = static_cast<unsigned __int128>(a[i]) +
                            static_cast<unsigned __int128>(b[i]) + carry;
    r[i] = static_cast<uint64_t>(sum);
    carry = sum >> 64;
  }
  return r;
}

std::array<uint64_t, 4> CpuSub(const std::array<uint64_t, 4>& a,
                               const std::array<uint64_t, 4>& b) {
  std::array<uint64_t, 4> r{};
  int64_t borrow = 0;
  for (size_t i = 0; i < 4; ++i) {
    unsigned __int128 minuend = static_cast<unsigned __int128>(a[i]);
    unsigned __int128 subtrahend =
        static_cast<unsigned __int128>(b[i]) + static_cast<unsigned __int128>(borrow);
    if (minuend >= subtrahend) {
      r[i] = static_cast<uint64_t>(minuend - subtrahend);
      borrow = 0;
    } else {
      r[i] = static_cast<uint64_t>((minuend + (static_cast<unsigned __int128>(1) << 64)) -
                                   subtrahend);
      borrow = 1;
    }
  }
  return r;
}

bool CpuGe(const std::array<uint64_t, 4>& a, const std::array<uint64_t, 4>& b) {
  for (size_t i = 0; i < 4; ++i) {
    size_t limb = 3 - i;
    if (a[limb] != b[limb]) {
      return a[limb] > b[limb];
    }
  }
  return true;
}

bool RunTest(const std::filesystem::path& headerPath) {
  auto maybeSource = LoadFile(headerPath);
  if (!maybeSource) {
    std::cerr << "MetalU256Smoke: failed to read " << headerPath << "\n";
    return false;
  }

  std::string source = BuildKernelSource(*maybeSource);

  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) {
    std::cerr << "MetalU256Smoke: no Metal device available\n";
    return false;
  }

  MTLCompileOptions* options = [MTLCompileOptions new];
  options.preprocessorMacros = @{ @"GPU_BACKEND_METAL" : @1 };

  NSError* error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                               options:options
                                                 error:&error];
  if (!library) {
    std::cerr << "MetalU256Smoke: library compilation failed: "
              << [[error localizedDescription] UTF8String] << "\n";
    return false;
  }

  id<MTLFunction> function = [library newFunctionWithName:@"u256_addsub"];
  if (!function) {
    std::cerr << "MetalU256Smoke: kernel not found\n";
    return false;
  }

  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
  if (!pipeline) {
    std::cerr << "MetalU256Smoke: pipeline creation failed: "
              << [[error localizedDescription] UTF8String] << "\n";
    return false;
  }

  id<MTLCommandQueue> queue = [device newCommandQueue];
  if (!queue) {
    std::cerr << "MetalU256Smoke: command queue creation failed\n";
    return false;
  }

  const size_t vectorCount = 3;
  const size_t bytesPerVector = sizeof(uint64_t) * 4;
  id<MTLBuffer> buffer = [device newBufferWithLength:vectorCount * bytesPerVector
                                              options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:buffer offset:0 atIndex:0];
  MTLSize threadsPerGrid = MTLSizeMake(1, 1, 1);
  MTLSize threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
  [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
  [encoder endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];

  const auto* raw = static_cast<const uint64_t*>([buffer contents]);
  std::array<uint64_t, 4> gpuSum = {raw[0], raw[1], raw[2], raw[3]};
  std::array<uint64_t, 4> gpuDiff = {raw[4], raw[5], raw[6], raw[7]};
  bool gpuGe = raw[8] != 0;

  std::array<uint64_t, 4> a = {0x0123456789ABCDEFull, 0x0FEDCBA987654321ull,
                               0x1111111111111111ull, 0x2222222222222222ull};
  std::array<uint64_t, 4> b = {0x9999999999999999ull, 0xAAAAAAAAAAAAAAAAull,
                               0xBBBBBBBBBBBBBBBBull, 0xCCCCCCCCCCCCCCCCull};

  auto cpuSum = CpuAdd(a, b);
  auto cpuDiff = CpuSub(a, b);
  bool cpuGe = CpuGe(a, b);

  if (gpuSum != cpuSum) {
    std::cerr << "MetalU256Smoke: sum mismatch\n";
    return false;
  }
  if (gpuDiff != cpuDiff) {
    std::cerr << "MetalU256Smoke: diff mismatch\n";
    return false;
  }
  if (gpuGe != cpuGe) {
    std::cerr << "MetalU256Smoke: comparison mismatch\n";
    return false;
  }

  std::cout << "MetalU256Smoke: u256_add/sub checks passed\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  std::filesystem::path execPath = std::filesystem::absolute(argv[0]);
  auto headerPath = execPath.parent_path() / ".." / "GPU" / "metal" / "MetalMath.h";
  return RunTest(headerPath) ? 0 : 1;
}

#else  // __APPLE__

int main() {
  std::cerr << "MetalU256Smoke: requires macOS with Metal support\n";
  return 0;
}

#endif
