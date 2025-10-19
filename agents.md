# Agent.md — Port JLP Kangaroo to Metal (surgical, minimal edits)

Goal: keep 90%+ of JeanLucPons/Kangaroo C++ code unchanged. Add a **pluggable GPU backend** and implement a **Metal** backend (C++ via metal-cpp / Obj-C++) that mirrors CUDA’s contract. Do **not** change algorithmic logic, DP format, distance math, or hashtable behavior.

---

## 0) Repo & branch
- Base: https://github.com/JeanLucPons/Kangaroo (upstream main).
- Create a feature branch: `feat/metal-backend`.

---

## 1) Add a tiny GPU backend interface

**Add:** `GPU/Backend.h`
```cpp
#pragma once
#include <cstdint>
struct GpuConfig {
  uint32_t threadsPerGroup;
  uint32_t groups;
  uint32_t iterationsPerDispatch;
  uint32_t jumpCount; // n = floor(log2(sqrt(range)))+1
  uint64_t dpMask;    // 0 => store all steps
  uint32_t maxFound;  // capacity of DP ring (items)
};
struct Buffers {
  void* kangaroos;   // device buffer (struct matches JLP’s GPU state)
  void* jumpDist;    // device buffer: ulong[4]*n
  void* jumpPx;      // device buffer: uint256_t*n
  void* jumpPy;      // device buffer: uint256_t*n
  void* dpItems;     // device buffer: uint32_t[ITEM_SIZE32*maxFound+1]
  void* prime;       // device buffer: uint256_t
  void* dpCount;     // device buffer: atomic uint
  uint32_t totalKangaroos;
};
class IGpuBackend {
public:
  virtual ~IGpuBackend() {}
  virtual bool init() = 0;
  virtual bool allocate(const Buffers& h, const GpuConfig& c) = 0;
  virtual bool uploadJumps(const void* d, const void* px, const void* py, uint32_t n) = 0;
  virtual bool uploadKangaroos(const void* host, size_t bytes) = 0;
  virtual bool runOnce() = 0;  // one dispatch with config.iterationsPerDispatch
  virtual bool readDP(void* hostDp, size_t bytes, uint32_t& outCount) = 0;
  virtual bool downloadKangaroos(void* host, size_t bytes) = 0; // for -check mode
  virtual void  resetDPCount() = 0;
  virtual void  shutdown() = 0;
};
```

**Add:** `GPU/BackendFactory.h/.cpp`
```cpp
// BackendFactory.h
#pragma once
#include "Backend.h"
enum class BackendKind { CUDA, METAL };
IGpuBackend* CreateBackend(BackendKind);

// BackendFactory.cpp
#include "BackendFactory.h"
#ifdef GPU_BACKEND_CUDA
  IGpuBackend* CreateBackend(BackendKind) { /* return existing CUDA-backed adapter */ }
#endif
#ifdef GPU_BACKEND_METAL
  IGpuBackend* CreateBackend(BackendKind) { /* return new MetalBackend instance */ }
#endif
```
**Modify small call sites** (e.g., where CUDA is created/launched; typically in `Kangaroo.cpp` / `Thread.cpp` or a `GPU/*` wrapper):
- Replace direct CUDA calls with `IGpuBackend`.
- Add a CLI flag `--gpu-backend=metal|cuda` (default: cuda if present, metal on macOS if CUDA missing).

---

## 2) Implement the Metal backend (host runtime)

**Add:** `GPU/metal/MetalBackend.hpp` (C++ header), `GPU/metal/MetalBackend.mm` (Objective-C++) and `GPU/metal/kernels.metal`.

**MetalBackend.hpp (sketch):**
```cpp
#pragma once
#include "../Backend.h"
struct MetalImpl; // PIMPL
class MetalBackend : public IGpuBackend {
public:
  MetalBackend();
  ~MetalBackend() override;
  bool init() override;
  bool allocate(const Buffers&, const GpuConfig&) override;
  bool uploadJumps(const void*, const void*, const void*, uint32_t) override;
  bool uploadKangaroos(const void*, size_t) override;
  bool runOnce() override;
  bool readDP(void* hostDp, size_t bytes, uint32_t& outCount) override;
  bool downloadKangaroos(void* host, size_t bytes) override;
  void  resetDPCount() override;
  void  shutdown() override;
private:
  MetalImpl* impl;
};
```

**MetalBackend.mm (essentials):**
- Create `MTLDevice`, `MTLCommandQueue`, compile `kernels.metal` to `MTLComputePipelineState`.
- Create `MTLBuffer`s for all device buffers in `Buffers` (+ private staging if needed).

- Bind buffers at the exact indices expected by the kernel 

---

## 3) Recreate CUDA kernel in **MSL** (same contract)

**Add:** `GPU/metal/kernels.metal`
- Keep structures **bit-identical** to JLP’s device structs (uint256_t layout, kangaroo state, DP item layout).
- Buffers (by index) must match host binding:

```
[[buffer(0)]] device gpu_kpoint_t *kangaroos
[[buffer(1)]] device atomic_uint *dp_count
[[buffer(2)]] device uint *dp_items           // ITEM_SIZE32*max_found + 1
[[buffer(3)]] constant jump_distance_t *jump_dist
[[buffer(4)]] constant jump_point_t *jump_px
[[buffer(5)]] constant jump_point_t *jump_py
[[buffer(6)]] constant uint &max_found
[[buffer(7)]] constant ulong &dp_mask
[[buffer(8)]] constant uint &nb_run
[[buffer(9)]] constant uint &threads_per_group
[[buffer(10)]] constant uint &total_threads
[[buffer(11)]] constant uint &total_kangaroos
[[buffer(12)]] constant uint256_t &prime
[[buffer(13)]] constant uint &jump_count      // NEW: n = floor(log2(sqrt(range)))+1
```

- Walk rule **must** match CUDA:


- DP emit:


> Keep wide-int helpers: 64-bit add/sub/carry, 64×64→128 via mulhi equivalent, modular reduction folding to 256-bit, modular inverse (binary GCD or Fermat). Preserve semantics, not necessarily identical code.

---

## 4) Build system (CMake) toggle

**Edit:** top-level `CMakeLists.txt`
- Add options:
```cmake
option(GPU_BACKEND_METAL "Build Metal backend" ON)
option(GPU_BACKEND_CUDA  "Build CUDA backend"  OFF) # ON if NV is available
```
- If `GPU_BACKEND_METAL`:
  - Set `add_definitions(-DGPU_BACKEND_METAL)`
  - Add Objective-C++ and Metal sources:
    - `GPU/metal/MetalBackend.mm`
    - `GPU/metal/kernels.metal` (compile with `xcrun -sdk macosx metal` or `add_custom_command`).
  - Link frameworks: `Metal`, `MetalKit`, `Foundation`.
  - Set source file property for `.mm` as Objective-C++.

- If `GPU_BACKEND_CUDA`:
  - Keep existing CUDA targets and `-DGPU_BACKEND_CUDA`.

---

## 5) Host integration (minimal call-site edits)

- Where CUDA was created, do:
```cpp
#include "GPU/BackendFactory.h"
auto backend = CreateBackend(
#ifdef __APPLE__
  BackendKind::METAL
#else
  BackendKind::CUDA
#endif
);
backend->init();
backend->allocate(buffers, config);
backend->uploadJumps(jumpDist, jumpPx, jumpPy, config.jumpCount);
backend->uploadKangaroos(hostK, kangaroosBytes);

// First dispatch can use fewer iterations to reduce latency to first collision:
backend->runOnce();             // nb_run = iterationsPerDispatch
uint32_t dpCount = 0;
backend->readDP(hostDP, dpBytes, dpCount);
// → process DP items with existing CPU code (unchanged).

// Loop: runOnce + readDP + process, until solved/stop.
```

- Do **not** change hashtable, DP processing, or key recovery logic.

---

## 6) Jump table parity (host)

- Pass `jumpCount` to GPU (buffer 13).

---

## 7) Acceptance tests (must pass)

1. **CPU–GPU step parity (-check mode):**

2. **DP format parity:**
   - Run a small range (`2^24`) with `dpBits=8`. Compare the first 10 DP items emitted by GPU vs CPU path for the same seed (X, dist, idx/type). Expect identical items in the same order.



---

## 8) Notes / guardrails

- Keep **struct packing** identical between CPU and GPU (no padding drift).
- Keep **leading-bits** DP test (use high limb) to match the original.
- Do not change hashtable behavior, solution formula, or CPU collision handling.
- MetalGPU code Must be mapped 1:1 with what works for Apple Silicon, Considering It will be 1 GPU per Machine

---

## 9) Deliverables checklist

- [ ] `GPU/Backend.h`, `GPU/BackendFactory.h/.cpp`
- [ ] `GPU/metal/MetalBackend.hpp`, `GPU/metal/MetalBackend.mm`
- [ ] `GPU/metal/kernels.metal`
- [ ] CMake toggles + linkage (Metal frameworks)
- [ ] Minimal edits in existing call sites to use `IGpuBackend`
- [ ] CPU–GPU parity `-check` workflow implemented and passing
- [ ] README_DEV.md: how to build Metal target and run acceptance tests
