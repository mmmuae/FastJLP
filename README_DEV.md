# Kangaroo Metal Backend Developer Guide

This document explains how to build the experimental Metal backend, run the
available unit tests, and iterate on the GPU implementation on macOS systems.

## Prerequisites

* macOS 13 or newer with Command Line Tools or Xcode installed.
* A Metal compatible GPU (Apple Silicon such as M1/M2 or an AMD discrete GPU).
* CMake 3.20+ and Ninja (recommended) or Make.

## Configure the project

The CMake build exposes toggles for CUDA and Metal backends.  On macOS the
Metal backend is enabled by default.  To generate a build directory that targets
Metal only:

```bash
cmake -S . -B build -G Ninja \
  -DGPU_BACKEND_METAL=ON \
  -DGPU_BACKEND_CUDA=OFF \
  -DBUILD_METAL_TESTS=ON
```

The configure step validates that an Objective-C++ compiler and required Apple
frameworks are available.  Failing this check usually means the Command Line
Tools or Xcode are missing.

Define `WITHGPU` when configuring if you want the main solver to build the GPU
entry points in addition to the Metal runtime:

```bash
cmake -S . -B build -G Ninja \
  -DGPU_BACKEND_METAL=ON \
  -DGPU_BACKEND_CUDA=OFF \
  -DBUILD_METAL_TESTS=ON \
  -DCMAKE_CXX_FLAGS="-DWITHGPU"
```

## Build targets

Build the main binary and Metal unit tests with:

```bash
cmake --build build
```

This produces the following executables in `build/`:

* `kangaroo` – main solver with Metal backend support.
* `MetalPackingTests` – deterministic round-trip tests for the kangaroo packing
  helpers.
* `MetalDistinguishedPointTests` – deterministic round-trip tests for the
  distinguished-point helpers.

## Running the unit tests

Execute the Metal unit tests directly from the build directory:

```bash
./build/MetalPackingTests
./build/MetalDistinguishedPointTests
```

Both tests must pass without warnings.  By default the build enables
`-Wall -Wextra -Wpedantic`; you can add `-Werror` by configuring CMake with
`-DWARNINGS_AS_ERRORS=ON` if you want warnings to break the build locally.

## Metal smoke test

A lightweight Objective-C++ smoke test exists in `tests/MetalBackendSmoke.mm` to
validate runtime integration.  Build and run it with Clang when the Apple Metal
SDK is available:

```bash
clang++ tests/MetalBackendSmoke.mm -fobjc-arc -std=c++17 -O0 -I. \
  -framework Foundation -framework Metal -framework MetalKit \
  -DGPU_BACKEND_METAL -o MetalBackendSmoke
./MetalBackendSmoke
```

## Acceptance testing

The solver's `-check` mode now performs a CPU ↔ GPU parity sweep for the Metal
backend.  Build with `-DWITHGPU` and run the following command to verify that a
single GPU dispatch matches the CPU step exactly:

```bash
./build/kangaroo -gpu --gpu-backend=metal -check
```

The run prints `CPU/GPU ok` when the herd state and distinguished points match.

### Remaining roadmap

1. **DP format parity:** capture the first few distinguished points produced by
   the GPU and CPU for a small range to ensure exact matches.
2. **Instant find scenario:** reproduce the fast reference solve to confirm the
   Metal backend recovers keys within the expected latency envelope.
3. **Throughput sanity:** stress larger ranges with high `dpBits` values to
   monitor ring occupancy and verify no distinguished points are dropped.

Document findings in `DOC/TESTING_METAL.md` as the coverage expands.
