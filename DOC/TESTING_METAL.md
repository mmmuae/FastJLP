# Metal Backend Testing Guide

This document tracks early testing scaffolding for the Metal GPU backend. The focus is on
sanity checks that can run before the full kangaroo kernel is complete so regressions are
caught quickly.

> **Selecting the Metal backend:** When running the main solver with GPU support enabled,
> add `--gpu-backend=metal` alongside `-gpu` to ensure the Metal runtime is used. Metal-only
> builds default to the Metal backend automatically, but the flag is required if CUDA is
> also available in the binary.

## 1. Host/runtime smoke test

```
clang++ tests/MetalBackendSmoke.mm \
  -fobjc-arc -std=c++17 -O0 \
  -I. \
  -framework Foundation -framework Metal -framework MetalKit \
  -DGPU_BACKEND_METAL \
  -o MetalBackendSmoke
./MetalBackendSmoke
```

This verifies that the Objective-C++ runtime plumbing is working end-to-end:

* kernel source discovery succeeds
* buffers allocate and accept data uploads
* command buffer submission completes without errors
* DP ring and kangaroo downloads return the uploaded data (since the kernel is still a stub)

Any failure here indicates a host/runtime regression that should be fixed before debugging
kernel math.

## 2. CPU ↔ GPU parity (-check mode)

Build the solver with `-DWITHGPU` so the GPU entry points are compiled, then run the
following command to exercise the parity workflow:

```
./build/kangaroo -gpu --gpu-backend=metal -check
```

The check performs one GPU dispatch, downloads the herd, steps the same herd on the CPU,
and compares `(X, Y, dist)` for every kangaroo. The run prints `CPU/GPU ok` when the Metal
kernel matches the CPU reference implementation.

## 3. Upcoming parity checks

Promote the remaining workflows once the kernel logic and harness are ready:

1. **Distinguished point format parity** – emit a small number of DP items on both CPU and
   GPU with the same seed and compare serialized records byte-for-byte.
2. **Instant find regression** – solve the known narrow-range puzzle (`≈2^31` span) with
   `dpBits = 0` and ensure a collision appears immediately, matching the CUDA behavior.

Document the command lines and expected timings when each test is automated so they can be
wired into CI for Apple Silicon machines.
