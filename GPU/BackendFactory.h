#pragma once

#include "Backend.h"

enum class BackendKind { CUDA, METAL };

IGpuBackend* CreateBackend(BackendKind kind);
const char* BackendName(BackendKind kind);
bool IsBackendAvailable(BackendKind kind);
BackendKind GetDefaultBackend();
