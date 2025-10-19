#ifndef GPU_METAL_CONSTANTS_H
#define GPU_METAL_CONSTANTS_H

#ifdef GPU_BACKEND_METAL

#ifdef __METAL_VERSION__

#ifndef NB_JUMP
#define NB_JUMP 32
#endif

#ifndef GPU_GRP_SIZE
#define GPU_GRP_SIZE 128
#endif

#ifndef NB_RUN
#define NB_RUN 64
#endif

#ifndef TAME
#define TAME 0
#endif

#ifndef WILD
#define WILD 1
#endif

#else  // __METAL_VERSION__

#include "../../Constants.h"

static_assert(NB_JUMP == 32, "Metal backend constants must match Constants.h (NB_JUMP)");
static_assert(GPU_GRP_SIZE == 128, "Metal backend constants must match Constants.h (GPU_GRP_SIZE)");
static_assert(NB_RUN == 64, "Metal backend constants must match Constants.h (NB_RUN)");
static_assert(TAME == 0, "Metal backend constants must match Constants.h (TAME)");
static_assert(WILD == 1, "Metal backend constants must match Constants.h (WILD)");

#endif  // __METAL_VERSION__

#endif  // GPU_BACKEND_METAL

#endif  // GPU_METAL_CONSTANTS_H

