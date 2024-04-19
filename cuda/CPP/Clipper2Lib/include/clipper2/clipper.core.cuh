#ifndef CLIPPER_CORE_CUH_
#define CLIPPER_CORE_CUH_
#include "clipper.core.h"

namespace Clipper2Lib {
struct cuPoint64 {
	int64_t x;
	int64_t y;
};

struct cuPath64 {
	__host__ cuPath64();
	__host__ void init(const Path64& path);
	__host__ ~cuPath64();
	cuPoint64* points;
	int size;
};

struct cuPaths64 {
	__host__ cuPaths64();
	__host__ void init(const Paths64& paths);
	__host__ ~cuPaths64();
	cuPath64* cupaths;
	int size;
};

}

#endif
