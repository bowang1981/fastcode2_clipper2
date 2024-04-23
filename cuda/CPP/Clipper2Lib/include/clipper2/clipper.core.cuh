#ifndef CLIPPER_CORE_CUH_
#define CLIPPER_CORE_CUH_
#include "clipper.core.h"

namespace Clipper2Lib {
struct cuPoint64 {
	int64_t x;
	int64_t y;
};

struct cuRect64 {
  int64_t left;
  int64_t top;
  int64_t right;
  int64_t bottom;
 // __device__ __host__ bool intersects(const cuRect64& rect );
  // __device__ __host__ bool contains(const cuRect64& rect );
};

struct cuPath64 {
	__host__ cuPath64();
	__host__ void init(const Path64& path);
	__host__ void init(int sz);
	__host__ ~cuPath64();
	__host__ Path64 toPath64() const;
//	__device__ __host__ cuRect64 getBoundary();
	cuPoint64* points;
	int size;
};

struct cuPaths64 {
	__host__ cuPaths64();
	__host__ void init(const Paths64& paths);
	__host__ void init(int sz);
	__host__ void initShapeOnly(const Paths64& paths, int factor);
	__host__ Paths64 toPaths64() const;
	__host__ ~cuPaths64();
	cuPath64* cupaths;
	int size;
};

}

#endif
