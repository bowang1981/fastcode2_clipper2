#ifndef CLIPPER_CORE_CUH_
#define CLIPPER_CORE_CUH_
#include "clipper.core.h"

namespace Clipper2Lib {

__device__ int64_t d2i(double x);
struct cuPoint64 {
	int64_t x;
	int64_t y;
	__host__ __device__ cuPoint64(int64_t x1, int64_t y1);

};

__device__ cuPoint64 fromDouble(double x1, double y1);

struct cuPointD {
	double x;
	double y;
	__host__ __device__ cuPointD(double x1, double y1);
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
	__host__ __device__ void push_back(int64_t x, int64_t y);
	__host__ __device__ void append(cuPoint64 pt);
	__device__ void appendD(cuPointD pt);
//	__device__ __host__ cuRect64 getBoundary();
	cuPoint64* points;
	int size;
	int capacity;
};

struct cuPathD {
	__host__ cuPathD();
	__host__ void init(int sz);
	__host__ __device__ void push_back(double x, double y);
	cuPointD* points;
	int size;
	int capacity;
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

__device__ void Append(cuPath64& input, int64_t x, int64_t y);
__host__ __device__ double CrossProduct(const cuPointD& pt1, const cuPointD& pt2,
		const cuPointD& pt3);
__host__ __device__ double CrossProduct(const cuPoint64& pt1, const cuPoint64& pt2,
		const cuPoint64& pt3);
__host__ __device__ double DotProduct(const cuPointD& vec1, const cuPointD& vec2);
__host__ __device__ double CrossProduct(const cuPointD& vec1, const cuPointD& vec2);
}

#endif
