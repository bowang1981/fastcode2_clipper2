#include "clipper2/clipper.cuh"
#include "clipper2/clipper.core.cuh"
#include "clipper2/clipper.rectclip.cuh"
#include <iostream>
#include <vector>



namespace Clipper2Lib {
__global__ void test_print() {
	// print("Just test the cmake usage on CUDA!");
}

void wrap_test_print() {
	test_print<<<1, 1>>>();
	std::cout << "CUDA kernel launched!";
	return;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ int64_t d2i(double x)
{
	return __double2ll_rn(x);
}

cuPointD::cuPointD(double x1, double y1)
{
	x = x1;
	y = y1;
}

cuPoint64::cuPoint64(int64_t x1, int64_t y1)
{
	x = x1;
	y = y1;
}

__device__ cuPoint64 fromDouble(double x1, double y1)
{
	cuPoint64 p(0, 0);
	p.x = d2i(x1);
	p.y = d2i(y1);
	return p;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

cuPath64::cuPath64()
{
}

void cuPath64::init(const Path64& path)
{
	size = path.size();
	capacity = size;
    cudaError_t err = cudaMallocManaged(&points,size*sizeof(cuPoint64));
    if (err != cudaSuccess)
    {
        std::cout << "Memory allocation failed"<<std::endl;
    }

    for(size_t i = 0;i<size;++i){
    	points[i].x = path[i].x;
    	points[i].y = path[i].y;
    }
}

void cuPath64::push_back(int64_t x, int64_t y)
{
	points[size].x = x;
	points[size].y = y;
	size = size + 1;
}

__device__ void Append(cuPath64& input, int64_t x, int64_t y)
{
	input.points[input.size].x = x;
	input.points[input.size].y = y;
	input.size = input.size + 1;
}


Path64 cuPath64::toPath64() const
{
	Path64 p;
	p.reserve(size);
	for (auto i = 0; i < size; ++i) {
		Point64 p1;
		p1.x = points[i].x;
		p1.y = points[i].y;
		p.push_back(p1);
	}
	return p;
}

void cuPath64::init(int sz)
{
	size = 0;
	capacity = sz;
    cudaError_t err = cudaMallocManaged(&points,sz*sizeof(cuPoint64));
    if (err != cudaSuccess)
    {
        std::cout << "Memory allocation failed"<<std::endl;
    }
}



cuPath64::~cuPath64()
{
	cudaFree(points);
}

void cuPathD::init(int sz) {
	capacity = sz;
	size = 0;
    cudaError_t err = cudaMallocManaged(&points,sz*sizeof(cuPointD));
    if (err != cudaSuccess)
    {
        std::cout << "Memory allocation failed"<<std::endl;
    }
}

void cuPathD::push_back(double x, double y)
{
	points[size].x = x;
	points[size].y = y;
	size = size + 1;
}

cuPaths64::cuPaths64() {
}

void cuPaths64::init(const Paths64& paths)
{
	size = paths.size();
    cudaError_t err = cudaMallocManaged(&cupaths, size*sizeof(cuPath64));
    if (err != cudaSuccess)
    {
        std::cout << "Memory allocation failed"<<std::endl;
    }

    for(size_t i = 0;i<size;++i){
    	cupaths[i].init(paths[i]);
    }

}

Paths64 cuPaths64::toPaths64() const
{
	Paths64 res;
	res.reserve(size);
	for (int i = 0; i < size; ++i) {
		Path64 p = cupaths[i].toPath64();
		res.push_back(p);
	}

	return res;
}

void cuPaths64::initShapeOnly(const Paths64& paths, int factor)
{
	size = paths.size();

    cudaError_t err = cudaMallocManaged(&cupaths, size*sizeof(cuPath64));
    if (err != cudaSuccess)
    {
        std::cout << "Memory allocation failed"<<std::endl;
    }

    for(size_t i = 0;i<size;++i){
    	cupaths[i].init(factor * paths[i].size());
    }

}

void cuPaths64::init(int sz)
{
	size = sz;
    cudaError_t err = cudaMallocManaged(&cupaths, size*sizeof(cuPath64));
    if (err != cudaSuccess)
    {
        std::cout << "Memory allocation failed"<<std::endl;
    }
}


cuPaths64::~cuPaths64(){
	cudaFree(cupaths);
}

// common functions



__host__ __device__ double CrossProduct(const cuPoint64& pt1, const cuPoint64& pt2, const cuPoint64& pt3)
  {
	return (static_cast<double>(pt2.x - pt1.x) * static_cast<double>(pt3.y -
      pt2.y) - static_cast<double>(pt2.y - pt1.y) * static_cast<double>(pt3.x - pt2.x));
  }

__host__ __device__ double CrossProduct(const cuPointD& pt1, const cuPointD& pt2,
		const cuPointD& pt3)
  {
	return (static_cast<double>(pt2.x - pt1.x) * static_cast<double>(pt3.y -
      pt2.y) - static_cast<double>(pt2.y - pt1.y) * static_cast<double>(pt3.x - pt2.x));
  }

__host__ __device__ double DotProduct(const cuPointD& vec1, const cuPointD& vec2)
{
  return (vec1.x * vec2.x) + (vec1.y * vec2.y);
}

}


