#include "clipper2/clipper.cuh"
#include "clipper2/clipper.core.cuh"
#include "clipper2/clipper.rectclip.cuh"
#include <iostream>
#include <vector>

#include "../../Utils/Timer.h"



namespace Clipper2Lib {
__global__ void test_print() {
	// print("Just test the cmake usage on CUDA!");
}

void wrap_test_print() {
	test_print<<<1, 1>>>();
	std::cout << "CUDA kernel launched!" << std::endl;
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

void cuPath64::init(const Path64& path, cuPoint64* start)
{
	size = path.size();
	capacity = size;
	/*{
    cudaError_t err = cudaMallocManaged(&points,size*sizeof(cuPoint64));
		if (err != cudaSuccess)
		{
			std::cout << "Memory allocation failed"<<std::endl;
		}
	}*/
	points = start;

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

void cuPath64::append(cuPoint64 pt)
{
	points[size].x = pt.x;
	points[size].y = pt.y;
	size = size + 1;
}
void cuPath64::appendD(cuPointD pt)
{
	points[size].x = d2i(pt.x);
	points[size].y = d2i(pt.y);
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

void cuPath64::init(int sz, cuPoint64* start)
{
	size = 0;
	capacity = sz;
	points = start;
	/*
    cudaError_t err = cudaMallocManaged(&points,sz*sizeof(cuPoint64));
    if (err != cudaSuccess)
    {
        std::cout << "Memory allocation failed"<<std::endl;
    }*/
}



cuPath64::~cuPath64()
{
	// cudaFree(points);
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
	total_points = 0;
	for (auto path : paths) {
		total_points += path.size();
	}
    cudaError_t err = cudaMallocManaged(&cupaths, size*sizeof(cuPath64));
    if (err != cudaSuccess)
    {
        std::cout << "Memory allocation failed"<<std::endl;
    }
    cudaMallocManaged(&allpoints, total_points * sizeof(cuPoint64));

    int offset = 0;
    for(size_t i = 0;i<size;++i){
    	cupaths[i].init(paths[i], allpoints + offset);
    	offset = offset + paths[i].size();
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
	total_points = 0;
	for (auto path : paths) {
		total_points += path.size();
	}
	total_points = total_points * factor;
    cudaError_t err = cudaMallocManaged(&cupaths, size*sizeof(cuPath64));
    if (err != cudaSuccess)
    {
        std::cout << "Memory allocation failed"<<std::endl;
    }
    cudaMallocManaged(&allpoints, total_points * sizeof(cuPoint64));
    int offset = 0;
    for(size_t i = 0;i<size;++i){
    	int sz1 = factor * paths[i].size();
    	cupaths[i].init(sz1, allpoints + offset);
    	offset = offset + sz1;
    }

}

/*
void cuPaths64::init(int sz)
{
	size = sz;
    cudaError_t err = cudaMallocManaged(&cupaths, size*sizeof(cuPath64));
    if (err != cudaSuccess)
    {
        std::cout << "Memory allocation failed"<<std::endl;
    }
}
*/

cuPaths64::~cuPaths64(){
	cudaFree(cupaths);
	cudaFree(allpoints);
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

__host__ __device__ double CrossProduct(const cuPointD& vec1, const cuPointD& vec2)
{
  return static_cast<double>(vec1.y * vec2.x) - static_cast<double>(vec2.y * vec1.x);
}
__host__ __device__ double DotProduct(const cuPointD& vec1, const cuPointD& vec2)
{
  return (vec1.x * vec2.x) + (vec1.y * vec2.y);
}


/****************************************************************************************/
// Some testing codes for performance test.

void test_cuda_copy(int pctn) {
	cuPath64* paths, *dev_paths;
	paths = (cuPath64*)malloc(pctn * sizeof(cuPath64));
	cudaMalloc(&dev_paths, pctn * sizeof(cuPath64));
	cuPoint64* pts, *dev_pts;
	pts = (cuPoint64*)malloc(pctn* 100 * sizeof(cuPoint64));
	cudaMalloc(&dev_pts, pctn * 100 * sizeof(cuPoint64));
	{
		Timer t;
	cudaMemcpy(dev_paths, paths, pctn * sizeof(cuPath64), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pts, pts, pctn *100* sizeof(cuPath64), cudaMemcpyHostToDevice);
	std::cout << "Copy to Device[ " << pctn << "]"
			  << t.elapsed_str() << std::endl;
	}
	{
		Timer t;
	cudaMemcpy(paths, dev_paths, pctn * sizeof(cuPath64), cudaMemcpyDeviceToHost);
	cudaMemcpy(pts, dev_pts, pctn *100* sizeof(cuPath64), cudaMemcpyDeviceToHost);
	std::cout << "Copy to Host[ " << pctn << "]"
			  << t.elapsed_str() << std::endl;
	}
}


void test_convert_performance(const Paths64& input)
{
	{
		Timer t1;
		cuPaths64* paths;
		cudaMallocManaged(&paths, sizeof(cuPaths64));
		paths->init(input);

		std::cout << "Convert to CUDA format[ " << input.size() << "]"
				  << t1.elapsed_str() << std::endl;
		{
			Timer t2;
			Paths64 output = paths->toPaths64();
			std::cout << "Convert to Clipper format[ " << input.size() << "]"
					  << t2.elapsed_str() << std::endl;
		}
	}

	{
		Timer t1;
		cuPaths64* res;
		cudaMallocManaged(&res, sizeof(cuPaths64));
		res->initShapeOnly(input, 4);
		std::cout << "Convert to CUDA format Shape only[ " << input.size() << "]"
				  << t1.elapsed_str() << std::endl;
	}

	test_cuda_copy(10000);
	test_cuda_copy(100000);
	test_cuda_copy(1000000);
}









}


