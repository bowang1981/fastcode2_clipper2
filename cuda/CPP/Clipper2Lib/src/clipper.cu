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



///////////////////////////////////////////////////////////////////////////////////////////////////////

cuPath64::cuPath64()
{
}

void cuPath64::init(const Path64& path)
{
	size = path.size();
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
	size = sz;
    cudaError_t err = cudaMallocManaged(&points,size*sizeof(cuPoint64));
    if (err != cudaSuccess)
    {
        std::cout << "Memory allocation failed"<<std::endl;
    }
}



cuPath64::~cuPath64()
{
	cudaFree(points);
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



}


