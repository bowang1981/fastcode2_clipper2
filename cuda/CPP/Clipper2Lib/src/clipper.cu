#include "clipper2/clipper.cuh"
#include "clipper2/clipper.core.cuh"
#include <iostream>
#include <vector>
__global__ void test_print() {
	// print("Just test the cmake usage on CUDA!");
}

namespace Clipper2Lib {

void wrap_test_print() {
	test_print<<<1, 1>>>();
	std::cout << "CUDA kernel launched!";
	return;
}



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

cuPaths64::~cuPaths64(){
	cudaFree(cupaths);
}



}


