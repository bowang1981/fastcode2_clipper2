#include "clipper2/clipper.cuh"
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

struct cuPaths64 {
	__host__ cuPaths64();
	__host__ void init(const Paths64& paths);
	__host__ ~cuPaths64();
	cuPath64* cupaths;
	int size;
};


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

__global__ void area(cuPath64* path, float* res)
{
    size_t cnt = path->size;
    if (cnt < 3) return;
    double a = 0.0;

    {
        int total_threads = gridDim.x * blockDim.x;

        int id = blockIdx.x * blockDim.x + threadIdx.x;

        int patch_size = cnt / total_threads;
        patch_size = (patch_size / 2) * 2 + 2;
        float a1 = 0.0;
        for (int i = id * patch_size; i < cnt && i < (id+1) * patch_size; i = i + 2) {
            int prev = i - 1;
            if (i == 0) prev = cnt - 1;

            a1 += static_cast<double>(path->points[prev].y + path->points[i].y) * (path->points[prev].x - path->points[i].x);
            int next = i + 1;
            a1 += static_cast<double>(path->points[i].y + path->points[next].y) * (path->points[i].x - path->points[next].x);
        }

        a+= a1;
    }
    float res1 = atomicAdd(res, a);

}


float area_single(const Path64& path) {
	int cnt = path.size();
	cuPath64* p1;
    cudaError_t err = cudaMallocManaged(&p1, sizeof(cuPath64));

    if (err != cudaSuccess)
    {
        std::cout << "Memory allocation failed"<<std::endl;
    }
    p1->init(path);

	float* res;
	cudaMallocManaged(&res, sizeof(float));

	area<<<1, 10>>>(p1, res);
	cudaDeviceSynchronize();

	if (cnt & 1)
	      *res = *res + static_cast<double>(path[cnt-2].y + path[cnt-1].y) * (path[cnt-2].x - path[cnt-1].x);
	float area1 = (*res) * 0.5;
	cudaFree(res);
	cudaFree(p1);
	return area1;

}


}


