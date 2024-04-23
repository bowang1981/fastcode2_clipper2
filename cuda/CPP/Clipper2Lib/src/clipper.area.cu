#include "clipper2/clipper.area.cuh"
#include "clipper2/clipper.cuh"
#include "clipper2/clipper.core.cuh"

namespace Clipper2Lib {

/**********************************************************************************************************************
////////// Area Calculation related functions //////////////////////////////
***********************************************************************************************************************/

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