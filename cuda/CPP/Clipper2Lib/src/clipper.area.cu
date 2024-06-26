#include "clipper2/clipper.area.cuh"
#include "clipper2/clipper.cuh"
#include "clipper2/clipper.core.cuh"
#include "../../Utils/Timer.h"

namespace Clipper2Lib {

/**********************************************************************************************************************
////////// Area Calculation related functions //////////////////////////////
***********************************************************************************************************************/

__global__ void area(cuPath64* path, double* res)
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
    double res1 = atomicAdd(res, a);

}


double area_single(const Path64& path) {
	int cnt = path.size();
	cuPath64* p1;
    cudaError_t err = cudaMallocManaged(&p1, sizeof(cuPath64));
    cuPoint64* points;
    cudaMallocManaged(&points, path.size() * sizeof(cuPoint64));
    if (err != cudaSuccess)
    {
        std::cout << "Memory allocation failed"<<std::endl;
    }
    p1->init(path, points);


	double* res;
	cudaMallocManaged(&res, sizeof(double));

	cudaDeviceSynchronize();
    {
        Timer t1;
	    area<<<1, 32>>>(p1, res);
	    cudaDeviceSynchronize();
        std::cout << "CUDA: Kernel Run time: "<< t1.elapsed_str() << std::endl;
    }
	if (cnt & 1)
	      *res = *res + static_cast<double>(path[cnt-2].y + path[cnt-1].y) * (path[cnt-2].x - path[cnt-1].x);
	double area1 = (*res) * 0.5;
	cudaFree(res);
	cudaFree(p1);
	cudaFree(points);
	return area1;

}


__global__ void area_paths_kernel(cuPaths64* input, float* res)
{
	// TODO David, please add.

    int num_threads = gridDim.x * blockDim.x;

    int input_size = input->size;
	int batch = input_size / num_threads;
	int id = blockIdx.x * blockDim.x + threadIdx.x;

    int start = id * batch;
    int end = start + batch;

    double a = 0.0;
    for(int i = start; i<end; ++i){
        cuPath64* path = &input->cupaths[i];
        // area_paths_kernel_single(path, res);
        size_t cnt = path->size;
        if(cnt<3){
            continue;
        }
        float a1 = 0.0;
        int last_idx = cnt - 1;
        float m1 =path->points[0].x + path->points[last_idx].x;
        float m2 = path->points[0].y - path->points[last_idx].y;
        a1 += (m1 * m2);
        for(int i = 1; i<cnt; ++i){
            int j = i-1;
            m1 =path->points[i].x + path->points[j].x;
            m2 = path->points[i].y - path->points[j].y;
            a1 += (m1 * m2);
        }
        a1 /= 2;
        a += a1;
    }
    atomicAdd(res, a);
}

float area_paths(const Paths64& paths)
{
	// TODO: David, please add here
    cuPaths64* cu_paths;
    cudaMallocManaged(&cu_paths, sizeof(cuPaths64));
    cu_paths->init(paths);

    float* res;
    cudaMallocManaged(&res, sizeof(float));

    {
        Timer t;
        area_paths_kernel<<<1, 10>>>(cu_paths, res);
        cudaDeviceSynchronize();
        std::cout << "CUDA: Kernel Run time: "
            << t.elapsed_str() << std::endl;
    }

    float total_area = *res;
    cudaFree(res);
    cudaFree(cu_paths);
    return total_area;
}

}
