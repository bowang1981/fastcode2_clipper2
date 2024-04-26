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
    cuPoint64* points;
    cudaMallocManaged(&points, path.size() * sizeof(cuPoint64));
    if (err != cudaSuccess)
    {
        std::cout << "Memory allocation failed"<<std::endl;
    }
    p1->init(path, points);

	float* res;
	cudaMallocManaged(&res, sizeof(float));

	area<<<1, 10>>>(p1, res);
	cudaDeviceSynchronize();

	if (cnt & 1)
	      *res = *res + static_cast<double>(path[cnt-2].y + path[cnt-1].y) * (path[cnt-2].x - path[cnt-1].x);
	float area1 = (*res) * 0.5;
	cudaFree(res);
	cudaFree(p1);
	cudaFree(points);
	return area1;

}

__global__ void area_paths_kernel(cuPaths64* path, float* res)
{
	// TODO David, please add.

    cuPath64* current_path = path->cupaths;
    size_t cnt = current_path->size;
    if (cnt < 3) return; // Skip paths with less than 3 points
    float a = 0.0;
    

    int total_threads = gridDim.x * blockDim.x;

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    int patch_size = cnt / total_threads;
    int start = id*patch_size;
    int end = start + cnt;
    size_t lastIdx = end-1;
    for(int i = start;  i < cnt && i<end; ++i){
        int j = i == start ? lastIdx: i-1;
        a += (current_path->points[i].x + current_path->points[j].x) * (current_path->points[i].y - current_path->points[j].y) * 0.5;
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

    area_paths_kernel<<<1, 10>>>(cu_paths, res);
    cudaDeviceSynchronize();

    float total_area = *res; // Assuming the area needs to be halved
    cudaFree(res);
    cudaFree(cu_paths);
    return total_area;
	// return 0;
}

}


/*

    inline double Area_OpenMP_With_Clipper2_Area(const Paths64& paths)
    {
        double totalSum = 0.0;
        #pragma omp parallel for reduction(+:totalSum) 
        for (const auto &p: paths) {
            const auto localSum = Area(p);
            totalSum+=localSum;
        }
        return totalSum;
    }


    // In this version, we implement our own area calculation function
    inline double Area_OpenMP(const Paths64& paths)
    {
        double totalSum = 0.0;
        #pragma omp parallel for reduction(+:totalSum) 
        for (const auto &p: paths) {
            double localSum = 0.0;
            const auto pathSize = p.size();
            if(pathSize<3){
                continue;
            }
            const auto lastIdx = pathSize - 1;
            for (size_t i = 0; i < pathSize; ++i) {
                const auto& it1 = p[i];
                const auto& it2 = p[i == 0 ? lastIdx : i - 1];
                localSum += (it1.x + it2.x) * (it1.y - it2.y);
            }
            totalSum += localSum;
        }
        return totalSum/2;
    }

*/
