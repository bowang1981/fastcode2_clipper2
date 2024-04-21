
#include "clipper2/clipper.cuh"
#include "clipper2/clipper.core.cuh"
#include <iostream>
#include <vector>
__global__ void test_print() {
	// print("Just test the cmake usage on CUDA!");
}

namespace Clipper2Lib {

/**********************************************************************************************************************
////////// RectClipping related functions //////////////////////////////
***********************************************************************************************************************/
__host__ __device__ bool intersects(const cuRect64& cur, const cuRect64& rect )
{
	if (rect.left > cur.right || cur.left > rect.right || cur.top > rect.bottom || rect.top > cur.bottom) return false;
	return true;
}
__host__ __device__ bool contains(const cuRect64& cur, const cuRect64& rect )
{
	if (rect.top > cur.top && rect.bottom < cur.bottom && rect.left > cur.left && rect.right < cur.right ) return true;
	return false;
}

__host__ __device__ cuRect64 getBoundary(const cuPath64& path)
{
	cuRect64 res;
	res.bottom = path.points[0].y;
	res.top = path.points[0].y;
	res.left = path.points[0].x;
	res.right = path.points[0].y;
	for (size_t i = 1; i < path.size; ++i) {
		res.bottom = max(res.bottom, path.points[i].y);
		res.top = min(res.top, path.points[i].y);
		res.left = min(res.left, path.points[i].x);
		res.right = max(res.right, path.points[i].y);
	}
	return res;
}
__global__ void rectclip_internal(cuPaths64* input, cuRect64* rect, cuPaths64* output)
{

}

// output[i]: 0(no overlap), 1 (inside), 2 (overlap)
__global__ void filter(cuPaths64* input, cuRect64* rect, int* output)
{
	int thread_no = gridDim.x * blockDim.x;
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int ctn = input->size / thread_no + 1;
	for (int i = ctn * id; i < (ctn * (id+1)) && i < input->size; ++i) {
		cuRect64 r1 = getBoundary(input->cupaths[i]);

		if(intersects(*rect, r1) == false) {
			output[i] = 0;
		}
		else if (contains(*rect, r1)) {
			output[i] = 1;
		} else {
			output[i] = 2;
		}
	}

}

void rectclip_execute(const Paths64& input, const Rect64& rect, Paths64& output) {
	cuPaths64* paths;
	cudaMallocManaged(&paths, sizeof(cuPaths64));
	paths->init(input);

	int* filterarr;
	cudaMallocManaged(&filterarr, input.size()*sizeof(int));

	cuRect64* r1;
	cudaMallocManaged(&r1, sizeof(cuRect64));
	r1->top = rect.top;
	r1->bottom = rect.bottom;
	r1->left = rect.left;
	r1->right = rect.right;

	filter<<<1, 32>>>(paths, r1, filterarr);
	cudaDeviceSynchronize();
	Paths64 insides;

	for (int i = 0; i < input.size(); ++i) {
		if (filterarr[i] == 2) {
			output.push_back(input[i]);
		} else if (filterarr[i] == 1) {
			insides.push_back(input[i]);
		}
	}
	///TBD: Now we need to do the clip on the output, and append the insides after that.


	cudaFree(r1);
	cudaFree(filterarr);
	cudaFree(paths);
}


}
