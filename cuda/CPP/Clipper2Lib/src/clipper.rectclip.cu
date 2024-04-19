
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
__global__ void rectclip_internal(cuPaths64* input, Rect64* rect, cuPaths64* output)
{

}

void rectclip_execute(const Paths64& input, const Rect64& rect, Paths64& output) {
	cuPaths64* paths;
	cudaMallocManaged(&paths, sizeof(cuPaths64));
	paths->init(input);

	cuPaths64* res;
	cudaMallocManaged(&res, sizeof(cuPaths64));

	Rect64* rect1;
	cudaMallocManaged(&rect1, sizeof(Rect64));
	*rect1 = rect;

	rectclip_internal<<<1, 1>>>(paths, rect1, res);

	cudaFree(paths);
	cudaFree(res);
}


}
