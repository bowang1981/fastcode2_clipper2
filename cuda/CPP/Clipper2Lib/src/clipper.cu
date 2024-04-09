#include "clipper2/clipper.cuh"
#include <iostream>
__global__ void test_print() {
	// print("Just test the cmake usage on CUDA!");
}

void wrap_test_print() {
	test_print<<<1, 1>>>();
	std::cout << "CUDA kernel launched!";
	return;
}
