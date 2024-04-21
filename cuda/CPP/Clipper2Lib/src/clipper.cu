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

// TODO: to be checked ==========
cuOutPt2::cuOutPt2(){
}
cuOutPt2 cuOutPt2::init(const OutPt2 &outpt2){
    if (outpt2.edge == nullptr){
        return cuOutPt2();
    }

    pt.x = outpt2.pt.x;
    pt.y = outpt2.pt.y;
    owner_idx = outpt2.owner_idx;
    edge = new cuOutPt2List();
    edge->init(outpt2.edge);
    next =  init(outpt2.next);
    prev =  init(outpt2.prev);
    return *this;
}


cuOutPt2List::cuOutPt2List(){
}
void cuOutPt2List::init(const OutPt2List &outpt2list){
    size = outpt2list.size();
    cudaError_t err = cudaMallocManaged(&list,size*sizeof(cuOutPt2));
    if (err != cudaSuccess)
    {
        std::cout << "Memory allocation failed"<<std::endl;
    }
    for(size_t i = 0;i<size;++i){
        list[i] = list[i].init(outpt2list[i]);
    }
}

void cuOutPt2List::init(int sz){
    size = sz;
    cudaError_t err = cudaMallocManaged(&list,size*sizeof(cuOutPt2));
    if (err != cudaSuccess)
    {
        std::cout << "Memory allocation failed"<<std::endl;
    }
}

cuOutPt2List::~cuOutPt2List(){
    cudaFree(list);
}
// ================


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


