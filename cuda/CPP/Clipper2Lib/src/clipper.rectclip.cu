
#include "clipper2/clipper.cuh"
#include "clipper2/clipper.core.cuh"
#include "clipper2/clipper.rectclip.cuh"
#include <iostream>
#include <vector>
__global__ void test_print() {
	// print("Just test the cmake usage on CUDA!");
}

namespace Clipper2Lib {

/*
    // TODO: to be checked ======
    class cuOutPt2;
    struct cuOutPt2List{
        __host__ cuOutPt2List();
        __host__ void init(const OutPt2List &outpt2list);
        __host__ void init(int sz);
        __host__ ~cuOutPt2List();
        cuOutPt2* list;
        int size;
    };

    class cuOutPt2 {
        public:
            __host__ cuOutPt2();
            __host__ cuOutPt2 init(const OutPt2 &outpt2);
            cuPoint64 pt;
            size_t owner_idx;
            cuOutPt2List* edge;
            cuOutPt2* next;
            cuOutPt2* prev;
        };
        */

/*
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
}*/
// ================

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


__host__ __device__ bool GetLocation(const cuRect64& rec, const cuPoint64& pt, Location& loc)
  {
    if (pt.x == rec.left && pt.y >= rec.top && pt.y <= rec.bottom)
    {
      loc = Location::Left;
      return false;
    }
    else if (pt.x == rec.right && pt.y >= rec.top && pt.y <= rec.bottom)
    {
      loc = Location::Right;
      return false;
    }
    else if (pt.y == rec.top && pt.x >= rec.left && pt.x <= rec.right)
    {
      loc = Location::Top;
      return false;
    }
    else if (pt.y == rec.bottom && pt.x >= rec.left && pt.x <= rec.right)
    {
      loc = Location::Bottom;
      return false;
    }
    else if (pt.x < rec.left) loc = Location::Left;
    else if (pt.x > rec.right) loc = Location::Right;
    else if (pt.y < rec.top) loc = Location::Top;
    else if (pt.y > rec.bottom) loc = Location::Bottom;
    else loc = Location::Inside;
    return true;
  }

__host__ __device__ Location GetAdjacentLocation(Location loc, bool isClockwise)
  {
    int delta = (isClockwise) ? 1 : 3;
    return static_cast<Location>((static_cast<int>(loc) + delta) % 4);
  }


__host__ __device__ bool HeadingClockwise(Location prev, Location curr)
  {
    return (static_cast<int>(prev) + 1) % 4 == static_cast<int>(curr);
  }

__host__ __device__ bool AreOpposites(Location prev, Location curr)
  {
    return abs(static_cast<int>(prev) - static_cast<int>(curr)) == 2;
  }

__host__ __device__ double CrossProduct(const cuPoint64& pt1, const cuPoint64& pt2, const cuPoint64& pt3)
  {
	return (static_cast<double>(pt2.x - pt1.x) * static_cast<double>(pt3.y -
      pt2.y) - static_cast<double>(pt2.y - pt1.y) * static_cast<double>(pt3.x - pt2.x));
  }

__host__ __device__ bool IsClockwise(Location prev, Location curr,
    const cuPoint64& prev_pt, const cuPoint64& curr_pt, const cuPoint64& rect_mp)
  {
    if (AreOpposites(prev, curr))
      return CrossProduct(prev_pt, rect_mp, curr_pt) < 0;
    else
      return HeadingClockwise(prev, curr);
  }


/*
__host__ __device__ cuOutPt2* UnlinkOp(cuOutPt2* op)
  {
	if (op->next == op) return nullptr;
    op->prev->next = op->next;
    op->next->prev = op->prev;
    return op->next;
  }

__host__ __device__ cuOutPt2* UnlinkOpBack(cuOutPt2* op)
  {
    if (op->next == op) return nullptr;
    op->prev->next = op->next;
    op->next->prev = op->prev;
    return op->prev;
  }
*/
__host__ __device__ uint32_t GetEdgesForPt(const cuPoint64& pt, const cuRect64& rec)
  {
    uint32_t result = 0;
    if (pt.x == rec.left) result = 1;
    else if (pt.x == rec.right) result = 4;
    if (pt.y == rec.top) result += 2;
    else if (pt.y == rec.bottom) result += 8;
    return result;
  }

__host__ __device__ bool IsHeadingClockwise(const cuPoint64& pt1, const cuPoint64& pt2, int edgeIdx)
  {
    switch (edgeIdx)
    {
    case 0: return pt2.y < pt1.y;
    case 1: return pt2.x > pt1.x;
    case 2: return pt2.y > pt1.y;
    default: return pt2.x < pt1.x;
    }
  }

__host__ __device__ bool HasHorzOverlap(const cuPoint64& left1, const cuPoint64& right1,
    const cuPoint64& left2, const cuPoint64& right2)
  {
    return (left1.x < right2.x) && (right1.x > left2.x);
  }

__host__ __device__ bool HasVertOverlap(const cuPoint64& top1, const cuPoint64& bottom1,
    const cuPoint64& top2, const cuPoint64& bottom2)
  {
    return (top1.y < bottom2.y) && (bottom1.y > top2.y);
  }
/*

__host__ __device__ void UncoupleEdge(cuOutPt2* op)
  {
    if (!op->edge) return;
    for (size_t i = 0; i < op->edge->size(); ++i)
    {
      OutPt2* op2 = (*op->edge)[i];
      if (op2 == op)
      {
        (*op->edge)[i] = nullptr;
        break;
      }
    }
    op->edge = nullptr;
  }

__host__ __device__ void SetNewOwner(cuOutPt2* op, size_t new_idx)
  {
    op->owner_idx = new_idx;
    OutPt2* op2 = op->next;
    while (op2 != op)
    {
      op2->owner_idx = new_idx;
      op2 = op2->next;
    }
  }
  */

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

__device__ void Append(cuPath64& input, int64_t x, int64_t y)
{
	input.points[input.size].x = x;
	input.points[input.size].y = y;
	input.size = input.size + 1;
}

__global__ void testonly_updateverticecount(cuPaths64* input, cuPaths64* output) {
	int thread_no = gridDim.x * blockDim.x;
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int ctn = input->size / thread_no + 1;
	for (int i = ctn * id; i < (ctn * (id+1)) && i < input->size; ++i) {

		for (int j = 0; j < input->cupaths[i].size; ++j) {
			int next = (i+1) % input->cupaths[i].size;
			Append(output->cupaths[i], input->cupaths[i].points[j].x, input->cupaths[i].points[j].y);
			Append(output->cupaths[i], (input->cupaths[i].points[j].x + input->cupaths[i].points[next].x)/2 + 2,
										 (input->cupaths[i].points[j].y + input->cupaths[i].points[next].y) / 2 + 2);
		}

	}
}

// TODO: change type of parameters
__global__ void executeClip(const cuPaths64& input, cuRect64* rect ,cuPaths64& output)
{/*
  int thread_no = gridDim.x * blockDim.x;
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int ctn = input->size / thread_no + 1;
  cuRect64 r1 = getBoundary(input->cupaths[i]);*/
  // TODO...
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

	filter<<<10, 128>>>(paths, r1, filterarr);
	cudaDeviceSynchronize();
	Paths64 overlaps;

	for (int i = 0; i < input.size(); ++i) {
		if (filterarr[i] == 1) { // inside, just add into output
			output.push_back(input[i]);
		} else if (filterarr[i] == 2) { // overlap, need to continue process
			overlaps.push_back(input[i]);
		}
	}

	cuPaths64* ins;
	cudaMallocManaged(&ins, sizeof(cuPaths64));
	ins->init(overlaps);

	cuPaths64* outs;
	cudaMallocManaged(&outs, sizeof(cuPaths64));
	outs->initShapeOnly(overlaps, 2);

	testonly_updateverticecount<<<1, 1>>>(ins, outs);
	cudaDeviceSynchronize();
	output = outs->toPaths64();

	cudaFree(ins);
	cudaFree(outs);
	/*

	///TBD: continue process overlaps, and add into output
  cudaMallocManaged(&inputClip, sizeof(cuPaths64));
  inputClip->init(overlaps);
  cudaMallocManaged(&outputClip, sizeof(cuPaths64));
  outputClip->initShapeOnly(overlaps, 2); // just create the shape and reserve 2 x points for each polygon.
  executeClip<<<10, 128>>>(inputClip, r1 ,outputClip ...);

*/
	cudaFree(r1);
	cudaFree(filterarr);
	cudaFree(paths);
}


}
