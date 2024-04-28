#include "clipper2/clipper.offset.cuh"
#include "clipper2/clipper.core.cuh"
#include "../../Utils/Timer.h"

namespace Clipper2Lib {





__device__ bool AlmostZero(double value, double epsilon = 0.001)
{
	return fabs(value) < epsilon;
}

__device__ double Hypot(double x, double y)
{
	//see https://stackoverflow.com/a/32436148/359538
	return sqrt(x * x + y * y);
}

__device__ cuPointD NormalizeVector(const cuPointD& vec)
{
	double h = Hypot(vec.x, vec.y);
	if (AlmostZero(h)) return cuPointD(0,0);
	double inverseHypot = 1 / h;
	return cuPointD(vec.x * inverseHypot, vec.y * inverseHypot);
}

__device__ cuPointD GetAvgUnitVector(const cuPointD& vec1, const cuPointD& vec2)
{
	return NormalizeVector(cuPointD(vec1.x + vec2.x, vec1.y + vec2.y));
}

__device__ cuPointD TranslatePoint(const cuPointD& pt, double dx, double dy)
{
	return cuPointD(pt.x + dx, pt.y + dy);
}

__device__ cuPointD ReflectPoint(const cuPointD& pt, const cuPointD& pivot)
{
	return cuPointD(pivot.x + (pivot.x - pt.x), pivot.y + (pivot.y - pt.y));
}

__device__ cuPointD IntersectPoint(const cuPointD& pt1a, const cuPointD& pt1b,
	const cuPointD& pt2a, const cuPointD& pt2b)
{
	if (pt1a.x == pt1b.x) //vertical
	{
		if (pt2a.x == pt2b.x) return cuPointD(0, 0);

		double m2 = (pt2b.y - pt2a.y) / (pt2b.x - pt2a.x);
		double b2 = pt2a.y - m2 * pt2a.x;
		return cuPointD(pt1a.x, m2 * pt1a.x + b2);
	}
	else if (pt2a.x == pt2b.x) //vertical
	{
		double m1 = (pt1b.y - pt1a.y) / (pt1b.x - pt1a.x);
		double b1 = pt1a.y - m1 * pt1a.x;
		return cuPointD(pt2a.x, m1 * pt2a.x + b1);
	}
	else
	{
		double m1 = (pt1b.y - pt1a.y) / (pt1b.x - pt1a.x);
		double b1 = pt1a.y - m1 * pt1a.x;
		double m2 = (pt2b.y - pt2a.y) / (pt2b.x - pt2a.x);
		double b2 = pt2a.y - m2 * pt2a.x;
		if (m1 == m2) return cuPointD(0, 0);
		double x = (b2 - b1) / (m1 - m2);
		return cuPointD(x, m1 * x + b1);
	}
}

__device__ cuPoint64 GetPerpendic(const cuPoint64& pt, const cuPointD& norm, double delta)
{
	return cuPoint64(d2i(pt.x + norm.x * delta), d2i(pt.y + norm.y * delta));
}

__device__ cuPointD GetPerpendicD(const cuPoint64& pt, const cuPointD& norm, double delta)
{
	return cuPointD(pt.x + norm.x * delta, pt.y + norm.y * delta);
}

__device__ cuPointD GetUnitNormal(const cuPoint64& pt1, const cuPoint64& pt2)
{
	double dx, dy, inverse_hypot;
	if (pt1.x == pt2.x && pt1.y == pt2.y) return cuPointD(0.0, 0.0);
	dx = static_cast<double>(pt2.x - pt1.x);
	dy = static_cast<double>(pt2.y - pt1.y);
	inverse_hypot = 1.0 / hypot(dx, dy);
	dx *= inverse_hypot;
	dy *= inverse_hypot;
	return cuPointD(dy, -dx);
}

__device__ cuPointD GetNorm(const cuPath64& path, int i)
{
	int sz = path.size;
	int next = (i + 1) % sz;
	return GetUnitNormal(path.points[i], path.points[next]);
}

__device__ void DoBevel(cuPath64& input, double group_delta_,
		size_t j, size_t k, cuPath64& output)
{
	int64_t x1, y1, x2, y2;
	cuPointD normj = GetNorm(input, j);
	cuPointD normk = GetNorm(input, k);
	if (j == k)
	{
		double abs_delta = abs(group_delta_);
		x1 = input.points[j].x - abs_delta * normj.x;
		y1 = input.points[j].y - abs_delta * normj.y;

		x2 = input.points[j].x + abs_delta * normj.x;
		y2 = input.points[j].y + abs_delta * normj.y;
	}
	else
	{
		x1 = input.points[j].x + group_delta_ * normk.x;
		y1 = input.points[j].y + group_delta_ * normk.y;
		x2 = input.points[j].x + group_delta_ * normj.x;
		y2 = input.points[j].y + group_delta_ * normj.y;
	}
	output.push_back(d2i(x1), d2i(y1));
	output.push_back(d2i(x2), d2i(y2));
}

__device__ void DoSquare(cuPath64& path, size_t j, size_t k,
		 cuPath64& path_out, double group_delta_)
{
	cuPointD normj = GetNorm(path, j);
	cuPointD normk = GetNorm(path, k);
	cuPointD vec(0, 0);
	if (j == k) {
		vec.x = normj.y;
		vec.y = -normj.x;
	}
	else {
		vec = GetAvgUnitVector(
			cuPointD(-normk.y, normk.x),
			cuPointD(normj.y, -normj.x));
	}

	double abs_delta = abs(group_delta_);

	// now offset the original vertex delta units along unit vector
	cuPointD ptQ(0, 0);
	ptQ.x = path.points[j].x;
	ptQ.y = path.points[j].y;
	ptQ = TranslatePoint(ptQ, abs_delta * vec.x, abs_delta * vec.y);
	// get perpendicular vertices
	cuPointD pt1 = TranslatePoint(ptQ, group_delta_ * vec.y, group_delta_ * -vec.x);
	cuPointD pt2 = TranslatePoint(ptQ, group_delta_ * -vec.y, group_delta_ * vec.x);
	// get 2 vertices along one edge offset
	cuPointD pt3 = GetPerpendicD(path.points[k], normk, group_delta_);
	if (j == k)
	{
		cuPointD pt4 = cuPointD(pt3.x + vec.x * group_delta_, pt3.y + vec.y * group_delta_);
		cuPointD pt = IntersectPoint(pt1, pt2, pt3, pt4);

		//get the second intersect point through reflecion
		cuPointD d1 = ReflectPoint(pt, ptQ);
		path_out.push_back(d2i(d1.x), d2i(d1.y));
		path_out.push_back(d2i(pt.x), d2i(pt.y));
	}
	else
	{
		cuPointD pt4 = GetPerpendicD(path.points[j], normk, group_delta_);
		cuPointD pt = IntersectPoint(pt1, pt2, pt3, pt4);

		path_out.push_back(d2i(pt.x), d2i(pt.y));
		//get the second intersect point through reflecion
		cuPointD d1 = ReflectPoint(pt, ptQ);
		path_out.push_back(d2i(d1.x), d2i(d1.y));
	}
}

__device__ void DoMiter(const cuPath64& path, size_t j, size_t k, double cos_a,
		cuPath64& path_out,  double group_delta_)
{
	cuPointD normj = GetNorm(path, j);
	cuPointD normk = GetNorm(path, k);
	double q = group_delta_ / (cos_a + 1);

	path_out.push_back(d2i(path.points[j].x + (normk.x + normj.x) * q),
		d2i(path.points[j].y + (normk.y + normj.y) * q));
}

__device__ void DoRound(cuPath64& path, size_t j, size_t k,
						double angle, double group_delta_, cuPath64& path_out,
						OffsetParam* param, int* debug)
{
	/*if (deltaCallback64_) {
		// Bo: this code should not be needed, as we only support the simple version
		// when deltaCallback64_ is assigned, group_delta_ won't be constant,
		// so we'll need to do the following calculations for *every* vertex.
		double abs_delta = std::fabs(group_delta_);
		double arcTol = (arc_tolerance_ > floating_point_tolerance ?
			std::min(abs_delta, arc_tolerance_) :
			std::log10(2 + abs_delta) * default_arc_tolerance);
		double steps_per_360 = std::min(PI / std::acos(1 - arcTol / abs_delta), abs_delta * PI);
		step_sin_ = std::sin(2 * PI / steps_per_360);
		step_cos_ = std::cos(2 * PI / steps_per_360);
		if (group_delta_ < 0.0) step_sin_ = -step_sin_;
		steps_per_rad_ = steps_per_360 / (2 * PI);
	}*/
	cuPointD normj = GetNorm(path, j);
	cuPointD normk = GetNorm(path, k);
	cuPoint64 pt = path.points[j];
	cuPointD offsetVec = cuPointD(normk.x * group_delta_, normk.y * group_delta_);

	if (j == k)  {
		offsetVec.x = -offsetVec.x;
		offsetVec.y = -offsetVec.y;
		//offsetVec.Negate();
	}

	path_out.push_back(d2i(pt.x + offsetVec.x), d2i(pt.y + offsetVec.y));
	int steps = d2i(ceil(param->steps_per_rad_ * abs(angle))); // #448, #456
	int range = steps / 2;
	for (int i = 1; i < steps; ++i) // ie 1 less than steps
	{
		offsetVec = cuPointD(offsetVec.x * param->step_cos_ - param->step_sin_ * offsetVec.y,
			offsetVec.x * param->step_sin_ + offsetVec.y * param->step_cos_);
		if ((steps % range) == 0) {
			path_out.push_back(d2i(pt.x + offsetVec.x), d2i(pt.y + offsetVec.y));
		}
	}
	cuPoint64 p1 = GetPerpendic(path.points[j], normj, group_delta_);
	path_out.push_back(p1.x, p1.y);

}

__device__ void OffsetPoint( cuPath64& path, size_t j, size_t k,
		double group_delta_, cuPath64& path_out,
		OffsetParam* param, int* debug)
{
	//TODO: Bo to finish
	// Let A = change in angle where edges join
	// A == 0: ie no change in angle (flat join)
	// A == PI: edges 'spike'
	// sin(A) < 0: right turning
	// cos(A) < 0: change in angle is more than 90 degree

	if (path.points[j].x == path.points[k].x &&
			path.points[j].y == path.points[k].y) { k = j; return; }
	cuPointD normj = GetNorm(path, j);
	cuPointD normk = GetNorm(path, k);
	double sin_a = CrossProduct(normj, normk);
	double cos_a = DotProduct(normj, normk);
	if (sin_a > 1.0) sin_a = 1.0;
	else if (sin_a < -1.0) sin_a = -1.0;

	/*if (deltaCallback64_) { //Bo: not needed for our test?
		group_delta_ = deltaCallback64_(path, norms, j, k);
		if (group.is_reversed) group_delta_ = -group_delta_;
	}*/
	if (fabs(group_delta_) <= param->floating_point_tolerance)
	{
		path_out.append(path.points[j]);
		return;
	}

	if (cos_a > -0.99 && (sin_a * group_delta_ < 0)) // test for concavity first (#593)
	{
		// is concave
		path_out.append(GetPerpendic(path.points[j], normk, group_delta_));
		// this extra point is the only (simple) way to ensure that
	  // path reversals are fully cleaned with the trailing clipper
		path_out.append(path.points[j]); // (#405)
		path_out.append(GetPerpendic(path.points[j], normj, group_delta_));
	}
	else if (cos_a > 0.999 && param->join_type_ != 2) // Clipper2Lib::JoinType::Round)
	{
		// enum class JoinType { Square, Bevel, Round, Miter };
		// almost straight - less than 2.5 degree (#424, #482, #526 & #724)
		//DoMiter(const cuPath64& path, size_t j, size_t k, double cos_a,
			//	cuPath64& path_out, cuPathD& norms, double group_delta_)
		DoMiter(path, j, k, cos_a, path_out, group_delta_);
	}
	else if (param->join_type_ == 3) // Clipper2Lib::JoinType::Miter)
	{
		// miter unless the angle is sufficiently acute to exceed ML
		if (cos_a > param->temp_lim_ - 1) DoMiter(path, j, k, cos_a,  path_out, group_delta_);
		else DoSquare(path, j, k, path_out, group_delta_);
	}
	else if (param->join_type_ == 2) // Clipper2Lib::JoinType::Round)
		DoRound(path, j, k, atan2(sin_a, cos_a), group_delta_, path_out, param, debug);
	else if ( param->join_type_ == 1) // Clipper2Lib::JoinType::Bevel)
		DoBevel(path, group_delta_,j, k, path_out);
	else
		DoSquare(path, j, k, path_out, group_delta_);

}

__device__ void OffsetPolygon(cuPath64& path, cuPath64& path_out, double group_delta,
		 OffsetParam* param, int* debug)
{
	// TODO: Bo to finish
	for (int j = 0, k = path.size -1; j < path.size; k = j, ++j) {
		OffsetPoint(path, j, k, group_delta, path_out, param, debug);
	}

}


__global__ void offset_kernel(cuPaths64* input, cuPaths64* output,
							double group_delta, OffsetParam* param, int* debug)
{

	// call offsetPolygon here
	int total = gridDim.x * blockDim.x;
	int batch = input->size / total + 1;
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = id * batch; i < (id+1) * batch && i < input->size; ++i) {
		OffsetPolygon(input->cupaths[i], output->cupaths[i], group_delta, param, debug);
	}

}

__global__ void setvalueonly_kernel(cuPaths64* input, cuPaths64* output,
							double group_delta, OffsetParam* param, int* debug)
{

	// call offsetPolygon here
	int total = gridDim.x * blockDim.x;
	int batch = input->size / total + 1;
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = id * batch; i < (id+1) * batch && i < input->size; ++i) {
		//OffsetPolygon(input->cupaths[i], output->cupaths[i], group_delta, param, debug);
		for (int j = 0; j < output->cupaths[i].size; ++j)
		{
			output->cupaths[i].push_back(j, 1+j);

		}
	}

}

void offset_execute(const Paths64& input, double delta, Paths64& output,  const OffsetParam& param)
{
	//TODO: Bo to finish
	// once this is done, we can change the Exectue_Internal to call this function
	// call the kernel offset_kernel here
	// We only support offsetPolygon


	Timer t;
	cuPaths64* paths;
	cudaMallocManaged(&paths, sizeof(cuPaths64));
	paths->init(input);

	cuPaths64* res;
	cudaMallocManaged(&res, sizeof(cuPaths64));
	res->initShapeOnly(input, 4);
	OffsetParam* cuparam;
	cudaMallocManaged(&cuparam, sizeof(OffsetParam));
	cuparam->floating_point_tolerance = param.floating_point_tolerance;
	cuparam->join_type_ = param.join_type_;
	cuparam->step_cos_ = param.step_cos_;
	cuparam->step_sin_ = param.step_sin_;
	cuparam->steps_per_rad_ = param.steps_per_rad_;//
	cuparam->temp_lim_ = param.temp_lim_;
	int* debug;
	cudaMallocManaged(&debug, 16 * sizeof(int));

	int device = -1;
	cudaGetDevice(&device);
	cudaMemPrefetchAsync(paths->allpoints, paths->total_points, device, NULL);
	cudaMemPrefetchAsync(res->allpoints, res->total_points, device, NULL);
	std::cout << "CUDA: Prepare Data: "
	             << t.elapsed_str() << std::endl;

	{
	 Timer t1;
	// std::cout << "Luanch CUDA:"  << std::endl;
	offset_kernel<<<32, 64>>>(paths, res, delta, cuparam, debug);
	//setvalueonly_kernel<<<1, 1>>>(paths, res, delta, cuparam, debug);
	cudaDeviceSynchronize();
    std::cout << "CUDA: Kernel Run time: "
              << t1.elapsed_str() << std::endl;
	}
    // std::cout << "CUDA: Kernel run until toPaths64: "
    //          << t1.elapsed_str() << std::endl;
	{
	Timer t1;
	output = res->toPaths64();
     std::cout << "Convert data from cuPaths64: "
             << t1.elapsed_str() << std::endl;
	}
	// std::cout << std::endl;
	cudaFree(debug);
	cudaFree(paths);
	cudaFree(res);
}

} // end of namespace
