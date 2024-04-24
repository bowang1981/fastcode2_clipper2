#include "clipper2/clipper.offset.cuh"
#include "clipper2/clipper.core.cuh"


namespace Clipper2Lib {



__device__ void DoBevel(cuPath64& input, double group_delta_,
		size_t j, size_t k, cuPath64& output, cuPathD& norms)
{
	int64_t x1, y1, x2, y2;
	if (j == k)
	{
		double abs_delta = abs(group_delta_);
		x1 = input.points[j].x - abs_delta * norms.points[j].x;
		y1 = input.points[j].y - abs_delta * norms.points[j].y;

		x2 = input.points[j].x + abs_delta * norms.points[j].x;
		y2 = input.points[j].y + abs_delta * norms.points[j].y;
	}
	else
	{
		x1 = input.points[j].x + group_delta_ * norms.points[k].x;
		y1 = input.points[j].y + group_delta_ * norms.points[k].y;
		x2 = input.points[j].x + group_delta_ * norms.points[j].x;
		y2 = input.points[j].y + group_delta_ * norms.points[j].y;
	}
	output.push_back(d2i(x1), d2i(y1));
	output.push_back(d2i(x2), d2i(y2));
}

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

__device__ void DoSquare(cuPath64& path, size_t j, size_t k,
		cuPathD& norms, cuPath64& path_out, double group_delta_)
{

	cuPointD vec(0, 0);
	if (j == k) {
		vec.x = norms.points[j].y;
		vec.y = -norms.points[j].x;
	}
	else {
		vec = GetAvgUnitVector(
			cuPointD(-norms.points[k].y, norms.points[k].x),
			cuPointD(norms.points[j].y, -norms.points[j].x));
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
	cuPointD pt3 = GetPerpendicD(path.points[k], norms.points[k], group_delta_);
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
		cuPointD pt4 = GetPerpendicD(path.points[j], norms.points[k], group_delta_);
		cuPointD pt = IntersectPoint(pt1, pt2, pt3, pt4);

		path_out.push_back(d2i(pt.x), d2i(pt.y));
		//get the second intersect point through reflecion
		cuPointD d1 = ReflectPoint(pt, ptQ);
		path_out.push_back(d2i(d1.x), d2i(d1.y));
	}
}

__device__ void DoMiter(const cuPath64& path, size_t j, size_t k, double cos_a,
		cuPath64& path_out, cuPathD& norms, double group_delta_)
{
	double q = group_delta_ / (cos_a + 1);

	path_out.push_back(d2i(path.points[j].x + (norms.points[k].x + norms.points[j].x) * q),
		d2i(path.points[j].y + (norms.points[k].y + norms.points[j].y) * q));
}

__device__ void DoRound(cuPath64& path, size_t j, size_t k,
						double angle, double group_delta_, cuPath64& path_out,
						cuPathD& norms, double steps_per_rad_, double step_sin_,
						double step_cos_)
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

	cuPoint64 pt = path.points[j];
	cuPointD offsetVec = cuPointD(norms.points[k].x * group_delta_, norms.points[k].y * group_delta_);

	if (j == k)  {
		offsetVec.x = -offsetVec.x;
		offsetVec.y = -offsetVec.y;
		//offsetVec.Negate();
	}

	path_out.push_back(d2i(pt.x + offsetVec.x), d2i(pt.y + offsetVec.y));
	int steps = d2i(ceil(steps_per_rad_ * abs(angle))); // #448, #456
	for (int i = 1; i < steps; ++i) // ie 1 less than steps
	{
		offsetVec = cuPointD(offsetVec.x * step_cos_ - step_sin_ * offsetVec.y,
			offsetVec.x * step_sin_ + offsetVec.y * step_cos_);

		path_out.push_back(d2i(pt.x + offsetVec.x), d2i(pt.y + offsetVec.y));
	}
	cuPoint64 p1 = GetPerpendic(path.points[j], norms.points[j], group_delta_);
	path_out.push_back(p1.x, p1.y);

}

__device__ void OffsetPoint( cuPath64& path, size_t j, size_t k,
		cuPathD& norms, double group_delta_, cuPath64& path_out,
		double steps_per_rad_, double step_sin_,
		double step_cos_, int join_type_,
		double floating_point_tolerance, double temp_lim_)
{
	//TODO: Bo to finish
	// Let A = change in angle where edges join
	// A == 0: ie no change in angle (flat join)
	// A == PI: edges 'spike'
	// sin(A) < 0: right turning
	// cos(A) < 0: change in angle is more than 90 degree

	if (path.points[j].x == path.points[k].x &&
			path.points[j].y == path.points[k].y) { k = j; return; }

	double sin_a = CrossProduct(norms.points[j], norms.points[k]);
	double cos_a = DotProduct(norms.points[j], norms.points[k]);
	if (sin_a > 1.0) sin_a = 1.0;
	else if (sin_a < -1.0) sin_a = -1.0;

	/*if (deltaCallback64_) { //Bo: not needed for our test?
		group_delta_ = deltaCallback64_(path, norms, j, k);
		if (group.is_reversed) group_delta_ = -group_delta_;
	}*/
	if (fabs(group_delta_) <= floating_point_tolerance)
	{
		path_out.append(path.points[j]);
		return;
	}

	if (cos_a > -0.99 && (sin_a * group_delta_ < 0)) // test for concavity first (#593)
	{
		// is concave
		path_out.append(GetPerpendic(path.points[j], norms.points[k], group_delta_));
		// this extra point is the only (simple) way to ensure that
	  // path reversals are fully cleaned with the trailing clipper
		path_out.append(path.points[j]); // (#405)
		path_out.append(GetPerpendic(path.points[j], norms.points[j], group_delta_));
	}
	else if (cos_a > 0.999 && join_type_ != 2) // Clipper2Lib::JoinType::Round)
	{
		// enum class JoinType { Square, Bevel, Round, Miter };
		// almost straight - less than 2.5 degree (#424, #482, #526 & #724)
		//DoMiter(const cuPath64& path, size_t j, size_t k, double cos_a,
			//	cuPath64& path_out, cuPathD& norms, double group_delta_)
		DoMiter(path, j, k, cos_a, path_out, norms, group_delta_);
	}
	else if (join_type_ == 3) // Clipper2Lib::JoinType::Miter)
	{
		// miter unless the angle is sufficiently acute to exceed ML
		if (cos_a > temp_lim_ - 1) DoMiter(path, j, k, cos_a,  path_out, norms, group_delta_);
		else DoSquare(path, j, k, norms, path_out, group_delta_);
	}
	else if (join_type_ == 2) // Clipper2Lib::JoinType::Round)
		DoRound(path, j, k, std::atan2(sin_a, cos_a), group_delta_, path_out,
				norms,  steps_per_rad_,  step_sin_,
				 step_cos_);
	else if ( join_type_ == 1) // Clipper2Lib::JoinType::Bevel)
		DoBevel(path, group_delta_,j, k,
				path_out, norms);
	else
		DoSquare(path, j, k, norms, path_out, group_delta_);

}

__device__ void OffsetPolygon(cuPath64& path, cuPath64& path_out, double group_delta,
		cuPathD& norms,
		double steps_per_rad_, double step_sin_,
		double step_cos_, int join_type_,
		double floating_point_tolerance, double temp_lim_)
{
	// TODO: Bo to finish
	for (int j = 0, k = path.size -1; j < path.size; k = j, ++j) {
		OffsetPoint(path, j, k, norms, group_delta, path_out, steps_per_rad_, step_sin_,
				 step_cos_, join_type_,
				 floating_point_tolerance,  temp_lim_ );
	}

}


__global__ void offset_kernel(cuPaths64* input, cuPaths64* output, double group_delta)
{
	// TODO: Bo to finish
	// call offsetPolygon here
}

void offset_execute(const Paths64& input, const Rect64& rect, Paths64& output)
{
	//TODO: Bo to finish
	// once this is done, we can change the Exectue_Internal to call this function
	// call the kernel offset_kernel here
	// We only support offsetPolygon

}

} // end of namespace
