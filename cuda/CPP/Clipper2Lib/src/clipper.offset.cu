#include "clipper2/clipper.offset.cuh"
#include "clipper2/clipper.core.cuh"


namespace Clipper2Lib {



__device__ void DoBevel(cuPath64& input, double group_delta_,
		size_t j, size_t k, cuPath64& output, cuPathD& norms)
{
	int64_t x1, y1, x2, y2;
	if (j == k)
	{
		double abs_delta = std::abs(group_delta_);
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
/*
__device__ void ClipperOffset::DoSquare(cuPath64& path, size_t j, size_t k,
		cuPathD& norms, cuPath64& path_out, double group_delta_)
{
	cuPointD vec;
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
	cuPointD ptQ;
	ptQ.x = path.points[j].x;
	ptQ.y = path.points[j].y;
	ptQ = TranslatePoint(ptQ, abs_delta * vec.x, abs_delta * vec.y);
	// get perpendicular vertices
	PointD pt1 = TranslatePoint(ptQ, group_delta_ * vec.y, group_delta_ * -vec.x);
	PointD pt2 = TranslatePoint(ptQ, group_delta_ * -vec.y, group_delta_ * vec.x);
	// get 2 vertices along one edge offset
	PointD pt3 = GetPerpendicD(path[k], norms[k], group_delta_);
	if (j == k)
	{
		PointD pt4 = PointD(pt3.x + vec.x * group_delta_, pt3.y + vec.y * group_delta_);
		PointD pt = IntersectPoint(pt1, pt2, pt3, pt4);

		//get the second intersect point through reflecion
		path_out.push_back(Point64(ReflectPoint(pt, ptQ)));
		path_out.push_back(Point64(pt));
	}
	else
	{
		PointD pt4 = GetPerpendicD(path[j], norms[k], group_delta_);
		PointD pt = IntersectPoint(pt1, pt2, pt3, pt4);

		path_out.push_back(Point64(pt));
		//get the second intersect point through reflecion
		path_out.push_back(Point64(ReflectPoint(pt, ptQ)));
	}
}
*/
__device__ void DoMiter(const cuPath64& path, size_t j, size_t k, double cos_a,
		cuPath64& path_out, cuPathD& norms, double group_delta_)
{
	double q = group_delta_ / (cos_a + 1);

	path_out.push_back(d2i(path.points[j].x + (norms.points[k].x + norms.points[j].x) * q),
		d2i(path.points[j].y + (norms.points[k].y + norms.points[j].y) * q));
}
/*
void ClipperOffset::DoRound(const Path64& path, size_t j, size_t k, double angle)
{
	if (deltaCallback64_) {
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
	}

	Point64 pt = path[j];
	PointD offsetVec = PointD(norms[k].x * group_delta_, norms[k].y * group_delta_);

	if (j == k) offsetVec.Negate();

	path_out.push_back(Point64(pt.x + offsetVec.x, pt.y + offsetVec.y));
	int steps = static_cast<int>(std::ceil(steps_per_rad_ * std::abs(angle))); // #448, #456
	for (int i = 1; i < steps; ++i) // ie 1 less than steps
	{
		offsetVec = PointD(offsetVec.x * step_cos_ - step_sin_ * offsetVec.y,
			offsetVec.x * step_sin_ + offsetVec.y * step_cos_);

		path_out.push_back(Point64(pt.x + offsetVec.x, pt.y + offsetVec.y));
	}
	path_out.push_back(GetPerpendic(path[j], norms[j], group_delta_));
}

__device__ void ClipperOffset::OffsetPoint( cuPath64& path, size_t j, size_t k,
		cuPathD& norms, double group_delta_)
{
	// Let A = change in angle where edges join
	// A == 0: ie no change in angle (flat join)
	// A == PI: edges 'spike'
	// sin(A) < 0: right turning
	// cos(A) < 0: change in angle is more than 90 degree

	if (path[j] == path[k]) { k = j; return; }

	double sin_a = CrossProduct(norms.points[j], norms.points[k]);
	double cos_a = DotProduct(norms.points[j], norms.points[k]);
	if (sin_a > 1.0) sin_a = 1.0;
	else if (sin_a < -1.0) sin_a = -1.0;

	if (deltaCallback64_) { //needed?
		group_delta_ = deltaCallback64_(path, norms, j, k);
		if (group.is_reversed) group_delta_ = -group_delta_;
	}
	if (fabs(group_delta_) <= floating_point_tolerance)
	{
		path_out.push_back(path[j]);
		return;
	}

	if (cos_a > -0.99 && (sin_a * group_delta_ < 0)) // test for concavity first (#593)
	{
		// is concave
		path_out.push_back(GetPerpendic(path[j], norms[k], group_delta_));
		// this extra point is the only (simple) way to ensure that
	  // path reversals are fully cleaned with the trailing clipper
		path_out.push_back(path[j]); // (#405)
		path_out.push_back(GetPerpendic(path[j], norms[j], group_delta_));
	}
	else if (cos_a > 0.999 && join_type_ != JoinType::Round)
	{
		// almost straight - less than 2.5 degree (#424, #482, #526 & #724)
		DoMiter(path, j, k, cos_a);
	}
	else if (join_type_ == JoinType::Miter)
	{
		// miter unless the angle is sufficiently acute to exceed ML
		if (cos_a > temp_lim_ - 1) DoMiter(path, j, k, cos_a);
		else DoSquare(path, j, k);
	}
	else if (join_type_ == JoinType::Round)
		DoRound(path, j, k, std::atan2(sin_a, cos_a));
	else if ( join_type_ == JoinType::Bevel)
		DoBevel(path, j, k);
	else
		DoSquare(path, j, k);
}

void ClipperOffset::OffsetPolygon(Group& group, const Path64& path)
{
	path_out.clear();
	for (Path64::size_type j = 0, k = path.size() -1; j < path.size(); k = j, ++j)
		OffsetPoint(group, path, j, k);
	solution.push_back(path_out);
}
*/

__global__ void offset_kernel(cuPaths64* input, cuPaths64* output, double group_delta)
{
	// call offsetPolygon here
}

void offset_execute(const Paths64& input, const Rect64& rect, Paths64& output)
{
	// once this is done, we can change the Exectue_Internal to call this function
	// call the kernel offset_kernel here
	// We only support offsetPolygon

}

} // end of namespace
