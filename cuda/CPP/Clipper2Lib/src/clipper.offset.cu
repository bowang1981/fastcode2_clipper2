#include "clipper2/clipper.offset.cuh"
#include "clipper2/clipper.core.cuh"


namespace Clipper2Lib {

__device__ void Append(cuPath64& input, int64_t x, int64_t y)
{
	input.points[input.size].x = x;
	input.points[input.size].y = y;
	input.size = input.size + 1;
}
/*
 *  We need port the following functions into kernel functions
__device__ void ClipperOffset::DoBevel(cuPath64* input, size_t j, size_t k, cuPath64* output)
{
	PointD pt1, pt2;
	if (j == k)
	{
		double abs_delta = std::abs(group_delta_);
		pt1 = PointD(path[j].x - abs_delta * norms[j].x, path[j].y - abs_delta * norms[j].y);
		pt2 = PointD(path[j].x + abs_delta * norms[j].x, path[j].y + abs_delta * norms[j].y);
	}
	else
	{
		pt1 = PointD(path[j].x + group_delta_ * norms[k].x, path[j].y + group_delta_ * norms[k].y);
		pt2 = PointD(path[j].x + group_delta_ * norms[j].x, path[j].y + group_delta_ * norms[j].y);
	}
	path_out.push_back(Point64(pt1));
	path_out.push_back(Point64(pt2));
}

void ClipperOffset::DoSquare(const Path64& path, size_t j, size_t k)
{
	PointD vec;
	if (j == k)
		vec = PointD(norms[j].y, -norms[j].x);
	else
		vec = GetAvgUnitVector(
			PointD(-norms[k].y, norms[k].x),
			PointD(norms[j].y, -norms[j].x));

	double abs_delta = std::abs(group_delta_);

	// now offset the original vertex delta units along unit vector
	PointD ptQ = PointD(path[j]);
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

void ClipperOffset::DoMiter(const Path64& path, size_t j, size_t k, double cos_a)
{
	double q = group_delta_ / (cos_a + 1);

	path_out.push_back(Point64(
		path[j].x + (norms[k].x + norms[j].x) * q,
		path[j].y + (norms[k].y + norms[j].y) * q));
}

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

void ClipperOffset::OffsetPoint(Group& group, const Path64& path, size_t j, size_t k)
{
	// Let A = change in angle where edges join
	// A == 0: ie no change in angle (flat join)
	// A == PI: edges 'spike'
	// sin(A) < 0: right turning
	// cos(A) < 0: change in angle is more than 90 degree

	if (path[j] == path[k]) { k = j; return; }

	double sin_a = CrossProduct(norms[j], norms[k]);
	double cos_a = DotProduct(norms[j], norms[k]);
	if (sin_a > 1.0) sin_a = 1.0;
	else if (sin_a < -1.0) sin_a = -1.0;

	if (deltaCallback64_) {
		group_delta_ = deltaCallback64_(path, norms, j, k);
		if (group.is_reversed) group_delta_ = -group_delta_;
	}
	if (std::fabs(group_delta_) <= floating_point_tolerance)
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

void offset_execute(const Paths64& input, const Rect64& rect, Paths64& output)
{

}

} // end of namespace
