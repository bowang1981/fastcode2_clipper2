/*
 * clipper.offset.cuh
 *
 *  Created on: Apr 23, 2024
 *      Author: bow2
 */

#ifndef CLIPPER_OFFSET_CUH_
#define CLIPPER_OFFSET_CUH_


#include <stdio.h>
#include <vector>
#include "clipper.core.h"
// #include "clipper.rectclip.h"
// #include "clipper.offset.h"

namespace Clipper2Lib {

    // ============
	struct OffsetParam {
		double steps_per_rad_;
		double step_sin_;
		double step_cos_;
		int join_type_;
		double floating_point_tolerance;
		double temp_lim_;
	};

    void offset_execute(const Paths64& input, double delta,
    		Paths64& output, const OffsetParam& param);

}


#endif /* CLIPPER_OFFSET_CUH_ */
