/*
 * clipper.rectclip.cuh
 *
 *  Created on: Apr 18, 2024
 *      Author: bow2
 */

#ifndef CLIPPER_RECTCLIP_CUH_
#define CLIPPER_RECTCLIP_CUH_

#include <stdio.h>
#include <vector>
#include "clipper.core.h"
#include "clipper.area.cuh"
// #include "clipper.h"
//#include "clipper.rectclip.h"
// #include "clipper.offset.h"

namespace Clipper2Lib {

void rectclip_execute(const Paths64& input, const Rect64& rect, Paths64& output);

}



#endif /* CLIPPER_RECTCLIP_CUH_ */
