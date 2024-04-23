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

    void offset_execute(const Paths64& input, const Rect64& rect, Paths64& output);

}


#endif /* CLIPPER_OFFSET_CUH_ */
