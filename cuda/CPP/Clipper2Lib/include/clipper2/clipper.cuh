#ifndef CLIPPER_CUH_
#define CLIPPER_CUH_
#include <stdio.h>
#include <vector>
#include "clipper.core.h"
#include "clipper.area.cuh"
// #include "clipper.h"
//#include "clipper.rectclip.h"
// #include "clipper.offset.h"

namespace Clipper2Lib {

void wrap_test_print();

void rectclip_execute(const Paths64& input, const Rect64& rect, Paths64& output);

void test_convert_performance(const Paths64& input);

}
#endif
