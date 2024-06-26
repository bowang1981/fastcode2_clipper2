/*
 * clipper.area.cuh
 *
 *  Created on: Apr 18, 2024
 *      Author: bow2
 */

#ifndef CLIPPER_AREA_CUH_
#define CLIPPER_AREA_CUH_
#include "clipper2/clipper.core.h"
namespace Clipper2Lib {

double area_single(const Path64& path);
float area_paths(const Paths64& paths);

}




#endif /* CLIPPER_AREA_CUH_ */
