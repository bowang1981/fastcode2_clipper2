//
// Created by bowang on 3/28/24.
//

#ifndef FASTCODE2_CLIPPER2_TESTGENERATOR_H
#define FASTCODE2_CLIPPER2_TESTGENERATOR_H
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <string>
#include "clipper2/clipper.h"

namespace TestGenerator {
    Clipper2Lib::Paths64 CreateRectangles(int cnt);
    Clipper2Lib::Path64 MakeRandomPoly(int width, int height, unsigned vertCnt);
    Clipper2Lib::Paths64 CreatePolygons(int polygon_cnt, int vtxcount);
};


#endif //FASTCODE2_CLIPPER2_TESTGENERATOR_H
