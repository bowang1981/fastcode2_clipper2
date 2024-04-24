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
    Clipper2Lib::Path64 MakeNoSelfIntesectPolygon(int width, int height, unsigned Ctn,
    											  int64_t x_offset = 0, int64_t y_offset = 0);
    // create a set of polygons, with each bbox will less than width and height, and vertx count less than vertCtn
    Clipper2Lib::Paths64 MakeNoSelfIntesectPolygons(int polygon_ctn, int width, int height, unsigned vertCtn);

    void SaveAndDisplay(Clipper2Lib::Paths64& paths,const std::string& filepath,
    					int w = 800, int h= 600,
                        Clipper2Lib::FillRule fillrule = Clipper2Lib::FillRule::NonZero
                        );
    Clipper2Lib::Paths64 MakeTestCase(int64_t cnt);
    Clipper2Lib::Paths64 CreateTestCase_1K();
    Clipper2Lib::Paths64 CreateTestCase_100K();
    Clipper2Lib::Paths64 CreateTestCase_1M();
    Clipper2Lib::Paths64 CreateTestCase_5M();

};


#endif //FASTCODE2_CLIPPER2_TESTGENERATOR_H
