//
// Created by bowang on 3/28/24.
//

#include "TestGenerator.h"

using namespace Clipper2Lib;

namespace TestGenerator {
    Path64 MakeRandomRectangle(int minWidth, int minHeight, int maxWidth, int maxHeight,
                               int maxRight, int maxBottom) {
        int w = maxWidth > minWidth ? minWidth + rand() % (maxWidth - minWidth) : minWidth;
        int h = maxHeight > minHeight ? minHeight + rand() % (maxHeight - minHeight) : minHeight;
        int l = rand() % (maxRight - w);
        int t = rand() % (maxBottom - h);
        Path64 result;
        result.reserve(4);
        result.push_back(Point64(l, t));
        result.push_back(Point64(l + w, t));
        result.push_back(Point64(l + w, t + h));
        result.push_back(Point64(l, t + h));
        return result;
    }

    Paths64 CreateRectangles(int cnt) {
        Paths64 sub;

        for (int i = 0; i < cnt; ++i)
            sub.push_back(MakeRandomRectangle(10, 10,
                                              100, 100,
                                              800, 600));

        return sub;
    }

    Path64 MakeRandomPoly(int width, int height, unsigned vertCnt)
    {
        Path64 result;
        result.reserve(vertCnt);
        for (unsigned i = 0; i < vertCnt; ++i)
            result.push_back(Point64(rand() % width, rand() % height));
        return result;
    }

    Paths64 CreatePolygons(int polygon_cnt, int vtxcount) {
        Paths64 sub;

        for (int i = 0; i < polygon_cnt; ++i)
            sub.push_back(MakeRandomPoly(10, 10, vtxcount));

        return sub;
    }
} // end of namespace