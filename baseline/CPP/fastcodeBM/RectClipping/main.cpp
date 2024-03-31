//
// Created by bowang on 3/28/24.
//
#include "TestGenerator.h"
#include "RectClipping.h"
#include "OffsettingTest.h"
#include "UnionTest.h"

int main(int argc, char* argv[])
{
    srand((unsigned)time(0));
    // RectClippingTest::DoRectanglesTest(500);
    // RectClippingTest::DoPolygonTest(10);
    //OffsettingTest::doOffsetTest1();
    UnionTest::DoSquares();
    UnionTest::DoPolygons();
}
