//
// Created by bowang on 3/28/24.
//
#include "TestGenerator.h"
#include "RectClipping.h"
#include "OffsettingTest.h"
#include "clipper2/clipper.cuh"
#include "AreaCalcTest.h"

int main(int argc, char* argv[])
{
    srand((unsigned)time(0));
    AreaCalcTest::DoTestAreaCalc();
    Clipper2Lib::wrap_test_print();
    // RectClippingTest::DoRectanglesTest(500);
    // RectClippingTest::DoPolygonTest(500);
    // OffsettingTest::doOffsetTest1();
    // OffsettingTest::doOffsetTest();

}
