//
// Created by bowang on 3/28/24.
//
#include "TestGenerator.h"
#include "RectClipping.h"
#include "OffsettingTest.h"
#include "clipper2/clipper.cuh"
#include "AreaCalcTest.h"
#include "UnionTest.h"
#include <unistd.h>

using namespace Clipper2Lib;
int main(int argc, char* argv[])
{
    srand((unsigned)time(0));
    // AreaCalcTest::DoTestAreaCalc();
    // Clipper2Lib::wrap_test_print();


    // Clipper2Lib::Paths64 ps100 = TestGenerator::MakeTestCase(1000);
    // TestGenerator::SaveAndDisplay(ps100, "testcase100.svg", 2000, 2000);

    // UnionTest::DoSquares();
    // RectClippingTest::DoRectanglesTest(1000);
    RectClippingTest::DoRectClippingTest_1K();
    // RectClippingTest::DoRectClippingTest_1M();//error
    // RectClippingTest::DoPolygonTest(500);
    // OffsettingTest::doOffsetTest1();
    // OffsettingTest::doOffsetTest();

}
