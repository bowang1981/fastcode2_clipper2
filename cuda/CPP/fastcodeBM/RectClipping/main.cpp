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
    Clipper2Lib::wrap_test_print();
    AreaCalcTest::benchmark_cuda();

   // UnionTest::testSplitPathsPerf();
    // UnionTest::benchmarks();
     // OffsettingTest::doConvertTests();
    // OffsettingTest::benchmarks();
   // Clipper2Lib::Paths64 ps100 = TestGenerator::MakeTestCase(10, 30);
    //TestGenerator::SaveAndDisplay(ps100, "testcase100.svg", 800, 800);

    // UnionTest::DoSquares();
    // RectClippingTest::DoRectanglesTest(1000);
    // RectClippingTest::DoRectClippingTest_1K();
    // RectClippingTest::DoRectClippingTest_1M();
    // RectClippingTest::DoPolygonTest(500);
    // OffsettingTest::doOffsetTest1();
    // OffsettingTest::doOffsetTest();

}
