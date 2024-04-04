//
// Created by bowang on 3/28/24.
//

//
// Created by bowang on 3/28/24.
//
#include "TestGenerator.h"
#include "RectClipping.h"
#include "OffsettingTest.h"
#include "UnionTest.h"
#include "AreaCalcTest.h"
#include <omp.h>
#include <stdio.h>
using namespace  std;


// double test_omp()
// {
//     double res = 0;
// #pragma  omp parallel for reduction(+:res) num_threads(16)
//     for (int i = 0 ; i < 100; ++i) {
//         res += i * i;
//     }
//     return res;
// }
int main(int argc, char* argv[])
{
    // cout << test_omp() << std::endl;
    srand((unsigned)time(0));

    // RectClippingTest::DoRectanglesTest(5000000);
    // RectClippingTest::DoPolygonTest(50000);
    // // OffsettingTest::doOffsetTest();
    // // OffsettingTest::doOffsetTest1();
    // // UnionTest::DoSquares();
    // UnionTest::DoPolygons();
    // // AreaCalcTest::DoTestAreaCalc();
    // AreaCalcTest::DoTestAreaCalc2();

    AreaCalcTest::DoTestAreaCalc();
    // AreaCalcTest::DoTestAreaCalc2();
       
    AreaCalcTest::DoTestAreaCalc3();
}