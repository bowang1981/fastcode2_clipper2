//
// Created by bowang on 3/29/24.
//

#include "OffsettingTest.h"
#include "clipper2/clipper.h"
#include "TestGenerator.h"
using namespace Clipper2Lib;
namespace OffsettingTest {

    void doOffsetTest() {
        Paths64 subject;
        subject.push_back(MakePath({ 3480,2570, 3640,1480,
                                     3620,1480, 3260,2410, 2950,2190, 2580,880,
                                     4400,1290, 3700,1960, 3720,2750 }));
        Paths64 solution;
        ClipperOffset offsetter;
        offsetter.AddPaths(subject, JoinType::Round, EndType::Polygon);
        offsetter.Execute(-70, solution);
        solution = SimplifyPaths(solution, 2.5);
    }

    void doOffsetTest1() {
        Paths64 subject = TestGenerator::CreateRectangles(500);
        Paths64 solution;
        ClipperOffset offsetter;
        offsetter.AddPaths(subject, JoinType::Round, EndType::Polygon);
        offsetter.Execute(10, solution);
        solution = SimplifyPaths(solution, 2.5);
    }



}
