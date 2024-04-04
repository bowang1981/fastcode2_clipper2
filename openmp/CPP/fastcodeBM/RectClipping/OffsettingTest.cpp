//
// Created by bowang on 3/29/24.
// @author Fred
//
#include <iostream>
#include "OffsettingTest.h"
#include "clipper2/clipper.h"
#include "TestGenerator.h"
#include "rdtsc.h"

using namespace Clipper2Lib;

namespace OffsettingTest {

    void doOffsetTestBasic() {
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

    void doOffsetTest(int cnt, int runs) {
        Paths64 subject, solution;
        long long jiffies;
        tsc_counter t0, t1;
        for (int i = 0; i < runs; ++i) {
            subject = TestGenerator::CreateRectangles(cnt);
            ClipperOffset offsetter;
            offsetter.AddPaths(subject, JoinType::Round, EndType::Polygon);
            RDTSC(t0);
            offsetter.Execute(10, solution);
            RDTSC(t1);
            jiffies += COUNTER_DIFF(t1, t0, CYCLES);
            solution = SimplifyPaths(solution, 2.5);
        }
        std::cout << "Offsetting rectangles of size " << cnt << " : average "
                  << static_cast<double>(jiffies) / runs << " cycles" << std::endl;
    }



}
