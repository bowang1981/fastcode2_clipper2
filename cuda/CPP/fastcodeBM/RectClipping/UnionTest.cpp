//
// Created by bowang on 3/30/24.
//

#include "UnionTest.h"
#include "TestGenerator.h"
#include <vector>
#include "../../Utils/Timer.h"
#include <unistd.h>

using namespace Clipper2Lib;
using namespace std;
namespace UnionTest {
    void DoSquares()
    {
        Paths64 subjects, solution;
        ClipType cliptype = ClipType::Union;
        FillRule fillrule = FillRule::NonZero;
        subjects = TestGenerator::CreateRectangles(4);

        {
            Timer t;
            solution = Union(subjects, fillrule);
            std::cout << "Union on massive polygons: "<< t.elapsed_str() << std::endl;
        }


       // TestGenerator::SaveAndDisplay(solution, fillrule, "uniontest.svg");


    }

    void DoPolygons() {
        static const int w = 8000, h = 6000;
        int polyCount = 100;
        Paths64 subjects, solution;
        ClipType cliptype = ClipType::Union;
        FillRule fillrule = FillRule::NonZero;
        subjects = TestGenerator::MakeNoSelfIntesectPolygons(polyCount, w, h, 800);


        {
            std::cout << "start uunion  ppolyygonns" << std::endl;
            Timer t;
            solution = Union(subjects, fillrule);
            std::cout << "Union on massive polygons: "<< t.elapsed_str() << std::endl;
        }
        // TestGenerator::SaveAndDisplay(solution, fillrule, "uniontest_openmp.svg");
    }
};
