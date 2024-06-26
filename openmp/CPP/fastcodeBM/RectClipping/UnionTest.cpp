//
// Created by bowang on 3/30/24.
//

#include "UnionTest.h"
#include "TestGenerator.h"
#include <vector>
#include "../../Utils/Timer.h"
#include "clipper2/clipper_openmp.h"

using namespace Clipper2Lib;
using namespace std;
namespace UnionTest {
    void DoSquares()
    {
        static const int w = 800, h = 600;
        static const int size = 10;
        Path64 shape;
        shape.push_back(Point64(0, 0));
        shape.push_back(Point64(size, 0));
        shape.push_back(Point64(size, size));
        shape.push_back(Point64(0, size));
        Paths64 subjects, solution;
        ClipType cliptype = ClipType::Union;
        FillRule fillrule = FillRule::NonZero;
        subjects = TestGenerator::CreateRectangles(40000);

        //SaveTest("squares.txt", false, &subjects, nullptr, nullptr, 0, 0, ClipType::Union, FillRule::NonZero);
        std::vector<Paths64> subjectsVec;
        std::vector<int> nums = {1, 2, 4, 8, 16, 32};
        for (auto num : nums)
        {
            Timer t;
            solution = Union_OpenMP(subjects, fillrule, num);
            std::cout << "thread_num[" << num << "]: Union_OpenMP on massive polygons: "
                      << t.elapsed_str() << std::endl;
        }

       // TestGenerator::SaveAndDisplay(solution, fillrule, "uniontest_openmp.svg");

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

        // TestGenerator::SaveAndDisplay(subjects, fillrule, "uniontestinput.svg");
        std::vector<int> nums = {1, 2, 4, 8, 16, 32, 40, 48, 64};



        // subjects = SimplifyPaths(subjects, 5);

        // TestGenerator::SaveAndDisplay(solution, fillrule, "uniontestoutput.svg");
        for (auto num : nums)
        {
            Timer t;
            solution = Union_OpenMP(subjects, fillrule, num);
            std::cout << "thread num [" << num << "] Union_OpenMP on massive polygons: "
            << t.elapsed_str() << std::endl;
        }

        {
            std::cout << "start uunion  ppolyygonns" << std::endl;
            Timer t;
            solution = Union(subjects, fillrule);
            std::cout << "Union on massive polygons: "<< t.elapsed_str() << std::endl;
        }
        // TestGenerator::SaveAndDisplay(solution, fillrule, "uniontest_openmp.svg");
    }
};
