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

	void benchmark(int64_t pnum) {
		Paths64 ps1 = TestGenerator::MakeTestCase(pnum / 2, 100);
		Paths64 ps2 = TestGenerator::MakeTestCase(pnum / 2, 110);
		Paths64 ps;
		ps.reserve(ps1.size() + ps2.size());
		for (int i = 0; i < pnum; ++i) {
			if( i < ps1.size()) {
				ps.push_back(ps1[i]);
			}
			if (i < ps2.size()) {
				ps.push_back(ps2[i]);
			}
		}
        //TestGenerator::SaveAndDisplay(ps, "uniontest_case.svg");
        Paths64 solution;
        ClipType cliptype = ClipType::Union;
        FillRule fillrule = FillRule::NonZero;

        std::vector<int> nums = {1, 2, 4, 8, 16, 32};
        for (auto num : nums)
        {
            Timer t;
            solution = Union_OpenMP(ps, fillrule, num);
            std::cout << "thread_num[" << num << "]: Union_OpenMP on " << pnum <<" polygons: "
                      << t.elapsed_str() << std::endl;
           // TestGenerator::SaveAndDisplay(solution, "uniontest_openmp.svg");
        }



        {
            Timer t;
            solution = Union(ps, fillrule);
            std::cout << "Union Baseline on " << pnum << " polygons: "<< t.elapsed_str() << std::endl;
           // TestGenerator::SaveAndDisplay(solution, "uniontest_openmp.svg");
        }

	}

	void benchmarks() {
		benchmark(1000);
		benchmark(10000);
		benchmark(100000);
		benchmark(1000000);
	}

	void testSplitPaths(int cnt, int split)
	{
		Paths64 paths = TestGenerator::MakeTestCase(cnt, 100);
		std::vector<Paths64> subjectsVec;
		{
		Timer t;

		splitPaths(paths, split, true, subjectsVec);
        std::cout << "Split " << cnt << " polygons: "<< t.elapsed_str() << std::endl;
		}
	}

	void testSplitPathsPerf() {
		testSplitPaths(10000, 8);
		testSplitPaths(1000000, 8);
		testSplitPaths(1000000, 16);
	}
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
