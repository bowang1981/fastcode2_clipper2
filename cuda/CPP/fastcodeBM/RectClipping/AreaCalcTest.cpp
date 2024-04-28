//
// Created by bowang on 3/31/24.
//

#include "AreaCalcTest.h"
#include "TestGenerator.h"
#include "clipper2/clipper.cuh"
#include "../../Utils/Timer.h"

using namespace Clipper2Lib;
using namespace std;

namespace AreaCalcTest {
    void benchmark_cuda() {
        Path64 subject = TestGenerator::MakeNoSelfIntesectPolygon(50000000, 50000000, 5000000);
        {
            std::cout << "start calc  area" << std::endl;
            {
            Timer t;

            double area = Clipper2Lib::area_single(subject);
            std::cout << "CUDA: Area on complex polygons: " << t.elapsed_str() << std::endl;
            std::cout << "Area: " << setprecision(10) << area << std::endl;
            }
            {
            	Timer t1;
            cout << "Area(Baseline: " << setprecision(10) << Clipper2Lib::Area(subject) << std::endl;
            std::cout << "Baseline Area on complex polygons: " << t1.elapsed_str() << std::endl;
            }
        }

    }

    void benchmark_omp() {/*
        std::cout << "Test Area Calculation on a set of polygons" << std::endl;
        Paths64 subjects = TestGenerator::MakeNoSelfIntesectPolygons(1000, 500000, 50000, 20000);
        {
            std::cout << "start calc  area" << std::endl;
            Timer t;

            double area = Area<int64_t>(subjects);
            std::cout << "Area on complex polygons: " << t.elapsed_str() << std::endl;
            cout << "Area: " << setprecision(10) << area << std::endl;
        }
        std::vector<int> nums = { 1, 2, 4, 8, 16, 32, 40, 48, 64};
        for (auto num : nums)
        {
            std::cout << "start calc  area(openmp)" << std::endl;
            Timer t;

            double area = Area_OpenMP(subjects, num);
            std::cout << "threadnum [" << num << "]: Area_openMP on complex polygons: " << t.elapsed_str() << std::endl;
            cout << "Area: " << setprecision(10)<< area << std::endl;
        }*/
    }
}
