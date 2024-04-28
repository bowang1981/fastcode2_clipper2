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
    void DoTestAreaCalc() {
        Path64 subject = TestGenerator::MakeNoSelfIntesectPolygon(500000, 50000, 400000);

        {
            std::cout << "start calc area (original)" << std::endl;
            Timer t;
            double area = Clipper2Lib::Area(subject);
            std::cout << "Area on complex polygons: " << t.elapsed_str() << std::endl;
            cout << "Area: " << setprecision(10) << area << std::endl;
        }
        {
            std::cout << "start calc area (CUDA)" << std::endl;
            Timer t;

            double area = Clipper2Lib::area_single(subject);
            std::cout << "Area on complex polygons: " << t.elapsed_str() << std::endl;
            cout << "Area: " << setprecision(10) << area << std::endl;
        }


    }

    void DoTestAreaCalc2() {/*
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


    void DoTestAreaCalc3() {

  

        Paths64 subjects  = TestGenerator::MakeNoSelfIntesectPolygons(500000, 5000000, 50000, 200); 
        {
            std::cout << "\nStart DoTestAreaCalc3 (original implementation)" << std::endl;
            Timer t;

            const float area = Clipper2Lib::Area(subjects);
            std::cout << "Calulating the sum of " << subjects.size() <<" polygons.\n";
            std::cout << "Area on a large number of polygons: " << t.elapsed_str() << "\n";
            std::cout << "Area: " << setprecision(10) << area << std::endl;
        }
        {
            std::cout << "\nStart area_paths (cuda implementation)" << std::endl;
            Timer t;
            const float area =  Clipper2Lib::area_paths(subjects);
            std::cout << "Calulating the sum of " << subjects.size() <<" polygons.\n";
            std::cout << "Area on a large number of polygons: " << t.elapsed_str() << "\n";
            std::cout << "Area: " << setprecision(10) << area << std::endl;
        }


    }
}

    // void DoTestAreaCalc3() {
    //     auto subjects = TestGenerator::MakeNoSelfIntesectPolygons(10000, 500000, 50000, 2000);        
    //     {
    //         std::cout << "\nStart DoTestAreaCalc3 (original implementation)" << std::endl;
    //         Timer t;

    //         const double area = Area<int64_t>(subjects);
    //         std::cout << "Calulating the sum of " << subjects.size() <<" polygons.\n";
    //         std::cout << "Area on a large number of polygons: " << t.elapsed_str() << "\n";
    //         std::cout << "Area: " << setprecision(10) << area << std::endl;
    //     }
    //     {
    //         std::cout << "\nStart DoTestAreaCalc3 (partially openmp implementation)" << std::endl;
    //         Timer t;

    //         const double area = Area_OpenMP_With_Clipper2_Area(subjects);
    //         std::cout << "Calulating the sum of " << subjects.size() <<" polygons.\n";
    //         std::cout << "Area_OpenMP_With_Clipper2_Area on a large number of polygons: " << t.elapsed_str() << "\n";
    //         std::cout << "Area: " << setprecision(10) << area << std::endl;
    //     }
    //     {
    //         std::cout << "\nStart DoTestAreaCalc3 (openmp implementation)" << std::endl;
    //         Timer t;

    //         const double area = Area_OpenMP(subjects);
    //         std::cout << "Calulating the sum of " << subjects.size() <<" polygons.\n";
    //         std::cout << "Area_OpenMP on a large number of polygons: " << t.elapsed_str() << "\n";
    //         std::cout << "Area: " << setprecision(10) << area << std::endl;
    //     }
    // }
