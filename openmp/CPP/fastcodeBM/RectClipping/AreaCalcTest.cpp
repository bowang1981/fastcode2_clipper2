//
// Created by bowang on 3/31/24.
//

#include "AreaCalcTest.h"
#include "TestGenerator.h"
#include "clipper2/clipper_openmp.h"
#include "../../Utils/Timer.h"

using namespace Clipper2Lib;

using namespace std;

namespace AreaCalcTest {
    void DoTestAreaCalc() {
        Path64 subject = TestGenerator::MakeNoSelfIntesectPolygon(5000000, 50000, 50000000);
        {
            std::cout << "start calc  area" << std::endl;
            Timer t;

            double area = Area<int64_t>(subject);
            std::cout << "Area on complex polygons: " << t.elapsed_str() << std::endl;
            cout << "Area: " << setprecision(10) << area << std::endl;
        }
        // {
        //     std::cout << "start calc  area(openmp)" << std::endl;
        //     Timer t;

        //     double area = Area_OpenMP(subject);
        //     std::cout << "Area_openMP on complex polygons: " << t.elapsed_str() << std::endl;
        //     cout << "Area: " << setprecision(10)<< area << std::endl;
        // }
        std::vector<int> nums = { 1, 2, 4, 8, 16, 32, 40, 48, 64};
        for (auto num : nums)
        {
            std::cout << "start calc  area(openmp)" << std::endl;
            Timer t;

            double area = Area_OpenMP(subject, num);
            std::cout << "threadnum [" << num << "]: Area_openMP on complex polygons: " << t.elapsed_str() << std::endl;
            cout << "Area: " << setprecision(10)<< area << std::endl;
        }
    }

    // void DoTestAreaCalc2() {
    //     std::cout << "Test Area Calculation on a set of polygons" << std::endl;
    //     Paths64 subjects = TestGenerator::MakeNoSelfIntesectPolygons(1000, 500000, 50000, 20000);
    //     {
    //         std::cout << "start calc  area" << std::endl;
    //         Timer t;

    //         double area = Area<int64_t>(subjects);
    //         std::cout << "Area on complex polygons: " << t.elapsed_str() << std::endl;
    //         cout << "Area: " << setprecision(10) << area << std::endl;
    //     }
    //     std::vector<int> nums = { 1, 2, 4, 8, 16, 32, 40, 48, 64};
    //     for (auto num : nums)
    //     {
    //         std::cout << "start calc  area(openmp)" << std::endl;
    //         Timer t;

    //         double area = Area_OpenMP(subjects, num);
    //         std::cout << "threadnum [" << num << "]: Area_openMP on complex polygons: " << t.elapsed_str() << std::endl;
    //         cout << "Area: " << setprecision(10)<< area << std::endl;
    //     }
    // }


    // Distribute a thread to each polygon

    void DoTestAreaCalc3() {
        auto subjects = TestGenerator::MakeNoSelfIntesectPolygons(500000, 500000, 50000, 200);        
        // {
        //     std::cout << "\nStart DoTestAreaCalc3 (original implementation)" << std::endl;
        //     Timer t;
        //     const double area = Area<int64_t>(subjects);
        //     std::cout << "Calulating the sum of " << subjects.size() <<" polygons.\n";
        //     std::cout << "Area on a large number of polygons: " << t.elapsed_str() << "\n";
        //     std::cout << "Area: " << setprecision(10) << area << std::endl;
        // }
        // {
        //     std::cout << "\nStart DoTestAreaCalc3 (openmp implementation)\n";
        //     int nt = 4;
        //     std::cout<<"Num Threads "<<nt<<"\n";
        //     Timer t;
        //     const double area = Area_OpenMP_Massive_Polygons(subjects, nt);
        //     std::cout << "Calulating the sum of " << subjects.size() <<" polygons.\n";
        //     std::cout << "Area_OpenMP on a large number of polygons: " << t.elapsed_str() << "\n";
        //     std::cout << "Area: " << setprecision(10) << area << std::endl;
        // }
        std::vector<int> num_threads_vec = { 1, 2, 4, 8, 16, 32, 40, 48, 64};
        for (auto nt: num_threads_vec)
        {
            std::cout << "\nStart DoTestAreaCalc3 (openmp implementation)\n";
            std::cout<<"Num Threads "<<nt<<"\n";
            Timer t;

            const double area = Area_OpenMP_Massive_Polygons(subjects,nt);
            std::cout << "Calulating the sum of " << subjects.size() <<" polygons.\n";
            std::cout << "Area_OpenMP on a large number of polygons: " << t.elapsed_str() << "\n";
            std::cout << "Area: " << setprecision(10) << area << std::endl;
        }
        return;
    }
}
