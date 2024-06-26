//
// Created by bowang on 3/29/24.
//

#include "OffsettingTest.h"
#include "clipper2/clipper.h"
#include "clipper2/clipper_openmp.h"
#include "clipper2/clipper.cuh"
#include "TestGenerator.h"
#include "../../Utils/Timer.h"
using namespace Clipper2Lib;
namespace OffsettingTest {

	void doConvertTest(int pctn) {
		Paths64 ps = TestGenerator::MakeTestCase(pctn, 100);
		test_convert_performance(ps);
	}

	void doConvertTests()
	{
		doConvertTest(1000);
		doConvertTest(10000);
		doConvertTest(100000);
		doConvertTest(1000000);

	}

	void benchmark(int pcnt) {
		std::vector<int> thread_nums = {2, 4, 8, 16, 32};
        Paths64 subject = TestGenerator::MakeTestCase(pcnt, 100);

        {
        	// baseline
            Timer t;
            Paths64 solution;
			ClipperOffset offsetter;
			{ Timer t1;
			offsetter.AddPaths(subject, JoinType::Round, EndType::Polygon);
            std::cout << "Baseline: AddPaths " << pcnt << " polygons: "
                      << t.elapsed_str() << std::endl;
			}
			offsetter.Execute(3, solution, true);
            std::cout << "Baseline: Offset on " << pcnt << " polygons: "
                      << t.elapsed_str() << std::endl;
        }
        {
        	// OpenMP
        	for(auto tm : thread_nums) {
    			Paths64 solution3;
            	Timer t;
            	Clipper2Lib::Offset_OpenMP(subject, 3, solution3, tm, true);

                std::cout << "OpenMP Offset on [" << tm << " threads] " << pcnt << " polygons: "
                          << t.elapsed_str() << std::endl;
        	}
        }
        {
        	// CUDA
        	Timer t;
			ClipperOffset offsetter;
			Paths64 solution2;
			offsetter.AddPaths(subject, JoinType::Round, EndType::Polygon);
			offsetter.Execute_CUDA(3, solution2, true);
            std::cout << "CUDA Offset on " << pcnt << " polygons: "
                      << t.elapsed_str() << std::endl;
        }
	}

	void benchmarks()
	{
		benchmark(1000);
		benchmark(10000);
		benchmark(100000);
		benchmark(1000000);
	}


    void doOffsetTest() {
    	//for (int kk = 0; kk < 10000; ++kk)
		int64_t pcnt = 500000;
    	{

        Paths64 subject = TestGenerator::MakeTestCase(pcnt, 100);
        std::cout << "Test Case generation Done!!!";
        std::vector<int64_t> rp1, rp_cuda;
        Paths64 solution1, solution2;
        {
			//Paths64 solution;
            Timer t;

			ClipperOffset offsetter;
			offsetter.AddPaths(subject, JoinType::Round, EndType::Polygon);
			offsetter.Execute(3, solution1, false);
			// solution = SimplifyPaths(solution, 2.5);
            std::cout << "Baseline: Offset on " << pcnt << " polygons: "
                      << t.elapsed_str() << std::endl;
			rp1 = TestGenerator::GetPathsProp(solution1);

	        //TestGenerator::SaveAndDisplay(solution, "offset1.svg");
        }
        {
			//Paths64 solution;
        	Timer t;
			ClipperOffset offsetter;
			offsetter.AddPaths(subject, JoinType::Round, EndType::Polygon);
			offsetter.Execute_CUDA(3, solution2, false);
			//solution = SimplifyPaths(solution, 2.5);
            std::cout << "CUDA Offset on " << pcnt << " polygons: "
                      << t.elapsed_str() << std::endl;
			rp_cuda = TestGenerator::GetPathsProp(solution2);

	        //TestGenerator::SaveAndDisplay(solution, "offset2.svg");
        }

        {
			Paths64 solution3;
        	Timer t;
        	Clipper2Lib::Offset_OpenMP(subject, 3, solution3, 8, false);

            std::cout << "OpenMP Offset on [8 threads] " << pcnt << " polygons: "
                      << t.elapsed_str() << std::endl;
			rp_cuda = TestGenerator::GetPathsProp(solution3);

	        //TestGenerator::SaveAndDisplay(solution, "offset2.svg");
        }

        std::cout << std::endl;
        std::cout << "Baseline Information" << std::endl;
        for (auto r : rp1) {
        	std::cout << r << ",";
        }
        std::cout << std::endl;
        std::cout << "CUDA Information" << std::endl;
        for (auto r : rp_cuda) {
        	std::cout << r << ",";
        }
        if (rp1[4] != rp_cuda[4]) {

            /*
        	std::cout << "input:" << std::endl;
        	TestGenerator::printPath(subject);
        	std::cout << "output:baseline" << std::endl;
        	TestGenerator::printPath(solution1);
        	std::cout << "output:CUDA" << std::endl;
        	TestGenerator::printPath(solution2);*/
        	std::cout << "Result has difference !!!!" << std::endl;
        	//break;
        } else {
        	std::cout << "Result is correct" << std::endl;
        }
    	}
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
