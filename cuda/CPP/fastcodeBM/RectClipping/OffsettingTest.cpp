//
// Created by bowang on 3/29/24.
//

#include "OffsettingTest.h"
#include "clipper2/clipper.h"
#include "TestGenerator.h"
#include "../../Utils/Timer.h"
using namespace Clipper2Lib;
namespace OffsettingTest {

    void doOffsetTest() {
    	//for (int kk = 0; kk < 10000; ++kk)
		int64_t pcnt = 1000000;
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
			offsetter.Execute(3, solution1, true);
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
			offsetter.Execute_CUDA(3, solution2, true);
			//solution = SimplifyPaths(solution, 2.5);
            std::cout << "CUDA Offset on " << pcnt << " polygons: "
                      << t.elapsed_str() << std::endl;
			rp_cuda = TestGenerator::GetPathsProp(solution2);

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
