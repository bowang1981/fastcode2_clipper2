

#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <string>
#include "clipper2/clipper.h"
#include "TestGenerator.h"
#include "RectClipping.h"
#include "../../Utils/clipper.svg.h"
#include "../../Utils/ClipFileLoad.h"
#include "../../Utils/clipper.svg.utils.h"
#include "../../Utils/Colors.h"
#include "../../Utils/Timer.h"
#include "rdtsc.h"

using namespace std;
using namespace Clipper2Lib;
using namespace TestGenerator;

namespace RectClippingTest {
        void System(const std::string& filename);
        void PressEnterToExit();

        void MeasurePerformance(int min, int max, int step);

        const int width = 800, height = 600, margin = 120;

        void DoRectanglesTest(int cnt)
        {
            Paths64 sub, clp, sol, store;
            Rect64 rect = Rect64(margin, margin, width - margin, height - margin);
            clp.push_back(rect.AsPath());
            sub = TestGenerator::CreateRectangles(cnt);

            tsc_counter t0, t1;
            long long sum1 = 0;
            int runs = 10;
            //////////////////////////////////
            for (int i = 0; i < runs; i++) {
                RDTSC(t0);
                sol = RectClip(rect, sub);
                RDTSC(t1);
                sum1 += (COUNTER_DIFF(t1, t0, CYCLES));
            }
            //////////////////////////////////
            std::cout << "RectClipping: RectangleTest: " << static_cast<double>(sum1) / runs << " cycles" << std::endl;

            FillRule fr = FillRule::EvenOdd;
            SvgWriter svg;
            svg.AddPaths(sub, false, fr, 0x100066FF, 0x400066FF, 1, false);
            svg.AddPaths(clp, false, fr, 0x10FFAA00, 0xFFFF0000, 1, false);
            svg.AddPaths(sol, false, fr, 0x8066FF66, 0xFF006600, 1, false);
            svg.SaveToFile("rectclip2.svg", 800, 600, 0);
            // System("rectclip2.svg");
        }


        void DoPolygonTest(int count)
        {
            Paths64 sub, clp, sol;

            // generate random poly
            Rect64 rect = Rect64(margin, margin, width - margin, height - margin);
            clp.push_back(rect.AsPath());
            sub.push_back(MakeRandomPoly(width, height, count));

            tsc_counter t0, t1;
            long long sum1 = 0;
            int runs = 5;
            //////////////////////////////////
            for (int i = 0; i < runs; i++) {
                RDTSC(t0);
                sol = RectClip(rect, sub);
                RDTSC(t1);
                sum1 += (COUNTER_DIFF(t1, t0, CYCLES));
            }
            //////////////////////////////////
            std::cout << "RectClipping: PolygonTest: " << static_cast<double>(sum1) / runs << " cycles" << std::endl;

            FillRule fr = FillRule::EvenOdd;
            double frac = sol.size() ? 1.0 / sol.size() : 1.0;
            double cum_frac = 0;
            SvgWriter svg;
            svg.AddPaths(sub, false, fr, 0x100066FF, 0x800066FF, 1, false);
            svg.AddPaths(clp, false, fr, 0x10FFAA00, 0x80FF0000, 1, false);
            //svg.AddPaths(sol, false, fr, 0x30AAFF00, 0xFF00FF00, 1, false);
            for (const Path64 &sol_path: sol) {
                uint32_t c = RainbowColor(cum_frac, 64);
                cum_frac += frac;
                uint32_t c2 = (c & 0xFFFFFF) | 0x20000000;
                svg.AddPath(sol_path, false, fr, c2, c, 1.2, false);
            }
            svg.SaveToFile("rectclip3.svg", width, height, 0);
            // System("rectclip3.svg");
        }

        void MeasurePerformance(int min, int max, int step)
        {
            FillRule fr = FillRule::EvenOdd;
            Paths64 sub, clp, sol, store;
            Rect64 rect = Rect64(margin, margin, width - margin, height - margin);
            clp.push_back(rect.AsPath());

            for (int cnt = min; cnt <= max; cnt += step) {
                sub.clear();
                sub.push_back(MakeRandomPoly(width, height, cnt));

                std::cout << std::endl << cnt << " random poly" << std::endl;
                {
                    Timer t("Clipper64: ");
                    sol = Intersect(sub, clp, fr);
                }

                {
                    Timer t("RectClip: ");
                    sol = RectClip(rect, sub);
                }

            }

            SvgWriter svg;
            svg.AddPaths(sub, false, fr, 0x200066FF, 0x400066FF, 1, false);
            svg.AddPaths(clp, false, fr, 0x10FFAA00, 0xFFFF0000, 1, false);
            //svg.AddPaths(sol, false, fr, 0x8066FF66, 0xFF006600, 1, false);
            double frac = sol.size() ? 1.0 / sol.size() : 1.0;
            double cum_frac = 0;
            for (const Path64 &sol_path: sol) {
                uint32_t c = RainbowColor(cum_frac, 64);
                cum_frac += frac;
                uint32_t c2 = (c & 0xFFFFFF) | 0x20000000;
                svg.AddPath(sol_path, false, fr, c2, c, 1.2, false);
            }
            svg.SaveToFile("RectClipQ2.svg", 800, 600, 0);
            System("RectClipQ2.svg");
        }

        void System(const std::string& filename)
        {
#ifdef _WIN32
            system(filename.c_str());
#else
            system(("firefox " + filename).c_str());
#endif
        }

        void PressEnterToExit()
        {
            std::string s;
            std::cout << std::endl << "Press Enter to exit" << std::endl;
            std::getline(std::cin, s);
        }

}


