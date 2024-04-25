//
// Created by bowang on 3/28/24.
//

#include "TestGenerator.h"
#include "../../Utils/clipper.svg.h"
#include "../../Utils/ClipFileLoad.h"
#include "../../Utils/clipper.svg.utils.h"

using namespace Clipper2Lib;

namespace TestGenerator {
    Path64 MakeRandomRectangle(int minWidth, int minHeight, int maxWidth, int maxHeight,
                               int maxRight, int maxBottom) {
        int w = maxWidth > minWidth ? minWidth + rand() % (maxWidth - minWidth) : minWidth;
        int h = maxHeight > minHeight ? minHeight + rand() % (maxHeight - minHeight) : minHeight;
        int l = rand() % (maxRight - w);
        int t = rand() % (maxBottom - h);
        Path64 result;
        result.reserve(4);
        result.push_back(Point64(l, t));
        result.push_back(Point64(l + w, t));
        result.push_back(Point64(l + w, t + h));
        result.push_back(Point64(l, t + h));
        return result;
    }

    Paths64 CreateRectangles(int cnt) {
        Paths64 sub;

        for (int i = 0; i < cnt; ++i)
            sub.push_back(MakeRandomRectangle(10, 10,
                                              100, 100,
                                              800, 600));

        return sub;
    }

    Path64 MakeRandomPoly(int width, int height, unsigned vertCnt)
    {
        Path64 result;
        result.reserve(vertCnt);
        for (unsigned i = 0; i < vertCnt; ++i)
            result.push_back(Point64(rand() % width, rand() % height));
        return result;
    }

    Paths64 CreatePolygons(int polygon_cnt, int vtxcount) {
        Paths64 sub;

        for (int i = 0; i < polygon_cnt; ++i)
            sub.push_back(MakeRandomPoly(10, 10, vtxcount));

        return sub;
    }

    Clipper2Lib::Path64 MakeNoSelfIntesectPolygon(int64_t width, int64_t height, int64_t ctn,
    		int64_t x_offset, int64_t y_offset)
    {
    	if (ctn < 3) {
    		ctn = 3;
    	}
    	if (width <= 0) {
    		width =30;
    	}
    	if (height <= 0) {
    		height = 30;
    	}

        Path64 result;
        result.reserve(ctn);
        int64_t mid_y = height / 2;
        int64_t x_step = 2 * width / (ctn);
        if(x_step < 5) x_step = 5;

        int64_t max_x = (width / x_step) * x_step ;

        for (int64_t x = 0; x <= max_x; x = x+x_step) {
            int64_t y = mid_y + (rand() % height) / 2;
            result.push_back(Point64(x + x_offset, y+y_offset));
        }

        for (int64_t x = max_x; x >= 0; x = x-x_step) {
            int64_t y = mid_y - (rand() % height) / 2;
            result.push_back(Point64(x + x_offset, y + y_offset));
        }

        return result;

    }


    Clipper2Lib::Paths64 MakeTestCase(int64_t cnt,  int margin)
    {
    	Paths64 paths;;
    	int64_t x_off =0, y_off = 0;
    	int64_t rcnt = static_cast<int64_t>(sqrt(cnt));
    	int64_t ccnt = cnt / rcnt + 1;
    	for (int64_t r = 0; r < rcnt; ++r) {
    		x_off = 0;
    		int64_t height = (rand() % 300);
    		for (int64_t c = 0; c < ccnt; ++c) {
    			int64_t width = (rand() % 300);
    			Path64 p = MakeNoSelfIntesectPolygon(width, height, rand() % 50,
    			    		x_off, y_off);
    			if (p.size() >= 3) {
    			paths.push_back(p);
    			}
    			x_off += (width+margin);

    		}
    		y_off += (height+margin);
    	}
    	return paths;
    }

    Clipper2Lib::Paths64 MakeNoSelfIntesectPolygons(int64_t polygon_ctn, int64_t width,
    		                                   int64_t height, int64_t vertCtn)
    {
        Paths64 result;
        result.reserve(polygon_ctn);

        for (int i = 0; i < polygon_ctn; ++i) {
            int w1 = (rand() % width) + 1;
            int h1 = (rand() % height) + 1;
            if (w1 < 100) w1 = 100;
            if (h1 < 100) h1 = 100;
            result.push_back(MakeNoSelfIntesectPolygon(w1, h1, vertCtn));
        }
        return result;
    }

    void System(const std::string& filename)
    {
#ifdef _WIN32
        system(filename.c_str());
#else
        system(("firefox " + filename).c_str());
#endif
    }

    void SaveAndDisplay(Clipper2Lib::Paths64& paths,  const std::string& filepath,
    		            int w, int h,
                        Clipper2Lib::FillRule fillrule)
    {
        SvgWriter svg;
        SvgAddSubject(svg, paths, fillrule);
        SvgAddSolution(svg, paths, fillrule, false);
        SvgSaveToFile(svg, filepath, w, h, 10);
        System(filepath);
    }
    Clipper2Lib::Paths64 CreateTestCase_1K(int margin)
    {
    	return MakeTestCase(1000, margin);
    }

    Clipper2Lib::Paths64 CreateTestCase_100K(int margin)
    {
    	return MakeTestCase(100000, margin);
    }
    Clipper2Lib::Paths64 CreateTestCase_1M(int margin) {
    	return MakeTestCase(1000000, margin);
    }
    Clipper2Lib::Paths64 CreateTestCase_5M(int margin) {
    	return MakeTestCase(5000000, margin);
    }

    std::vector<int64_t> GetPathsProp(const Clipper2Lib::Paths64& paths)
	{
    	int64_t min_x = paths[0][0].x, min_y=paths[0][0].y;
    	int64_t max_x = paths[0][0].x, max_y = paths[0][0].y;
    	double sum_avg = 0;
    	for (auto path : paths) {
    		double avg = 0;
    		for (auto p : path) {
    			min_x = std::min(min_x, p.x);
    			min_y = std::min(min_y, p.y);
    			max_x = std::max(max_x, p.x);
    			max_y = std::max(max_y, p.y);
    			avg += p.x;
    			avg += p.y;
    		}
    		sum_avg += (avg / path.size());
    	}
    	std::vector<int64_t> res;
    	res.push_back(min_x);
    	res.push_back(min_y);
    	res.push_back(max_x);
    	res.push_back(max_y);
    	res.push_back(int64_t(sum_avg / paths.size()));
    	return res;
	}

    void printPath(const Clipper2Lib::Paths64& paths)
    {
    	for(auto pa : paths) {
    		std::cout << "{";
    		for (auto p : pa) {
    			std::cout<< p << ",";
    		}
    		std::cout << "}" << std::endl;
    	}
    }
} // end of namespace
