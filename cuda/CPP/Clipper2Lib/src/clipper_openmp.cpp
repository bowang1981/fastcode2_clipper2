#include <omp.h>
#include "clipper2/clipper_openmp.h"
namespace Clipper2Lib {
    inline Paths64 Union_OpenMP(const Paths64& subjects, FillRule fillrule, int thread_num )
    {
        std::vector<Paths64> subjectsVec, resultsVec;
        int sz1 = subjects.size() / thread_num + 1;
        for (int i = 0; i < thread_num; ++i) {
            Paths64 tmpObjs;
            for (int j = i * sz1; j < (i+1) * sz1 && j < subjects.size(); ++j) {
                tmpObjs.push_back(subjects[j]);
            }
            subjectsVec.push_back(tmpObjs);
        }
        #pragma omp parallel for num_threads(thread_num)
        for ( int i = 0; i < thread_num; ++i) {
            Paths64 result1;
            Clipper64 clipper;
            clipper.AddSubject(subjectsVec[i]);
            clipper.Execute(ClipType::Union, fillrule, result1);
            #pragma omp critical
            resultsVec.push_back(result1);
        }

        Paths64 result;
        Clipper64 clipper;
        for (int i = 0; i < resultsVec.size(); ++i) {
            clipper.AddSubject(resultsVec[i]);
        }
        clipper.Execute(ClipType::Union, fillrule, result);
        return result;
    }

    inline double Area_OpenMP(const Path64& path, int num_t)
    {
        size_t cnt = path.size();
        if (cnt < 3) return 0.0;
        double a = 0.0;
        // if (!(cnt & 1)) ++stop;
        if (cnt < 20) num_t = 1;
        #pragma  omp parallel num_threads(num_t)
        {
            int total_threads = omp_get_num_threads();

            int id = omp_get_thread_num();
            int patch_size = cnt / total_threads;
            patch_size = (patch_size / 2) * 2 + 2;
            double a1 = 0.0;
            for (int i = id * patch_size; i < cnt && i < (id+1) * patch_size; i = i + 2) {
                int prev = i - 1;
                if (i == 0) prev = cnt - 1;

                a1 += static_cast<double>(path[prev].y + path[i].y) * (path[prev].x - path[i].x);
                int next = i + 1;
                a1 += static_cast<double>(path[i].y + path[next].y) * (path[i].x - path[next].x);
            }

            #pragma omp atomic update
            a+= a1;
        }
        /*
        for (it1 = path.cbegin(); it1 != stop;)
        {
            a += static_cast<double>(it2->y + it1->y) * (it2->x - it1->x);
            it2 = it1 + 1;
            a += static_cast<double>(it1->y + it2->y) * (it1->x - it2->x);
            it1 += 2;
        }

         */
       if (cnt & 1)
           a += static_cast<double>(path[cnt-2].y + path[cnt-1].y) * (path[cnt-2].x - path[cnt-1].x);
        return a * 0.5;
    }

// In this version, we do the parallization in each polygon. We'll have another version on high level.
    inline double Area_OpenMP(const Paths64& paths, int thread_num)
    {
        double a = 0.0;
        for (Paths64::const_iterator paths_iter = paths.cbegin();
             paths_iter != paths.cend(); ++paths_iter)
        {
            a += Area_OpenMP(*paths_iter, thread_num);
        }
        return a;
    }

    // this will only process overlaping rectangles
    inline void RectClip_OpenMP(const Paths64& paths, Rect64& rect, Paths64& result)
    {
        RectClip64 rectClipper(rect);
        #pragma omp parallel for num_threads(32)
        for (const Path64& path : paths)
        {
        Rect64 path_bounds = GetBounds(path);
        OutPt2List partial_results = OutPt2List();
        std::vector<Location> start_locs = std::vector<Location>();
        OutPt2List edges[8];
        std::deque<OutPt2> op_container = std::deque<OutPt2>();
        rectClipper.ExecuteInternal(path, path_bounds, start_locs, partial_results, op_container, edges);
        rectClipper.CheckEdges(partial_results, edges);
        for (int i = 0; i < 4; ++i)
            rectClipper.TidyEdges(i, edges[i * 2], edges[i * 2 + 1], partial_results);

        Paths64 tmpResults;
        for (OutPt2*& op : partial_results)
        {
            Path64 tmp = rectClipper.GetPath(op);
            if (!tmp.empty())
            {
                tmpResults.emplace_back(tmp);
            }
        } 
        // The algorithm is already implemented, the critical section below is only for the display process that need only one final result
        // if we don't need to display it, or we have function to display the partial at the same time, we can remove the critical section
        // which means this not necessary in time calculation.

        #pragma omp critical
        {
            for (Path64 tmp : tmpResults)
            {
                result.emplace_back(tmp);
            }
        }

        }
    }

}
