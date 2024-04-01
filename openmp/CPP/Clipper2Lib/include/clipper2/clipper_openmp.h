//
// Created by bowang on 3/30/24.
//

#ifndef FASTCODE2_CLIPPER2_CLIPPER_OPENMP_H
#define FASTCODE2_CLIPPER2_CLIPPER_OPENMP_H
#include "clipper.h"
#include <omp.h>
namespace Clipper2Lib {
    inline Paths64 Union_OpenMP(const Paths64& subjects, FillRule fillrule, int thread_num = 4)
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

    inline double Area_OpenMP(const Path64& path, int num_t = 4)
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


}

#endif //FASTCODE2_CLIPPER2_CLIPPER_OPENMP_H
