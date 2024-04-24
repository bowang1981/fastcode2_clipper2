//
// Created by bowang on 3/30/24.
//

#ifndef FASTCODE2_CLIPPER2_CLIPPER_OPENMP_H
#define FASTCODE2_CLIPPER2_CLIPPER_OPENMP_H
#include "clipper.h"
#include <omp.h>
namespace Clipper2Lib {
     Paths64 Union_OpenMP(const Paths64& subjects, FillRule fillrule, int thread_num = 4);

     double Area_OpenMP(const Path64& path, int num_t = 4);

// In this version, we do the parallization in each polygon. We'll have another version on high level.
     double Area_OpenMP(const Paths64& paths, int thread_num);


}

#endif //FASTCODE2_CLIPPER2_CLIPPER_OPENMP_H
