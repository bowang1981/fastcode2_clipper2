/*
 * clipper.rectclip.cuh
 *
 *  Created on: Apr 18, 2024
 *      Author: bow2
 */

#ifndef CLIPPER_RECTCLIP_CUH_
#define CLIPPER_RECTCLIP_CUH_

#include <stdio.h>
#include <vector>
#include "clipper.core.h"
#include "clipper.core.cuh"
#include "clipper.area.cuh"
// #include "clipper.h"
//#include "clipper.rectclip.h"
// #include "clipper.offset.h"

namespace Clipper2Lib {

    // TODO: to be checked ======
    class cuOutPt2;
    struct cuOutPt2List{
        __host__ cuOutPt2List();
        __host__ void init(const OutPt2List &outpt2list);
        __host__ void init(int sz);
        __host__ ~cuOutPt2List();
        cuOutPt2* list;
        int size;
    };

    class cuOutPt2 {
        public:
            __host__ cuOutPt2();
            __host__ cuOutPt2 init(const OutPt2 &outpt2);
            cuPoint64 pt;
            size_t owner_idx;
            cuOutPt2List* edge;
            cuOutPt2* next;
            cuOutPt2* prev;
        };
    // ============
    
    void rectclip_execute(const Paths64& input, const Rect64& rect, Paths64& output);

}



#endif /* CLIPPER_RECTCLIP_CUH_ */
