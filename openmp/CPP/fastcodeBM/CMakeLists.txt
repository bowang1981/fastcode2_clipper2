add_executable(bm_clip RectClipping/RectClipping.cpp RectClipping/TestGenerator.cpp
        RectClipping/AreaCalcTest.cpp
        RectClipping/OffsettingTest.cpp RectClipping/UnionTest.cpp RectClipping/main.cpp RectClipping/rdtsc_works.cpp)
target_link_libraries(bm_clip PRIVATE Clipper2 Clipper2utils)
