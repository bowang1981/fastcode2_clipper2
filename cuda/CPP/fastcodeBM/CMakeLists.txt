add_executable(bm_clip RectClipping/RectClipping.cpp RectClipping/TestGenerator.cpp
		RectClipping/AreaCalcTest.cpp  RectClipping/UnionTest.cpp
        RectClipping/OffsettingTest.cpp RectClipping/main.cpp)
target_link_libraries(bm_clip PRIVATE Clipper2 Clipper2utils)

