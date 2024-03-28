//
// Created by bowang on 3/28/24.
//
#include "TestGenerator.h"
#include "RectClipping.h"

int main(int argc, char* argv[])
{
    srand((unsigned)time(0));
    RectClippingTest::DoRectanglesTest(500);
    RectClippingTest::DoPolygonTest(500);
}
