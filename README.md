Steps to build baseline
1. Switch to the new compiler with the following command
    scl enable devtoolset-12 bash
2. Go to the baseline/CPP folder, create build_root under that folder
   cd baseline/CPP;
   mkdir build_root
3. Go to build_root folder: cd build_root
4. Run cmake3 to generate the Makefiles in build_root folder. Make sure you are in build_root folder, for now, we need to disable clipper2 due to issue in googletest
   cmake3 .. -DCLIPPER2_TESTS=OFF
5. build the library using the make command, make sure you are in build_root folder
make -j 8

