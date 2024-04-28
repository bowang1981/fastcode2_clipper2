Important:
1. Most of the final changes including the openmp implemented are incuded in the cuda folder. Our testing is based on the folder.
2. The openmp has some study that we have done and inclluded in the mid-term report.
3. We need to use gcc10 and cuda-11.4 to build the cuda folder.
4. Must use cmake3, not the default cmake.

How to build this on ECE machines.
Steps to build baseline(or cuda or openmp, just replace baseline with cuda or openmp in the following steps)
1. Switch to the new compiler with the following command: 
    openmp: scl enable devtoolset-12 bash
or 
cuda: scl enable devtoolset-10 bash
export PATH=${PATH}:/usr/local/cuda-11.4/bin
   Important, cuda11...4 cannot work with gcc12, so we have to use devtoolset-10
so, use the following command for cuda folder:
 scl enable devtoolset-10 bash
2. Go to the baseline/CPP folder, create build_root under that folder:
   cd baseline/CPP;
   mkdir build_root
3. Go to build_root folder: cd build_root
4. Run cmake3 to generate the Makefiles in build_root folder. Make sure you are in build_root folder, for now, we need to disable clipper2's default test due to issue in googletest (I changed CMakefile to disable the TESTS build already):
   cmake3 ..
5. build the library using the make command, make sure you are in build_root folder:
make -j 8
6. build our benchmark folder under CPP/fastcodeBM, make sure you are in build_root/fastcodeBM
cd fastcodeBM;
make

8. run the BM test: ./bm_clip

Code structure
1. We have 3 folders, baseline, openmp, and cuda, the codes are all copied from baseline and we start from there to add modification for openmp and cuda.
2. All the tests are put under fastcodeBM, which will be duplicated at all the 3 folders(baseline, openmp and cuda)
3. To generate the test set, please use the class TestGenerator's functions.

