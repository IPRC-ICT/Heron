export OMP_NUM_THREADS=18
./test_gemm.sh > logs/gemm.log
./test_gemv.sh > logs/gemv.log
./test_bmm.sh > logs/bmm.log
./test_c1d.sh > logs/c1d.log
./test_c2d.sh > logs/c2d.log
./test_c3d.sh > logs/c3d.log
./test_dil.sh > logs/dil.log
./test_t2d.sh > logs/t2d.log
