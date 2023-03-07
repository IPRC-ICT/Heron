python gemm.py > logs/gemm.log 
python conv1d.py --enable_cudnn > logs/c1d.log
python conv2d.py --enable_cudnn > logs/c2d.log
python conv3d.py --enable_cudnn > logs/c3d.log
python dil.py --enable_cudnn > logs/dil.log
python gemv.py > logs/gemv.log
python bmm.py > logs/bmm.log
python scan.py > logs/scan.log
python t2d.py --enable_cudnn > logs/t2d.log
