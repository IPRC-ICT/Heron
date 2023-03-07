./benchdnn --matmul --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --mode=p --fix-times-per-prb=100 --wtag=ba 1024x1024:1024x1024
./benchdnn --matmul --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --mode=p --fix-times-per-prb=100 --wtag=ba 4096x4096:4096x4096
./benchdnn --matmul --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --mode=p --fix-times-per-prb=100 --wtag=ba 32x2048:2048x1008
./benchdnn --matmul --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --mode=p --fix-times-per-prb=100 --wtag=ba 32x4096:4096x4096
./benchdnn --matmul --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --mode=p --fix-times-per-prb=100 --wtag=ba 32x4096:4096x1008
