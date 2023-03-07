./benchdnn --matmul --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --mode=p --fix-times-per-prb=100 --wtag=abc 12x512x64:12x64x512
./benchdnn --matmul --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --mode=p --fix-times-per-prb=100 --wtag=abc 12x512x512:12x512x64
./benchdnn --matmul --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --mode=p --fix-times-per-prb=100 --wtag=abc 16x512x64:16x64x512
./benchdnn --matmul --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --mode=p --fix-times-per-prb=100 --wtag=abc 16x512x512:16x512x64
./benchdnn --matmul --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --mode=p --fix-times-per-prb=100 --wtag=abc 96x2048x128:96x128x2048
./benchdnn --matmul --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --mode=p --fix-times-per-prb=100 --wtag=abc 96x2048x2048:96x2048x128
