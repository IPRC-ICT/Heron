./benchdnn --matmul --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --mode=p --fix-times-per-prb=100 --wtag=ba 1x16:16x512
./benchdnn --matmul --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --mode=p --fix-times-per-prb=100 --wtag=ba 1x256:256x1024
./benchdnn --matmul --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --mode=p --fix-times-per-prb=100 --wtag=ba 1x1024:1024x1256
./benchdnn --matmul --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --mode=p --fix-times-per-prb=100 --wtag=ba 1x256:256x512
./benchdnn --matmul --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --mode=p --fix-times-per-prb=100 --wtag=ba 1x1024:1024x1024
