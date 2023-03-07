./benchdnn --conv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_I --mode=p --fix-times-per-prb=100 --stag=acb --wtag=AcB64a4b --dtag=acb mb1_ic512oc1024_iw892ow892kw1sw1dw1pw0
./benchdnn --conv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_I --mode=p --fix-times-per-prb=100 --stag=acb --wtag=AcB64a4b --dtag=acb mb1_ic1024oc512_iw892ow892kw1sw1dw1pw0
./benchdnn --conv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_I --mode=p --fix-times-per-prb=100 --stag=acb --wtag=AcB64a4b --dtag=acb mb1_ic1024oc512_iw892ow892kw1sw1dw1pw0
./benchdnn --conv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_I --mode=p --fix-times-per-prb=100 --stag=acb --wtag=AcB64a4b --dtag=acb mb1_ic512oc512_iw892ow898kw3sw1dw2pw4
./benchdnn --conv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_I --mode=p --fix-times-per-prb=100 --stag=acb --wtag=AcB64a4b --dtag=acb mb1_ic512oc512_iw892ow904kw3sw1dw4pw8
./benchdnn --conv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_I --mode=p --fix-times-per-prb=100 --stag=acb --wtag=AcB64a4b --dtag=acb mb1_ic512oc512_iw892ow916kw3sw1dw8pw16
