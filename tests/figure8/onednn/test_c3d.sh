./benchdnn --conv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_I --mode=p --fix-times-per-prb=100 --stag=ndhwc --wtag=AcdeB64a4b --dtag=ndhwc mb1_ic64oc64_id8od8kd3sd1dd1pd1_ih56oh56kh3sh1dh1ph1_iw56ow56kw3sw1dw1pw1
./benchdnn --conv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_I --mode=p --fix-times-per-prb=100 --stag=ndhwc --wtag=AcdeB64a4b --dtag=ndhwc mb1_ic64oc128_id8od4kd1sd2dd1pd0_ih56oh28kh1sh2dh1ph0_iw56ow28kw1sw2dw1pw0
./benchdnn --conv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_I --mode=p --fix-times-per-prb=100 --stag=ndhwc --wtag=AcdeB64a4b --dtag=ndhwc mb1_ic128oc128_id4od5kd3sd1dd1pd1_ih28oh29kh3sh1dh1ph1_iw28ow29kw3sw1dw1pw1
./benchdnn --conv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_I --mode=p --fix-times-per-prb=100 --stag=ndhwc --wtag=AcdeB32a4b --dtag=ndhwc mb1_ic128oc256_id4od2kd1sd2dd1pd0_ih28oh14kh1sh2dh1ph0_iw28ow14kw1sw2dw1pw0
./benchdnn --conv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_I --mode=p --fix-times-per-prb=100 --stag=ndhwc --wtag=AcdeB64a4b --dtag=ndhwc mb1_ic256oc256_id2od3kd3sd1dd1pd1_ih14oh15kh3sh1dh1ph1_iw14ow15kw3sw1dw1pw1
./benchdnn --conv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_I --mode=p --fix-times-per-prb=100 --stag=ndhwc --wtag=AcdeB48a4b --dtag=ndhwc mb1_ic256oc512_id2od1kd1sd2dd1pd0_ih14oh7kh1sh2dh1ph0_iw14ow7kw1sw2dw1pw0