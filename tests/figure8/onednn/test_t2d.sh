./benchdnn --deconv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_D --mode=p --fix-times-per-prb=100 --stag=nhwc --wtag=any --dtag=nhwc mb1_ic64oc64_ih56oh56kh3sh1dh1ph1_iw56ow56kw3sw1dw1pw1
./benchdnn --deconv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_I --mode=p --fix-times-per-prb=100 --stag=nhwc --wtag=any --dtag=nhwc mb1_ic64oc64_ih56oh56kh1sh1dh1ph0_iw56ow56kw1sw1dw1pw0
./benchdnn --deconv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_I --mode=p --fix-times-per-prb=100 --stag=nhwc --wtag=any --dtag=nhwc mb1_ic128oc128_ih28oh28kh3sh1dh1ph1_iw28ow28kw3sw1dw1pw1
./benchdnn --deconv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_I --mode=p --fix-times-per-prb=100 --stag=nhwc --wtag=any --dtag=nhwc mb1_ic128oc256_ih28oh56kh1sh2dh1ph0_iw28ow56kw1sw2dw1pw0
./benchdnn --deconv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_I --mode=p --fix-times-per-prb=100 --stag=nhwc --wtag=any --dtag=nhwc mb1_ic256oc256_ih14oh14kh3sh1dh1ph1_iw14ow14kw3sw1dw1pw1
./benchdnn --deconv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_I --mode=p --fix-times-per-prb=100 --stag=nhwc --wtag=any --dtag=nhwc mb1_ic256oc512_ih14oh28kh1sh2dh1ph0_iw14ow28kw1sw2dw1pw0
