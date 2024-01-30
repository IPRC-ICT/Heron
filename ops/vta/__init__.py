from .conv2d import * 
from .dense import * 
#from .batch_matmul import *
def getOpFromName(name):
    opmap = {
    "gemm" : heron_dense,
   #"c2d"  : heron_conv2d_nhwc,
   #"bmm"  : heron_batch_matmul, 
    }
    return opmap[name]
