from .conv2d_int8 import *
from .conv3d_int8 import *
from .conv1d_int8 import *
from .dense import *
from .batch_matmul import *
from .dil import *
from .gemv import *
from .t2d import *
from .scan import *

def getOpFromName(name):
    opmap = {
    "gemm" : heron_dense,
    "gemv" : heron_gemv,
    "c1d"  : heron_conv1d_nwc,
    "c2d"  : heron_conv2d_nhwc,
    "c3d"  : heron_conv3d_ndhwc,
    "bmm"  : heron_batch_matmul, 
    "dil"  : heron_dil_nhwc,
    "t2d"  : heron_conv2d_transposed_nhwc,
    "scan" : heron_scan
    }
    return opmap[name]
