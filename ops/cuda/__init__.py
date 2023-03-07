# Tensor core ops
from .conv1d import *
from .conv2d import *
from .conv2d_transposed import *
from .conv3d import *
from .batch_matmul import *
from .dense import *
from .gemv import *
from .mean import *
from .var import *
from .scan import *
from .dil import *
from .dep import *

def getOpFromName(name):
    opmap = {
    "gemv" : heron_gemv,
    "gemm" : heron_dense,
    "c1d"  : heron_conv1d_ncw_tensorcore,
    "c2d"  : heron_conv2d_nchw_tensorcore,
    "c3d"  : heron_conv3d_ncdhw_tensorcore,
    "bmm"  : heron_batch_matmul, 
    "t2d"  : heron_conv2d_nchw_transposed_tensorcore,
    "mean" : heron_mean,
    "var" : heron_var,
    "scan" : heron_scan,
    "dil" : heron_dil_tensorcore,
    "dep" : heron_depthwise_conv2d_tensorcore,
    # Others
    "c2d_nhwc": heron_conv2d_nhwc_tensorcore, 
    }
    return opmap[name]
