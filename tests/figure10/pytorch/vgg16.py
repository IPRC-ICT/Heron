import torch
import time
import numpy as np
import argparse


def conv2d_cuda(N, C, H, W, K, R, S, stride, padding, dilation, dtype):
    A_np = np.random.uniform(-10, 10, [N, C, H, W]).astype("float32")
    B_np = np.random.uniform(-10, 10, [K, C, R, S]).astype("float32")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if dtype == "FP16":  # HMMA-16, torch.float16 or torch.half
        A_torch = torch.tensor(A_np).type(torch.float16).cuda()
        B_torch = torch.tensor(B_np).type(torch.float16).cuda()
    elif dtype == "BF16":  # HMMA-16, only on NVIDIA A100, torch.bfloat16
        A_torch = torch.tensor(A_np).type(torch.bfloat16).cuda()
        B_torch = torch.tensor(B_np).type(torch.bfloat16).cuda()
    elif dtype == "FP32":
        A_torch = torch.tensor(A_np).type(torch.float32).cuda()
        B_torch = torch.tensor(B_np).type(torch.float32).cuda()
    elif dtype == "TF32":  # HMMA-19, NVIDIA A100
        # Please upgrade torch to 1.7; only supported on A100
        # https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        A_torch = torch.tensor(A_np).type(torch.float32).cuda()
        B_torch = torch.tensor(B_np).type(torch.float32).cuda()
    elif dtype == "INT8":  # IMMA, but pytorch has no support for INT8 GEMM
        A_torch = torch.tensor(A_np).type(torch.int8).cuda()
        B_torch = torch.tensor(B_np).type(torch.int8).cuda()
    # Pytorch has no int4 type
    elif dtype == "BOOL":  # BMMA, but pytorch has no support for GEMM GEMM
        A_torch = torch.tensor(A_np).type(torch.bool).cuda()
        B_torch = torch.tensor(B_np).type(torch.bool).cuda()
    elif dtype == "FP64":  # DMMA(FP64), only supported on A100
        A_torch = torch.tensor(A_np).type(torch.float64).cuda()
        B_torch = torch.tensor(B_np).type(torch.float64).cuda()
    else:
        assert False, "wrong type: " + dtype

    global RUN_NUMBER
    number, repeats = RUN_NUMBER

    time_record = []
    for i in range(repeats):
        for j in range(number):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            C_torch = torch.nn.functional.conv2d(
                A_torch, B_torch, bias=None, stride=stride, padding=padding, dilation=dilation
            )

            end.record()
            torch.cuda.synchronize()
            total = start.elapsed_time(end)
            time_record.append(total)
    mean_cost = np.mean(time_record)
    return mean_cost


test_shapes = [
            [1, [16, 224, 224, 3   , 64  , 3, 3, 1, 1, 1]],
            [1, [16, 224, 224, 64  , 64  , 3, 3, 1, 1, 1]],
            [1, [16, 112, 112, 64  , 128 , 3, 3, 1, 1, 1]],
            [2, [16, 112, 112, 128 , 128 , 3, 3, 1, 1, 1]],
            [1, [16, 56 , 56 , 128 , 256 , 3, 3, 1, 1, 1]],
            [3, [16, 56 , 56 , 256 , 256 , 3, 3, 1, 1, 1]],
            [1, [16, 28 , 28 , 256 , 512 , 3, 3, 1, 1, 1]],
            [3, [16, 28 , 28 , 512 , 512 , 3, 3, 1, 1, 1]],
            [1, [16, 14 , 14 , 512 , 512 , 3, 3, 1, 1, 1]],
            [3, [16, 7  , 7  , 512 , 512 , 3, 3, 1, 1, 1]] 

]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=4)

    args = parser.parse_args()

    assert torch.backends.cudnn.is_available()
    torch.backends.cudnn.enabled = True

    RUN_NUMBER = (args.number, args.repeats)

    args = parser.parse_args()
    total_time = 0
    dtype = "FP16"
    for i, tup in enumerate(test_shapes):
        times, shape = tup
        (N, H, W, C, K, R, S, stride, padding, dilation) = shape
        perf = conv2d_cuda(N, C, H, W, K, R, S, stride, padding, dilation, dtype)
        total_time += times * perf
    print("Total time : %f"%total_time)

