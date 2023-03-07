import torch
import time
import numpy as np
import argparse


def conv2d_cuda(N, C, H, W, K, R, S, stride, padding, dilation, dtype, layout):
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

    if layout == "nhwc":
        model = torch.nn.Conv2d(C, K, (R, S), stride, padding, bias=False).cuda().half()
        model = model.to(memory_format=torch.channels_last)

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
    return np.mean(time_record)

test_shapes = [
    (1, 64, 56, 56, 64, -1, 1, 1, -1, 1, 0, 1, -1),
    (8, 512, 28, 28, 128, -1, 1, 1, -1, 1, 1, 1, -1),

    (16, 1024, 14, 14, 512, -1, 1, 1, -1, 2, 0, 1, -1),
]
test_shapes_nhwc = [
    (32, 512, 7, 7, 512, -1, 3, 3, -1, 1, 0, 1, -1),
    (32, 256, 14, 14, 256, -1, 3, 3, -1, 1, 1, 1, -1),

]

example_text = """
    example:
        python conv2d.py --enable_cudnn --number 5 --repeats 5 --dtype FP16
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--enable_cudnn", action="store_true")
    parser.add_argument("--number", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)

    args = parser.parse_args()

    if args.enable_cudnn:
        assert torch.backends.cudnn.is_available()
        torch.backends.cudnn.enabled = True
    else:
        torch.backends.cudnn.enabled = False

    RUN_NUMBER = (args.number, args.repeats)

    dtype = "FP16"
    reses = []
    for shape in test_shapes:
        (N, C, H, W, K, _, R, S, _, stride, padding, dilation, _) = shape
        config = [N, H, W, C, K, R, S, stride, padding, dilation]
        perf = conv2d_cuda(N, C, H, W, K, R, S, stride, padding, dilation, dtype, layout = 'nchw')
        reses.append([config, perf])
    for shape in test_shapes_nhwc:
        (N, C, H, W, K, _, R, S, _, stride, padding, dilation, _) = shape
        config = [N, H, W, C, K, R, S, stride, padding, dilation]
        perf = conv2d_cuda(N, C, H, W, K, R, S, stride, padding, dilation, dtype, layout= "nhwc")
        reses.append([config, perf])
    for tup in reses:
        print("Case %s, perf %f ms"%(str(tup[0]), tup[1]))
