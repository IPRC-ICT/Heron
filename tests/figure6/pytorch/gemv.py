import torch
import argparse
import numpy as np
print(torch.version.cuda)

def gemm(H, W, K, dtype):
  A_np = np.random.uniform(-10, 10, [H, K]).astype("float32")
  B_np = np.random.uniform(-10, 10, [K, W]).astype("float32")

  # What's supported by NVIDIA? Refer to https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html

  # What's supported by pytorch? I don't know
  # Please sudo nvprof them!

  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False

  if dtype == "FP16": # HMMA-16, torch.float16 or torch.half
    A_torch = torch.tensor(A_np).type(torch.float16).cuda()
    B_torch = torch.tensor(B_np).type(torch.float16).cuda()
  elif dtype == "BF16": # HMMA-16, only on NVIDIA A100, torch.bfloat16
    A_torch = torch.tensor(A_np).type(torch.bfloat16).cuda()
    B_torch = torch.tensor(B_np).type(torch.bfloat16).cuda()
  elif dtype == "FP32":
    A_torch = torch.tensor(A_np).type(torch.float32).cuda()
    B_torch = torch.tensor(B_np).type(torch.float32).cuda()
  elif dtype == "TF32": # HMMA-19, NVIDIA A100
    # Please upgrade torch to 1.7; only supported on A100
    # https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    A_torch = torch.tensor(A_np).type(torch.float32).cuda()
    B_torch = torch.tensor(B_np).type(torch.float32).cuda()
  elif dtype == "INT8": # IMMA, but pytorch has no support for INT8 GEMM
    A_torch = torch.tensor(A_np).type(torch.int8).cuda()
    B_torch = torch.tensor(B_np).type(torch.int8).cuda()
  # Pytorch has no int4 type
  elif dtype == "BOOL": # BMMA, but pytorch has no support for GEMM GEMM
    A_torch = torch.tensor(A_np).type(torch.bool).cuda()
    B_torch = torch.tensor(B_np).type(torch.bool).cuda()
  elif dtype == "FP64": # DMMA(FP64), only supported on A100
    A_torch = torch.tensor(A_np).type(torch.float64).cuda()
    B_torch = torch.tensor(B_np).type(torch.float64).cuda()

  global RUN_NUMBER
  number, repeats = RUN_NUMBER

  time_record = []
  for i in range(repeats):
      for j in range(number):
          torch.cuda.synchronize()
          start = torch.cuda.Event(enable_timing=True)
          end = torch.cuda.Event(enable_timing=True)
          start.record()

          C_torch = torch.mm(A_torch, B_torch)

          end.record()
          torch.cuda.synchronize()
          total = start.elapsed_time(end)
          time_record.append(total)
  return np.mean(time_record)

test_shapes = [
        [512, 16, 1],
        [1024, 256, 1],
        [256, 1024, 1],
        [512, 256, 1],
        [1024, 1024, 1]
        ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable_cudnn", action="store_true")
    parser.add_argument("--number", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=4)

    args = parser.parse_args()

    if args.enable_cudnn:
        assert torch.backends.cudnn.is_available()
        torch.backends.cudnn.enabled = True
    else:
        torch.backends.cudnn.enabled = False

    RUN_NUMBER = (args.number, args.repeats)

    # H, W, K for [H, K] * [K, W]
    reses = []
    dtype = "FP16"
    for shape in test_shapes:
        H, K, W = shape
        perf = gemm(H, W, K, dtype)
        reses.append([shape, perf])

    for tup in reses:
        print("Case %s, perf %f ms"%(str(tup[0]), tup[1]))
