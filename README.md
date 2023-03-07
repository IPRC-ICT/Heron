# Heron

## Introduction

Automatically generation of performant programs for deep learning accelerators (DLA) extremely difficult. Heron solve this problem by automatically generating a constrained search space and explore it with a novel constrained genetic algorithm(CGA). 

## Installation
**Common requirements**
```
Python 3.6.10
```
Please run the following command to install the dependencies.
```shell
pip install -r requirements.txt
```

**Requirements for TensorCore code generation.** Make sure that your platform has GPUs with TensorCore available. Tested platform includes V100, T4, and A100.

First, check the dependancies needed.

```
CUDA11.2
```

Second, build TVM for gpu. You can follow [TVM build from source instructions](https://tvm.apache.org/docs/install/from_source.html) for details.


**Requirements for DL Boost code generation.**
 Make sure that your platform has CPUs with DL Boost available. We conducted our experiments on Intel’s Xeon Gold 6240 CPU.

First, check the dependancies needed.

```
# other versions may have errors related to intrinsic functions.
llvm 8.0.0
```

Second, build TVM for cpu. You can follow [TVM build from source instructions](https://tvm.apache.org/docs/install/from_source.html) for details.


## Tuning

**A quick start for TensorCore tuning.** 
Take gemm operation with (m,k,n) =(64, 64, 64) as an example, we show how to automatically schedule with Heron.

```shell
cd tests/quick_start/tensorcore
python run.py -p tensorcore -c quick_start.json
```

Then, you can get the following results in 2 minutes.
```shell
PASS
Case [64, 64, 64], latency 0.002236 ms.
```

**A quick start for DL Boost tuning.**
 Take gemm operation with (m,k,n) =(64, 64, 64) as an example, we show how to automatically schedule with Heron.
```shell
cd tests/quick_start/dlboost
python run.py -p dlboost -c quick_start.json
```

Then, you can get the following results in 2 minutes.
```shell
PASS
Case [64, 64, 64], latency 0.002658 ms.
```
**Complete evaluation**
For complete evaluation, please run the corresponding scripts. For example, to evaluate TensorCore operator performances shown in Figure 6, please use the following commands:
```shell
cd tests/Figure6
python run.y -c path/to/test_cases.json
``` 
