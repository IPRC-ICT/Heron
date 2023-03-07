
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ops = ["gemm", "c1d", "c2d", "c3d", "t2d", "dil", "bmm", "gemv", "scan"]

def geo_mean(x):
    return x.prod()**(1.0/len(x))

def get_heron_perfs():
    res = []
    for op in ops:
        opname = op + ".json.log"
        temp = []
        with open(opname, 'rb') as f:
            for line in f.readlines():
                line = str(line.strip())
                if "latency" not in line:
                    continue
                perf = float(line.split("latency")[1].split("ms")[0])
                temp.append(perf)
        res.append(temp)
    return res

def get_pytorch_perfs():
    res = []
    for op in ops:
        opname = "pytorch/logs/" + op + ".log"
        temp = []
        with open(opname, 'rb') as f:
            for line in f.readlines():
                line = str(line.strip())
                if "perf" not in line:
                    continue
                perf = float(line.split("perf")[1].split("ms")[0])
                temp.append(perf)
        res.append(temp)
    return res


if __name__ == "__main__":
    heron = get_heron_perfs()
    pytorch = get_pytorch_perfs()
    heron_relative = [1]*len(heron)
    pytorch_relative = [geo_mean(np.array(x1) / np.array(x2)) for x1, x2 in zip(heron, pytorch)] 

    heron_relative.append(1)
    pytorch_relative.append(geo_mean(np.array(pytorch_relative)))
    ops.append("GEO")

    width = 0.4
    x = np.arange(len(ops)).astype(float)
    plt.figure(figsize=(8,4))
    plt.bar(x, pytorch_relative, width =width, label='PyTorch', fc='y')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, heron_relative, width =width, tick_label=ops, label='Heron', fc='b')
    plt.legend()
    plt.savefig('figure6.png')
    plt.close()
    
