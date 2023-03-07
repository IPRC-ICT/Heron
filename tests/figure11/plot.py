import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def heat_map(name, data, vmin = 0, vmax = 60, axis_max = 50, axis_cell = 1):
    data = data[:2000, :]
    plt.figure(figsize=(6, 4), dpi=300)
    mem1 = data[:, 0] * 2 / 1024
    mem2 = data[:, 1] * 2 / 1024
    perfs = data[:,2]
    array_dim = int(axis_max // axis_cell)
    array = np.zeros((40, array_dim))
    for idx, perf in enumerate(perfs.tolist()):
        x = int(mem1[idx] // axis_cell)
        y = int(mem2[idx] // axis_cell)
        if x >= 35 :
            continue
        array[x,y] = max(perf, array[x,y])
    h = sns.heatmap(array, vmin = vmin, vmax = vmax, cmap = "viridis", cbar=False,\
                    linewidths=0.5, linecolor='black', xticklabels=10, yticklabels=10)
  # cb = h.figure.colorbar(h.collections[0])
    plt.yticks(fontproperties = 'Calibri', size = 14, weight='bold')
    plt.xticks(fontproperties = 'Calibri', size = 14, weight='bold')
    plt.savefig('space_%s.png'%(name))
    plt.close()


heron = np.loadtxt("heron.txt")
heron[:,2] = (2 * 1024 * 1024 * 1024) / np.exp(-heron[:, 2]) / 1e12
heat_map("heron", heron)
tvm = np.loadtxt("tvm.txt")
tvm[:,2] = (2 * 1024 * 1024 * 1024) / tvm[:, 2] / 1e12
heat_map("tvm", tvm)

