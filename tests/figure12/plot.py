import json
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
case_name = 'G1_1'
FLOPs = float(2*1024*1024*1024)

def get_perfs(method):
    perfs = []
    max_perfs = []
    path='OUTS/%s/%s/records.txt'%(method, case_name)
    for row in open(path):
        json_dict = json.loads(row)
        perf = json_dict['perf']
        if perf == 0:
            perf = 0.0
        else:
            perf = FLOPs / (math.exp(-perf)) / 1e12
        perfs.append(perf)
        if max_perfs != []:
            max_perfs.append(max(perf, max_perfs[-1]))
        else:
            max_perfs.append(perf)
    return perfs[:1000], max_perfs[:1000]




plt.figure()
x = np.arange(1000)
all_maxs = []
for method in ["SA",  "GA", "CRAND", "CGA"]:
    perfs, maxs = get_perfs(method)
    plt.plot(x, maxs, label = method)
    all_maxs.append(maxs)
maxres = np.array(all_maxs).T
print(maxres.shape)
plt.legend()
plt.savefig('figure12b.png')
plt.close()

