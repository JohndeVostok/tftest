import os
import json
import numpy as np
from config import *

if __name__ == "__main__":
    os.chdir(WORKPATH)
    time = []
    for i in range(MATRANGE):
        time.append([])
        for k in range(MATRANGE):
            time[i].append([])
            for j in range(MATRANGE):
                time[i][k].append([])
    for e in range(EPOCH):
        for i in range(MATRANGE):
            for k in range(MATRANGE):
                for j in range(MATRANGE):
                    with open(MATMULCPUPREFIX + "_" + str(i) + "_" + str(k) + "_" + str(j) + "_" + str(e) + ".json", "r") as f:
                        data = f.read()
                    tl = json.loads(data)
                    for node in tl["traceEvents"]:
                        if node["name"] == "MatMul" and "args" in node:
                            time[i][k][j].append(int(node["dur"]))
    
    mean = np.zeros([MATRANGE, MATRANGE, MATRANGE])
    midd = np.zeros([MATRANGE, MATRANGE, MATRANGE])
    for i in range(MATRANGE):
        for k in range(MATRANGE):
            for j in range(MATRANGE):
                tmp = sorted(time[i][k][j])
                op = int(EPOCH / 4)
                ed = int(EPOCH * 3 / 4)
                s = 0
                c = 0
                for t in tmp[op: ed]:
                    s += t
                    c += 1
                avg = s / c
                if EPOCH % 2 == 1:
                    mid = tmp[int(EPOCH / 2) + 1]
                else:
                    mid = (tmp[int(EPOCH / 2)] + tmp[int(EPOCH / 2) + 1]) / 2
                mean[i][k][j] = avg
                midd[i][k][j] = mid

    
    res = []
    for i in range(MATRANGE):
        for k in range(MATRANGE):
            for j in range(MATRANGE):
                res.append(str(i) + " " + str(k) + " " + str(j) + " " + str(mean[i, k, j]) + "\n")
    with open(MATMULCPURES, "w") as f:
        f.writelines(res)

