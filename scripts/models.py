import math
from config import *


RANDOMNORMAL = 0
MATMUL = 1
modelname = {"random_normal" : RANDOMNORMAL, "matmul" : MATMUL}


def random_normal(shape, device):
    size = 1
    for i in shape:
        size *= i
    return 0


def matmul(shape, device):
    size = 1
    for i in shape:
        size *= i
    with open(DATAPATH + "matmul_" + device + ".txt") as f:
        lines = f.readlines()
    x = []
    y = []
    for line in lines:
        tmp = line.strip().split()
        x.append(float(tmp[0]))
        y.append(math.log(float(tmp[1])))
    t = math.log(size, 2) / 3
    l = len(x)
    p = 0
    for i in range(l - 1):
        if t >= x[i] and t < x[i + 1]:
            p = (t - x[i]) / (x[i + 1] - x[i]) * (y[i + 1] - y[i]) + y[i]
    if t > x[l - 1]:
        p = t / x[l - 1] * y[l - 1]
    res = math.exp(p)
    return res


def find(nodename):
    if nodename.find("/") != -1:
        return -1
    op = nodename[:nodename.rfind("_")]
    if op in modelname:
        return modelname[op]
    else:
        return -1


def get(nid, parameter, device):
    if nid == -1:
        return 0
    elif nid == 0:
        return random_normal(parameter, device)
    elif nid == 1:
        return matmul(parameter, device)
    else:
        return 0

