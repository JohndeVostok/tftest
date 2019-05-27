import models
from config import *

test = "matmul"
device = "cpu"

if __name__ == "__main__":
    # nodename and nodemap
    with open(DATAPATH + test + "_nodename.txt") as f:
        lines = f.readlines()
    nodename = []
    nodemap = {}
    nid = 0
    n = int(lines[0])
    for line in lines[1:]:
        tmp = line.strip()
        nodename.append(tmp)
        nodemap[tmp] = nid
        nid += 1
    
    # graph
    with open(DATAPATH + test + "_graph.txt") as f:
        lines = f.readlines()
    nodein = [[] for i in range(n)]
    nodeout = [[] for i in range(n)]
    for i in range(n):
        tmp = lines[2 * i + 1].strip().split()
        for t in tmp:
            nodein[i].append(int(t))
        tmp = lines[2 * i + 2].strip().split()
        for t in tmp:
            nodeout[i].append(int(t))

    for i in range(n):
        tmp = []
        for j in nodein[i]:
            tmp.append(nodename[j])
        tmp = []
        for j in nodeout[i]:
            tmp.append(nodename[j])

    #parameter
    with open(DATAPATH + test + "_parameter.txt") as f:
        lines = f.readlines()
    parameter = [[] for i in range(n)]
    for line in lines:
        tmp = line.strip().split()
        nid = nodemap[tmp[0]]
        for t in tmp[1:]:
            parameter[nid].append(int(t))
   
    #predict
    core = 1
    queue = []
    running = [-1 for i in range(core)]
    runtime = [0 for i in range(core)]
    incnt = [len(nodein[i]) for i in range(n)]
    for i in range(n):
        if incnt[i] == 0:
            queue.append(i)
    time = 0
    while (len(queue) > 0):
        # add task
        for i in range(core):
            if running[i] == -1 and len(queue) > 0:
                node = queue[0]
                t = models.find(nodename[node])
                queue.pop(0)
                running[i] = node
                runtime[i] = models.get(t, parameter[node], device)
        # run task
        tmp = []
        for i in range(core):
            if running[i] != -1:
                tmp.append(runtime[i])
        t = min(tmp)
        time += t
        for i in range(core):
            if running[i] != -1:
                runtime[i] -= t
        for i in range(core):
            if running[i] != -1 and runtime[i] == 0:
                node = running[i]
                running[i] = -1
                for nout in nodeout[node]:
                    incnt[nout] -= 1
                    if incnt[nout] == 0:
                        queue.append(nout)
    print(time)

