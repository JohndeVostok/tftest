import os
import json
from config import *
from ast import literal_eval

test = "matmul"

if __name__ == "__main__":
    os.chdir(WORKPATH)
    with open(test + "_graph.json", "r") as f:
        data = f.read()
    gp = json.loads(data)

    nodes = {}
    for node in gp["nodes"]:
        n = node["name"]
        if not n in nodes:
            nodes[n] = {"in": [], "out": []}
        tmp = []
        for i in literal_eval(node["input"]):
            if i[0] == '^':
                tmp.append(i[1:])
            else:
                tmp.append(i)

        nodes[n]["in"] = tmp
        for i in tmp:
            if not i in nodes:
                odes[i] = {"in": [], "out": []}
            nodes[i]["out"].append(node["name"])

    nodemap = {}
    nodename = [str(len(nodes)) + "\n"]
    nid = 0
    for node in nodes:
        nodemap[node] = nid
        nodename.append(node + "\n")
        nid += 1

    graph = [str(len(nodes)) + "\n"]
    for node in nodes:
        tmp = ""
        for nodein in nodes[node]["in"]:
            tmp += str(nodemap[nodein]) + " "
        tmp += "\n"
        graph.append(tmp)
        tmp = ""
        for nodeout in nodes[node]["out"]:
            tmp += str(nodemap[nodeout]) + " "
        tmp += "\n"
        graph.append(tmp)

    with open(DATAPATH + test + "_nodename.txt", "w") as f:
        f.writelines(nodename)

    with open(DATAPATH + test + "_graph.txt", "w") as f:
        f.writelines(graph)

        
