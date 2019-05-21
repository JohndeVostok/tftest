import os
import json
from ast import literal_eval

test = "matmul"

if __name__ == "__main__":
#    os.chdir("d:/git/tftest")
    nodetime = {}
    with open(test + "_timeline.json", "r") as f:
        data = f.read()
    tl = json.loads(data)
    for event in tl["traceEvents"]:
        #print(event["args"])
        if "dur" in event:
            nodetime[event["args"]["name"]] = event["dur"]

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
                nodes[i] = {"in": [], "out": []}
            nodes[i]["out"].append(node["name"])

    with open("log", "w") as f:
        for node in nodes:
            f.write(str(node) + "\n")

    q = []
    for i in nodes:
        nodes[i]["t"] = 0
        if nodes[i]["in"] == []:
            q.append(i)
    res = []
    while not q == []:
        i = q[0]
        del(q[0])
        if i in nodetime:
            res.append(str(i))
            res.append(str(nodes[i]["t"]))
            nodes[i]["t"] = nodes[i]["t"] + nodetime[i]
            res.append(str(nodes[i]["t"]))
        for j in nodes[i]["out"]:
            if nodes[i]["t"] > nodes[j]["t"]:
                nodes[j]["t"] = nodes[i]["t"]
            nodes[j]["in"].remove(i)
            if nodes[j]["in"] == []:
                q.append(j)
    res = [i + "\n" for i in res]
    with open(test + "_res.txt", "w") as f:
        f.writelines(res)
    #for i in nodes:
    #    print(i, nodes[i]["t"])
