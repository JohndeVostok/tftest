import json
from ast import literal_eval

test = "mnist"

if __name__ == "__main__":
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
            nodes[n] = {"in" : [], "out" : []}
        tmp = literal_eval(node["input"])
        nodes[n]["in"] = tmp
        for i in tmp:
            if not i in nodes:
                nodes[i] = {"in" : [], "out": []}
            nodes[i]["out"].append(node["name"])

    q = []
    for i in nodes:
        nodes[i]["t"] = 0
        if nodes[i]["in"] == []:
            q.append(i)
    while not q == []:
        i = q[0]
        del(q[0])
        if i in nodetime:
            nodes[i]["t"] = nodes[i]["t"] + nodetime[i]
        for j in nodes[i]["out"]:
            if nodes[i]["t"] > nodes[j]["t"]:
                nodes[j]["t"] = nodes[i]["t"]
            nodes[j]["in"].remove(i)
            if nodes[j]["in"] == []:
                q.append(j)
    for i in nodes:
        print(i, nodes[i]["t"])
