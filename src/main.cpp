#include <cstdio>
#include <iostream>
#include <tensorflow/c/c_api.h>
//#include "tensorflow/core/public/session.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
//#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"
using namespace std;
using namespace tensorflow;

int main() {
    Status status;
    // Load GraphDef
    const string graphDir = "graph.meta";
    MetaGraphDef meta_graph_def;
    status = ReadBinaryProto(Env::Default(), graphDir, &meta_graph_def);
    auto graph_def = meta_graph_def.graph_def();
    
    // Convert GraphDef to Graph
    std::unique_ptr<Graph> new_graph(new Graph(OpRegistry::Global()));
    GraphConstructorOptions opts;
    status = ConvertGraphDefToGraph(opts, graph_def, new_graph.get());

    // Test Graph
    int num_node = new_graph->num_node_ids();
    for (int i = 0; i < num_node; i++) {
        auto p = new_graph->FindNodeId(i);
        cout << p->name() << endl;
    }
    cout << "Graph Test Passed." << endl;
    
    // Trst Devices


    return 0;
}
