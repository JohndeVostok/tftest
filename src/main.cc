#include <cstdio>
#include <iostream>
//#include <tensorflow/c/c_api.h>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/placer.h"

//#include "tensorflow/core/framework/device_attributes.pb_text.h"
//#include "tensorflow/core/framework/device_attributes.pb.h"
//#include "tensorflow/core/common_runtime/device_set.h"
using namespace std;
using namespace tensorflow;

int main() {
    Status status;
    
    // Test Graph
    printf("Test Graph.\n");
    
    // Load GraphDef
    const string graphDir = "graph.meta";
    MetaGraphDef meta_graph_def;
    status = ReadBinaryProto(Env::Default(), graphDir, &meta_graph_def);
    auto graph_def = meta_graph_def.graph_def();
    
    // Convert GraphDef to Graph
    unique_ptr <Graph> graph(new Graph(OpRegistry::Global()));
    GraphConstructorOptions opts;
    status = ConvertGraphDefToGraph(opts, graph_def, graph.get());

    for (auto node : graph->op_nodes()) {
        cout << node->name() << " '" << node->requested_device() << "'" << node->assigned_device_name_index() << endl;
    }
    
    // Test Devices
    printf("Test Devices.\n");
    unique_ptr<Device> device(DeviceFactory::NewDevice("CPU", {}, "/job:localhost/replica:0/task:0"));
    DeviceSet device_set;
    device_set.AddDevice(device.get());
    for (auto i : device_set.devices()) {
        cout << i->name() << endl;
        graph->InternDeviceName(i->name());
    }

    // Test Placer
    // TODO: placer bug.
    cout << "TEST::Placer" << endl;
    SessionOptions *session_options = nullptr;
    cout << "graph.ptr" << graph.get() << endl;
    Placer placer(graph.get(), &device_set);
    status = placer.Run(); 
    //auto node = graph->FindNodeId(0);
    //for (const auto node : graph->op_nodes()) {
    //    cout << "node.name:" << node->name() << " node.ptr:" << node << endl;
    //    cout << "node.device_idx:" << node->assigned_device_name_index() << endl;
        //" '" << node->requested_device() << "'" << node->assigned_device_name_index() << " " << endl;
        //cout << node->assigned_device_name() << endl;
        //cout << node->name() << ":requeseted:'" << node->requested_device() << "' assigned: '" << node->assigned_device_name() << "'";
    }
    cout << "Placer run correctly. but var offset invalid."
    cout << "TEST::Placer End" << endl;
    //SaveStatefulNodes(new_graph.get());

    // Test Optimizor
    //GraphOptimizationPassOptions optimization_options;
    //optimization_options.session_options = session_options;
    //optimization_options.graph = &;
    //optimization_options.flib_def = flib.get();
    //optimization_options.device_set = device_set_;

    return 0;
}
