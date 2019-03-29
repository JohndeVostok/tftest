from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.grappler import cluster
from tensorflow.python.grappler import graph_placer
from tensorflow.python.grappler import cost_analyzer
from tensorflow.python.ops import math_ops

import tensorflow as tf

def buildCluster(num_cpus=1, num_gpus=2):
    devices = []
    if num_gpus > 0:
        device_properties = device_properties_pb2.DeviceProperties(
            type='GPU',
            vendor='NVidia',
            model='GeForce GTX TITAN X',
            frequency=1076,
            num_cores=24,
            environment={'architecture': '5.2',
                         'cuda': '8000',
                         'cudnn': '6021'},
            num_registers=65536,
            l1_cache_size=24576,
            l2_cache_size=3145728,
            shared_memory_size_per_multiprocessor=98304,
            memory_size=12783648768,
            bandwidth=336480000)
        for i in range(num_gpus):
            devices.append(
                device_properties_pb2.NamedDevice(
                    properties=device_properties, name='/GPU:' + str(i)))

    assert num_cpus > 0
    device_properties = device_properties_pb2.DeviceProperties(
        type='CPU',
        frequency=1900,
        num_cores=2,
        l1_cache_size=32768,
        l2_cache_size=262144,
        l3_cache_size=3145728)
    for i in range(num_cpus):
        devices.append(
            device_properties_pb2.NamedDevice(
                properties=device_properties, name='/CPU:' + str(i)))

    return cluster.Cluster(devices=devices)


if __name__ == "__main__":
    """Place a trivial graph."""
    a = tf.random_normal([5000, 5000])
    b = tf.random_normal([5000, 5000])
    c = tf.matmul(a, b)
    d = tf.matmul(b, c)
    train_op = tf_ops.get_collection_ref(tf_ops.GraphKeys.TRAIN_OP)
    train_op.append(d)
    mg = meta_graph.create_meta_graph_def(graph=tf_ops.get_default_graph())
    with open("placer_test_before.txt", "w") as f:
        f.write(str(mg))
    #
    gcluster = buildCluster(num_cpus=1, num_gpus=0)

    rep = cost_analyzer.GenerateCostReport(mg, per_node_report=True, cluster=gcluster)
    with open("report.txt", "w") as f:
        f.write(str(rep, encoding="utf-8"))
    print(rep)

    # placed_mg = graph_placer.PlaceGraph(mg, verbose=True, allotted_time=15, cluster=gcluster)
    # with open("placer_test_after.txt", "w") as f:
    #     f.write(str(placed_mg))
    #
    # available_devices = [device.name for device in gcluster.ListDevices()]
    # print(available_devices)
    # for node in placed_mg.graph_def.node:
    #     print(node.name, node.device)
    #
    # self.assertEqual(4, len(placed_mg.graph_def.node))
    # self.assertItemsEqual([node.name for node in placed_mg.graph_def.node],
    #                       [node.name for node in mg.graph_def.node])
    #
    # available_devices = [device.name for device in gcluster.ListDevices()]
    # for node in placed_mg.graph_def.node:
    #     # The constant nodes are optimized away before the placer is run, and
    #     # therefore won't be placed.
    #     self.assertTrue(not node.device or node.device in available_devices)