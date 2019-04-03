#coding=utf-8

import tensorflow as tf
from tensorflow.python.framework import meta_graph
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.python.client import device_lib
from tensorflow.core.framework.device_attributes_pb2 import DeviceAttributes

if __name__ == "__main__":
    a = tf.random_normal([5000, 5000])
    b = tf.random_normal([5000, 5000])
    c = tf.matmul(a, b)
    d = tf.matmul(b, c)
    
    sess = tf.Session()
    mg = meta_graph.create_meta_graph_def(graph=tf.get_default_graph())
    with open("graph.meta", "wb") as f:
        f.write(mg.SerializeToString())

    devices = device_lib.list_local_devices()
    d = devices[0]
    print(d)
    with open("cpu_attr.meta", "wb") as f:
        f.write(d.SerializeToString())
"""
    device_properties = device_properties_pb2.DeviceProperties(
        type='CPU',
        frequency=1900,
        num_cores=2,
        l1_cache_size=32768,
        l2_cache_size=262144,
        l3_cache_size=3145728)
    cpu_device = device_properties_pb2.NamedDevice(
        properties=device_properties, name='/CPU:0')
    #print(cpu_device)
    #with open("cpu.meta", "wb") as f:
    #    f.write(cpu_device.SerializeToString())
"""
