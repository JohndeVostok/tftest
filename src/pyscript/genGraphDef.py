#coding=utf-8

import tensorflow as tf
from tensorflow.python.framework import meta_graph
from tensorflow.python.client import device_lib

if __name__ == "__main__":
    a = tf.random_normal([5000, 5000])
    b = tf.random_normal([5000, 5000])
    c = tf.matmul(a, b)
    d = tf.matmul(b, c)
    
    sess = tf.Session()
    mg = meta_graph.create_meta_graph_def(graph=tf.get_default_graph())
    with open("graph.txt", "w") as f:
        f.write(str(mg))
    with open("graph.meta", "wb") as f:
        f.write(mg.SerializeToString())
