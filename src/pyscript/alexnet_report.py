import os
import time
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.framework import meta_graph
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.python.grappler import cluster
from tensorflow.python.grappler import cost_analyzer
from tensorflow.python.framework import ops as tf_ops

batch_size = 32
num_bathes = 100


def print_tensor_info(tensor):
    print("tensor name:", tensor.op.name, "-tensor shape:", tensor.get_shape().as_list())


def inference(images):
    parameters = []

    with tf.name_scope("conv1") as scope:
        kernel1 = tf.Variable(tf.truncated_normal([11, 11, 3, 64], mean=0, stddev=0.1,
                                                  dtype=tf.float32), name="weights")
        conv = tf.nn.conv2d(images, kernel1, [1, 4, 4, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0, shape=[64], dtype=tf.float32), trainable=True, name="biases")
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_tensor_info(conv1)
        parameters += [kernel1, biases]
        lrn1 = tf.nn.lrn(conv1, 4, bias=1, alpha=1e-3 / 9, beta=0.75, name="lrn1")
        pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool1")
        print_tensor_info(pool1)

    with tf.name_scope("conv2") as scope:
        kernel2 = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=0.1)
                              , name="weights")
        conv = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[192])
                             , trainable=True, name="biases")
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        print_tensor_info(conv2)
        parameters += [kernel2, biases]
        lrn2 = tf.nn.lrn(conv2, 4, 1.0, alpha=1e-3 / 9, beta=0.75, name="lrn2")
        pool2 = tf.nn.max_pool(lrn2, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID", name="pool2")
        print_tensor_info(pool2)

    with tf.name_scope("conv3") as scope:
        kernel3 = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32, stddev=0.1)
                              , name="weights")
        conv = tf.nn.conv2d(pool2, kernel3, strides=[1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name="biases")
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel3, biases]
        print_tensor_info(conv3)

    with tf.name_scope("conv4") as scope:
        kernel4 = tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.1, dtype=tf.float32),
                              name="weights")
        conv = tf.nn.conv2d(conv3, kernel4, strides=[1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256]), trainable=True, name="biases")
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel4, biases]
        print_tensor_info(conv4)

    with tf.name_scope("conv5") as scope:
        kernel5 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1, dtype=tf.float32),
                              name="weights")
        conv = tf.nn.conv2d(conv4, kernel5, strides=[1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256]), name="biases")
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias)
        parameters += [kernel5, bias]
        pool5 = tf.nn.max_pool(conv5, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID", name="pool5")
        print_tensor_info(pool5)

    pool5 = tf.reshape(pool5, (-1, 6 * 6 * 256))
    weight6 = tf.Variable(tf.truncated_normal([6 * 6 * 256, 4096], stddev=0.1, dtype=tf.float32),
                          name="weight6")
    ful_bias1 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[4096]), name="ful_bias1")
    ful_con1 = tf.nn.relu(tf.add(tf.matmul(pool5, weight6), ful_bias1))

    weight7 = tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.1, dtype=tf.float32),
                          name="weight7")
    ful_bias2 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[4096]), name="ful_bias2")
    ful_con2 = tf.nn.relu(tf.add(tf.matmul(ful_con1, weight7), ful_bias2))

    weight8 = tf.Variable(tf.truncated_normal([4096, 1000], stddev=0.1, dtype=tf.float32),
                          name="weight8")
    ful_bias3 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1000]), name="ful_bias3")
    ful_con3 = tf.nn.relu(tf.add(tf.matmul(ful_con2, weight8), ful_bias3))

    weight9 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1), dtype=tf.float32, name="weight9")
    bias9 = tf.Variable(tf.constant(0.0, shape=[10]), dtype=tf.float32, name="bias9")
    output_softmax = tf.nn.softmax(tf.matmul(ful_con3, weight9) + bias9)

    return output_softmax, parameters


def build_cluster():
    devices = []
    device_properties = device_properties_pb2.DeviceProperties(
        type='CPU',
        frequency=2000,
        num_cores=12,
        l1_cache_size=32768,
        l2_cache_size=262144,
        l3_cache_size=30720*1024)
    for i in range(2):
        devices.append(
            device_properties_pb2.NamedDevice(
                properties=device_properties, name='/CPU:' + str(i)))
    return cluster.Cluster(devices=devices)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    image_size = 224
    images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3]))
    output, parameters = inference(images)
    init = tf.global_variables_initializer()
    objective = tf.nn.l2_loss(output)
    grad = tf.gradients(objective, parameters)
    train_op = tf_ops.get_collection_ref(tf_ops.GraphKeys.TRAIN_OP)
    train_op.append(output)
    mg = meta_graph.create_meta_graph_def(graph=tf_ops.get_default_graph())
    cluster = build_cluster()
    report = cost_analyzer.GenerateCostReport(mg, per_node_report=True, cluster=cluster)
    with open('alexnet_report.json', "w") as f:
        f.write(str(report, encoding="utf-8"))
