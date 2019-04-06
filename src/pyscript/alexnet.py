import os
import time
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.framework import meta_graph
#from tensorflow.python.grappler import cost_analyzer

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


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    image_size = 224
    images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3]))
    output, parameters = inference(images)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    objective = tf.nn.l2_loss(output)
    grad = tf.gradients(objective, parameters)

    mg = meta_graph.create_meta_graph_def(graph=sess.graph)
#    report = cost_analyzer.GenerateCostReport(mg, per_node_report=True)
#    with open('lenet5_report.json', "w") as f:
#        f.write(str(report, encoding="utf-8"))

    with open('alexnet_graph.json', "w") as f:
        nodes = []
        for n in sess.graph_def.node:
            nodes.append("{\"name\":\"" + str(n.name) + "\",\"input\":\"" + str(n.input) + "\"}")
        f.write("{\"nodes\":[\n")
        f.write(",".join(nodes))
        f.write("]}")

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    time_st = time.time()
    sess.run(grad, options=run_options, run_metadata=run_metadata)
    time_ed = time.time()

    with open('alexnet_runtime.json', 'w') as f:
        f.write(str(time_ed - time_st))

    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('alexnet_timeline.json', 'w') as f:
        f.write(ctf)
