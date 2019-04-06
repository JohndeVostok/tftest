import os
import time
import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10
IMAGE_SIZE = 28
NUM_CHANNEL = 1
NUM_LABEL = 10
CONV1_DEEP = 32
CONV1_SIZE = 5
CONV2_DEEP = 64
CONV2_SIZE = 5
FC_SIZE = 512
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_TATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
TRAIN_STEP = 10


def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv'):
        w = tf.get_variable('w', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNEL, CONV1_DEEP],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('b', shape=[CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, w, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, b))
    with tf.variable_scope('layer2-pool'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.variable_scope('layer3-conv'):
        w = tf.get_variable('w', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('b', shape=[CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, w, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b))
    with tf.variable_scope('layer4-pool'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2,[-1,nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_w = tf.get_variable('w', shape=[nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        try:
            if regularizer != None:
                tf.add_to_collection('loss', regularizer(fc1_w))
        except:
            pass
        fc1_b = tf.get_variable('b', shape=[FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)
    with tf.variable_scope('layer6-fc2'):
        fc2_w = tf.get_variable('w', shape=[FC_SIZE, NUM_LABEL],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        try:
            if regularizer != None:
                tf.add_to_collection('loss', regularizer(fc2_w))
        except:
            pass
        fc2_b = tf.get_variable('b', shape=[NUM_LABEL], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_w) + fc2_b

    return logit


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL],
                       name='x-input')
    y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_TATE)
    y = inference(x, True, regularizer)
    global_step = tf.Variable(0, trainable=False)

    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_average_ops = variable_average.apply(tf.trainable_variables())

    cross_entroy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entroy_mean = tf.reduce_mean(cross_entroy)

    loss = cross_entroy_mean + tf.add_n(tf.get_collection('loss'))

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                               mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss, global_step=global_step)
    train_op = tf.group(train_step, variable_average_ops)
    saver = tf.train.Saver()

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        xs, ys = mnist.train.next_batch(BATCH_SIZE)
        reshape_xs = np.reshape(xs, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL))
        time_st = time.time_ns()
        _, loss_value, step, learn_rate = sess.run([train_op, loss, global_step, learning_rate],
                                                   feed_dict={x: reshape_xs, y_: ys},
                                                   options=run_options, run_metadata=run_metadata)
        time_ed = time.time_ns()
    with open("lenet5_runtime.json", "w") as f:
        f.write(str(time_ed - time_st))
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('lenet5_timeline.json', 'w') as f:
        f.write(ctf)
    with open('lenet5_graph.json', "w") as f:
        nodes = []
        for n in tf.get_default_graph().as_graph_def().node:
            nodes.append("{\"name\":\"" + str(n.name) + "\",\"input\":\"" + str(n.input) + "\"}")
        f.write("{\"nodes\":[\n")
        f.write(",".join(nodes))
        f.write("]}")
