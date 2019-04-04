import numpy as np
import tensorflow as tf
import sys, os

if __name__ == '__main__':
    train_dir = os.path.join('demo_model/', "demo")
    a = tf.random_normal([5000, 5000])
    b = tf.random_normal([5000, 5000])
    c = tf.Variable(tf.matmul(a, b))
    res = tf.matmul(b, c, name='res')
    with tf.Session() as sess:
        feed_dict = dict()
        fetch_list = [res]
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # 训练和保存模型
        res = sess.run(feed_dict=feed_dict, fetches=fetch_list)
        saver.save(sess, train_dir)
