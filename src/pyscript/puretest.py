import os
import json
import tensorflow as tf
from tensorflow.python.client import timeline
import time

if __name__ == "__main__":
#    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    a = tf.random_normal([32, 224, 224, 3])
    b = tf.truncated_normal([11, 11, 3, 64], mean=0, stddev=0.1, dtype=tf.float32)
    c = tf.nn.conv2d(a, b, [1, 4, 4, 1], padding="SAME")
    d = tf.nn.max_pool(c, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool1")
#    d = tf.random_normal([32, 27, 27, 64])
    e = tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,     stddev=0.1)
    f = tf.nn.conv2d(d, e, [1, 1, 1, 1], padding="SAME")
    
    
    sess = tf.Session()

    init_op = tf.initialize_local_variables()
    sess.run(init_op)

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    writer=tf.summary.FileWriter("logs", sess.graph)

    op = time.time()
    sess.run(f, options=run_options, run_metadata=run_metadata)
    ed = time.time()
    print(ed-op)
    writer.add_run_metadata(run_metadata, 'step %03d' % 0)

    writer.close()

    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('test_timeline.json', 'w') as f:
        f.write(ctf)

