import os
import json
import tensorflow as tf
from tensorflow.python.client import timeline
import time
from config import *

if __name__ == "__main__":
    os.chdir(WORKPATH)
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    mat = []
    res = []
    for i in range(MATRANGE):
        mat.append(tf.random_normal([2**i, 2**i]))
    for i in range(MATRANGE):
        res.append(tf.matmul(mat[i], mat[i]))
    sess = tf.Session()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    for e in range(EPOCH):
        print("EPOCH", e)
        for i in range(MATRANGE):
            sess.run(res[i], options=run_options, run_metadata=run_metadata)
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open(MATMULCPUPREFIX + '_' + str(i) + '_' + str(e) + '.json', 'w') as f:
                f.write(ctf)

