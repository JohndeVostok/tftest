import os
import json
import tensorflow as tf
from tensorflow.python.client import timeline
import time
from config import *

if __name__ == "__main__":
    os.chdir(WORKPATH)
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    mat = [[] for i in range(MATRANGE)]
    res = [[[] for j in range(MATRANGE)] for i in range(MATRANGE)]
    for i in range(MATRANGE):
        for j in range(MATRANGE):
            mat[i].append(tf.random_normal([2**i, 2**j]))
    for i in range(MATRANGE):
        for k in range(MATRANGE):
            for j in range(MATRANGE):
                res[i][k].append(tf.matmul(mat[i][k], mat[k][j]))
    sess = tf.Session()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    for e in range(EPOCH):
        print("EPOCH", e)
        for i in range(MATRANGE):
            for k in range(MATRANGE):
                for j in range(MATRANGE):
                    sess.run(res[i][k][j], options=run_options, run_metadata=run_metadata)
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open(MATMULCPUPREFIX + '_' + str(i) + '_' + str(k) + '_' + str(j) + '_' + str(e) + '.json', 'w') as f:
                        f.write(ctf)

