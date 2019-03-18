import os
import json
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
import tensorflow as tf
from tensorflow.python.client import timeline

x = tf.random_normal([5000, 10000])
y = tf.random_normal([10000, 5000])
res = tf.matmul(x, y)

sess = tf.Session()

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

writer=tf.summary.FileWriter("logs", sess.graph) 

sess.run(res, options=run_options, run_metadata=run_metadata)
writer.add_run_metadata(run_metadata, 'step %03d' % 0)

writer.close()

tl = timeline.Timeline(run_metadata.step_stats)
ctf = tl.generate_chrome_trace_format()
with open('matmul_timeline.json', 'w') as f:
    f.write(ctf)

with open('matmul_graph.json', "w") as f:
    nodes = []
    for n in tf.get_default_graph().as_graph_def().node:
        nodes.append("{\"name\":\"" + str(n.name) + "\",\"input\":\"" + str(n.input) + "\"}")
    f.write("{\"nodes\":[\n")
    f.write(",".join(nodes))
    f.write("]}")

