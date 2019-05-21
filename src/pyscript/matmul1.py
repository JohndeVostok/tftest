import os
import json
import tensorflow as tf
from tensorflow.python.client import timeline

if __name__ == "__main__":
#    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    a = tf.random_normal([5000, 5000])
    b = tf.random_normal([5000, 5000])
    c = tf.random_normal([5000, 5000])
    d = tf.matmul(a, b)
    e = tf.matmul(b, c)
    f = tf.matmul(d, e)

    sess = tf.Session()

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    writer=tf.summary.FileWriter("logs", sess.graph)

    sess.run(f, options=run_options, run_metadata=run_metadata)
    writer.add_run_metadata(run_metadata, 'step %03d' % 0)

    writer.close()

    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('matmul1_timeline.json', 'w') as f:
        f.write(ctf)

    with open('matmul1_graph.json', "w") as f:
        nodes = []
        for n in tf.get_default_graph().as_graph_def().node:
            nodes.append("{\"name\":\"" + str(n.name) + "\",\"input\":\"" + str(n.input) + "\"}")
        f.write("{\"nodes\":[\n")
        f.write(",".join(nodes))
        f.write("]}")

