import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, w) + b)
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

writer=tf.summary.FileWriter("logs", sess.graph) 

for i in range(1, 1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	if i % 100 == 0:
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}, options=run_options, run_metadata=run_metadata)
		writer.add_run_metadata(run_metadata, 'step %03d' % i)
	else:
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

writer.close()
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

tl = timeline.Timeline(run_metadata.step_stats)
ctf = tl.generate_chrome_trace_format()
with open('mnist_timeline.json', 'w') as f:
    f.write(ctf)
with open('mnist_graph.json', "w") as f:
	for n in tf.get_default_graph().as_graph_def().node:
		f.write(n.name + "\n")	

