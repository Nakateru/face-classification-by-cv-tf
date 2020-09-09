"""
tensorboard --logdir="./log/"
"""

import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2
from resnet import resnet34

logs_dir = './log/'

inputs = tf.ones((20, 32, 32, 3))
model = resnet34()
model.build(input_shape=(None, 32, 32, 3))

graph_writer = tf.summary.create_file_writer(logdir=logs_dir)
with graph_writer.as_default():
    graph = model.call.get_concrete_function(inputs).graph
    summary_ops_v2.graph(graph.as_graph_def())
graph_writer.close()
