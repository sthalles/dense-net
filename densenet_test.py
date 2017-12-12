import os
import tensorflow as tf
import numpy as np
import densenet
from densenet_utils import densenet_arg_scope

slim = tf.contrib.slim

os.environ["CUDA_VISIBLE_DEVICES"]="0"
log_folder="./tboard_logs"

fake_input = np.zeros((1,224,224,3), dtype=np.float32)


with slim.arg_scope(densenet_arg_scope()):

    net, end_points = densenet.densenet_X(fake_input,
                                            21,
                                            theta=0.5,
                                            is_training=True,
                                            global_pool=False,
                                            output_stride=16,
                                            initial_output_stride=2,
                                            spatial_squeeze=False)


saver = tf.train.Saver()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(log_folder + "/train", sess.graph)
    sess.run(tf.global_variables_initializer())
    logits = sess.run(net)
    print(logits.shape) # (1, 14, 14, 21)
