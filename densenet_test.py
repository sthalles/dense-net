import tensorflow as tf
import numpy as np
from densenet import densenet_121
from densenet_utils import densenet_arg_scope
import os
slim = tf.contrib.slim

os.environ["CUDA_VISIBLE_DEVICES"]="1"
log_folder="./tboard_logs"

fake_input = np.zeros((1,65,65,3), dtype=np.float32)


with slim.arg_scope(densenet_arg_scope()):

    net, end_points = densenet_121(fake_input,
                                   21,
                                   is_training=True,
                                   global_pool=False,
                                   output_stride=16,
                                   include_root_max_poolling=False,
                                   spatial_squeeze=False)

#with slim.arg_scope(densenet_arg_scope()):
#    net, end_points = densenet_121(fake_input, 10, is_training=True)

    print(end_points)
    #tf.summary.image("output", net[:,:,:,:1], 1)
    #merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(log_folder + "/train", sess.graph)
    sess.run(tf.global_variables_initializer())

    logits = sess.run(net)

    #train_writer.add_summary(summary_string)
    print(logits.shape)
