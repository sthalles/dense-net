import os
import tensorflow as tf
import numpy as np
import densenet
from densenet_utils import densenet_arg_scope

slim = tf.contrib.slim

os.environ["CUDA_VISIBLE_DEVICES"]="1"
log_folder="./tboard_logs"

fake_input = np.zeros((1,224,224,3), dtype=np.float32)


with slim.arg_scope(densenet_arg_scope()):

    net, end_points = densenet.densenet_121(fake_input,
                                            21,
                                            theta=0.5,
                                            is_training=True,
                                            global_pool=False,
                                            output_stride=16,
                                            initial_output_stride=2,
                                            spatial_squeeze=False)

# with slim.arg_scope(densenet_arg_scope()):
#     net, end_points = densenet.densenet_121(inputs=fake_input,
#                                             num_classes=1000,
#                                             theta=0.5,
#                                             is_training=True,
#                                             scope='DenseNet_121')
    print(end_points)
    print(net)
    #tf.summary.image("output", net[:,:,:,:1], 1)
    #merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(log_folder + "/train", sess.graph)
    sess.run(tf.global_variables_initializer())

    logits = sess.run(net)

    #train_writer.add_summary(summary_string)
    print(logits.shape)
