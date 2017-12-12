# Densely Connected Convolutional Networks in Tensorflow

Blog post:

## Usage

The DenseNet library described here implements all 4 architectures used to train on ImageNet plus a custom constructor in which any network variation can be experimented.

### Classification

That follows a basic usage for classification task with 1000 classes.

{% highlight python %}
import tensorflow as tf
import numpy as np
import densenet
from densenet_utils import densenet_arg_scope
slim = tf.contrib.slim

fake_input = np.zeros((1,224,224,3), dtype=np.float32)

with slim.arg_scope(densenet_arg_scope()):
    net, end_points = densenet.densenet_121(inputs=fake_input,
                                            num_classes=1000,
                                            theta=0.5,
                                            is_training=True,
                                            scope='DenseNet_121')
    print(net.shape)# (1, 1000)
{% endhighlight %}

### Dense Prediction tasks

Basic usage for dense prediction problems. output_stride is set to 16 and initial_output_stride controls how much signal decimation is going to be performed in the beginning of the network.

{% highlight python %}
with slim.arg_scope(densenet_arg_scope()):

    net, end_points = densenet.densenet_121(fake_input,
                                   num_classes=21,
                                   theta=0.5,
                                   is_training=True,
                                   global_pool=False,
                                   output_stride=16,
                                   initial_output_stride=2,
                                   spatial_squeeze=False)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      logits = sess.run(net)
      print(logits.shape) # (1, 14, 14, 21)
{% endhighlight %}

For general using and experimenting with different configurations of DenseNet, use the ***densenet_X(...)** constructor.

{% highlight python %}
# Custom definition for DenseNet_121
def densenet_X(inputs,
                num_classes=None,
                theta=0.5,
                num_units_per_block=[6,12,24,16], # number of blocks equal to 4
                growth_rate=32,
                is_training=True,
                global_pool=True,
                output_stride=None,
                spatial_squeeze=True,
                initial_output_stride=4,
                reuse=None,
                scope='DenseNet_X'):
{% endhighlight %}