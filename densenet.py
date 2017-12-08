"""Contains definitions for the Densely Connected Convolutional Networks.

Densely Connected Convolutional Networks (DenseNets) were originally proposed in:
[1] Huang, Gao; Liu, Zhuang; Weinberger, Kilian Q.; van der Maaten, Laurens
    Densely Connected Convolutional Networks. arXiv:1608.06993

DenseNet-121 for image classification into 1000 classes:

   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(densenet.densenet_arg_scope()):
      net, end_points = densenet.densenet_121(inputs, 1000, is_training=False)

DenseNet-121 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(densenet.densenet_arg_scope()):
      net, end_points = densenet.densenet_121(inputs,
                                                21,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=16)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import densenet_utils

slim = tf.contrib.slim
densenet_arg_scope = densenet_utils.densenet_arg_scope


@slim.add_arg_scope
def bottleneck(features, unit_id, growth_rate, depth_bottleneck, stride, theta, rate=1,
               outputs_collections=None, scope=None):
    """Bottleneck residual unit variant with BN before convolutions.

    This is the bottleneck architecture proposed in [1].
    Each layer only produces K (growth_rate) output feature maps. The bottleneck structure consists of Bach Normalization,
    ReLU, and a 1x1 Convolution followed by Bach Normalization, ReLU, and a 3x3 Convolution. The 1x1 Conv acts as a bottleneck
    operation reducing the number of input parameters (to that unit) to 4 * the growth rate. Then, we apply a 3x3 convolution to output
    k (growth_rate) feature maps. Note that each of the convolutions is precided by Bach Normalization and ReLU activations.

    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block to perform downsampling.

    Args:
      inputs: A tensor of size [batch, height, width, channels].
      unit_id: The id of the unit within the block
      growth_rate: The depth of the ResNet unit output.
      depth_bottleneck: The depth of the bottleneck layers - 4*growth_rate.
      stride: The Densenet unit's stride. Determines the amount of downsampling of
        the block output, only performed in the last block unit.
      rate: An integer, rate for atrous convolution.
      outputs_collections: Collection to add the Densenet unit output.
      scope: Optional variable_scope.

    Returns:
      The DenseNet unit's output.
    """
    with tf.variable_scope(scope, 'bottleneck', [features]) as sc:
        net = slim.batch_norm(features, activation_fn=tf.nn.relu)

        net = slim.conv2d(net, depth_bottleneck, [1, 1], scope="conv1x1", activation_fn=None, normalizer_fn=None)

        # net = slim.dropout(net, keep_prob, scope='dropout1')
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)

        net = slim.conv2d(net, growth_rate, [3, 3], scope="conv3x3", activation_fn=None, normalizer_fn=None, rate=rate)

        # Only concatenate features after the first unit
        if unit_id > 0:
            net = tf.concat((net, features), axis=3, name="feature_concat")

        # Add transition layer when the stride is greater than 1, that happens only in the last
        # unit of a block, if the desired output stride is not yet reached
        if stride > 1:
            net = densenet_utils.transition_layer(net, stride, theta, scope='transition_layer')

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, net)


def densenet(inputs,
             blocks,
             num_classes=None,
             is_training=True,
             global_pool=True,
             output_stride=None,
             initial_output_stride=4,
             include_root_block=True,
             spatial_squeeze=True,
             reuse=None,
             scope=None):
    """Generator for DenseNet models.

    This function generates a family of DenseNet models. See the DenseNet_*()
    methods for specific model instantiations, obtained by selecting different
    block instantiations that produce DenseNets of various structures.

    Training for image classification on Imagenet is usually done with [224, 224]
    inputs, resulting in [7, 7] feature maps at the output of the last DenseNet
    block for the DenseNets defined in [1] that have nominal stride equal to 32.
    However, for dense prediction tasks we advise that one uses inputs with
    spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
    this case the feature maps at the DenseNet output will have spatial shape
    [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
    and corners exactly aligned with the input image corners, which greatly
    facilitates alignment of the features to the image. Using as input [225, 225]
    images results in [8, 8] feature maps at the output of the last DenseNet block.

    For dense prediction tasks, the DenseNet needs to run in fully-convolutional
    (FCN) mode and global_pool needs to be set to False. The DenseNets in [1] all
    have nominal stride equal to 32 and a good choice in FCN mode is to use
    output_stride=16 in order to increase the density of the computed features at
    small computational and memory overhead.

    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      blocks: A list of length equal to the number of DenseNet blocks. Each element
        is a DenseNet_utils.Block object describing the units in the block.
      num_classes: Number of predicted classes for classification tasks.
        If 0 or None, we return the features before the logit layer.
      is_training: whether batch_norm layers are in training mode.
      global_pool: If True, we perform global average pooling before computing the
        logits. Set to True for image classification, False for dense prediction.
      output_stride: If None, then the output will be computed at the nominal
        network stride. If output_stride is not None, it specifies the requested
        ratio of input to output spatial resolution.
      initial_output_stride: If true include the max pooling op after the convolution
      include_root_block: If True, include the initial convolution followed by
        max-pooling, if False excludes it. If excluded, `inputs` should be the
        results of an activation-less convolution.
      spatial_squeeze: if True, logits is of shape [B, C], if false logits is
          of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
          To use this parameter, the input images must be smaller than 300x300
          pixels, in which case the output logit layer does not contain spatial
          information and can be removed.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.


    Returns:
      net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
        If global_pool is False, then height_out and width_out are reduced by a
        factor of output_stride compared to the respective height_in and width_in,
        else both height_out and width_out equal one. If num_classes is 0 or None,
        then net is the output of the last DenseNet block, potentially after global
        average pooling. If num_classes is a non-zero integer, net contains the
        pre-softmax activations.
      end_points: A dictionary from components of the network to the corresponding
        activation.

    Raises:
      ValueError: If the target output_stride is not valid.
    """

    if initial_output_stride not in [2,4]:
        raise ValueError("Initial output stride must be a value of 2 or 4")

    with tf.variable_scope(scope, 'densenet', [inputs], reuse=reuse) as sc:

        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck,
                             densenet_utils.stack_blocks_dense],
                            outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = inputs
                output_stride_factor = initial_output_stride
                if include_root_block:

                    if output_stride is not None:
                        if output_stride % output_stride_factor != 0:
                            raise ValueError('The output_stride needs to be a multiple of %d.' % output_stride_factor)
                        output_stride /= output_stride_factor

                    # We do not include batch normalization or activation functions in
                    # conv1 because the first DenseNet unit will perform these.
                    with slim.arg_scope([slim.conv2d],
                                        activation_fn=None, normalizer_fn=None):
                        net = slim.conv2d(net, 64, [7, 7], stride=2, scope='conv1')

                    if initial_output_stride == 4:
                        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')

                net = densenet_utils.stack_blocks_dense(net, blocks, output_stride)
                # This is needed because the pre-activation variant does not have batch
                # normalization or activation functions in the residual unit output. See
                # Appendix of [2].
                net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                if global_pool:
                    # Global average pooling.
                    net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                    end_points['global_pool'] = net
                if num_classes is not None:
                    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                      normalizer_fn=None, scope='logits')
                    end_points[sc.name + '/logits'] = net
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                        end_points[sc.name + '/spatial_squeeze'] = net
                    end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points


densenet.default_image_size = 224


def densenet_block(scope, growth_rate, num_units, stride, theta):
    """Helper function for creating a DenseNet bottleneck block.

    Args:
      scope: The scope of the block.
      growth_rate: The depth that each of the DenseNet units output.
      num_units: The number of units in the block.
      stride: The stride of the block, implemented as a stride in the last unit.
        All other units have stride=1.

    Returns:
      A DenseNet bottleneck block.
    """
    return densenet_utils.Block(scope, bottleneck, [{
        'growth_rate': growth_rate,
        'depth_bottleneck': 4 * growth_rate,  # each 1×1 convolution produce 4k feature-maps
        'stride': 1,
        'theta': theta
    }] * (num_units - 1) + [{
        'growth_rate': growth_rate,
        'depth_bottleneck': 4 * growth_rate,
        'stride': stride,
        'theta': theta
    }])


densenet.default_image_size = 224


def densenet_121(inputs,
                 num_classes=None,
                 theta=0.5,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 initial_output_stride=4,
                 reuse=None,
                 scope='DenseNet_121'):
    """DenseNet-121 model of [1]. See densenet() for arg and return description."""
    blocks = [
        densenet_block('block1', growth_rate=32, num_units=6, stride=2, theta=theta),
        densenet_block('block2', growth_rate=32, num_units=12, stride=2, theta=theta),
        densenet_block('block3', growth_rate=32, num_units=24, stride=2, theta=theta),
        densenet_block('block4', growth_rate=32, num_units=16, stride=1, theta=theta),
    ]
    return densenet(inputs, blocks, num_classes, is_training=is_training,
                    global_pool=global_pool, output_stride=output_stride,
                    include_root_block=True, initial_output_stride=initial_output_stride,
                    spatial_squeeze=spatial_squeeze, reuse=reuse, scope=scope)


densenet_121.default_image_size = densenet.default_image_size


def densenet_169(inputs,
                 num_classes=None,
                 theta=0.5,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='DenseNet_169'):
    """DenseNet-101 model of [1]. See DenseNet() for arg and return description."""
    blocks = [
        densenet_block('block1', growth_rate=64, num_units=6, stride=2, theta=theta),
        densenet_block('block2', growth_rate=128, num_units=12, stride=2, theta=theta),
        densenet_block('block3', growth_rate=256, num_units=32, stride=2, theta=theta),
        densenet_block('block4', growth_rate=512, num_units=32, stride=1, theta=theta),
    ]
    return densenet(inputs, blocks, num_classes, is_training=is_training,
                    global_pool=global_pool, output_stride=output_stride,
                    include_root_block=True, spatial_squeeze=spatial_squeeze,
                    reuse=reuse, scope=scope)


densenet_169.default_image_size = densenet.default_image_size


def densenet_201(inputs,
                 num_classes=None,
                 theta=0.5,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='DenseNet_201'):
    """DenseNet-152 model of [1]. See DenseNet() for arg and return description."""
    blocks = [
        densenet_block('block1', growth_rate=64, num_units=6, stride=2, theta=theta),
        densenet_block('block2', growth_rate=128, num_units=12, stride=2, theta=theta),
        densenet_block('block3', growth_rate=256, num_units=48, stride=2, theta=theta),
        densenet_block('block4', growth_rate=512, num_units=32, stride=1, theta=theta),
    ]
    return densenet(inputs, blocks, num_classes, is_training=is_training,
                    global_pool=global_pool, output_stride=output_stride,
                    include_root_block=True, spatial_squeeze=spatial_squeeze,
                    reuse=reuse, scope=scope)


densenet_201.default_image_size = densenet.default_image_size


def densenet_161(inputs,
                 num_classes=None,
                 theta=0.5,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='DenseNet_161'):
    """DenseNet-200 model of [2]. See DenseNet() for arg and return description."""
    blocks = [
        densenet_block('block1', growth_rate=48, num_units=6, stride=2, theta=theta),
        densenet_block('block2', growth_rate=48, num_units=12, stride=2, theta=theta),
        densenet_block('block3', growth_rate=48, num_units=36, stride=2, theta=theta),
        densenet_block('block4', growth_rate=48, num_units=24, stride=1, theta=theta),
    ]
    return densenet(inputs, blocks, num_classes, is_training=is_training,
                    global_pool=global_pool, output_stride=output_stride,
                    include_root_block=True, spatial_squeeze=spatial_squeeze,
                    reuse=reuse, scope=scope)


densenet_161.default_image_size = densenet.default_image_size
