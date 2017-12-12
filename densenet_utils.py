"""Contains building blocks for various versions of DenseNets.

Densely Connected Convolutional Networks (DenseNets) were originally proposed in:
[1] Huang, Gao; Liu, Zhuang; Weinberger, Kilian Q.; van der Maaten, Laurens
    Densely Connected Convolutional Networks. arXiv:1608.06993

We can obtain different DenseNets variants by changing the network depth, width,
and also the form of residual unit. This module implements the infrastructure for
building them. Concrete DenseNet units and full DenseNet networks are implemented in
the accompanying densenet.py module.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

slim = tf.contrib.slim


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a DenseNet block.

    Its parts are:
      scope: The scope of the `Block`.
      unit_fn: The DenseNet unit function which takes as input a `Tensor` and
        returns another `Tensor` with the output of the DenseNet unit.
      args: A list of length equal to the number of units in the `Block`. The list
        contains one (growth_rate, depth_bottleneck, stride, theta) tuple for each unit in the
        block to serve as argument to unit_fn.
    """


def transition_layer(inputs, factor, theta, scope=None):
    """Subsamples the input along the spatial dimensions.

    The transition layers described in the paper consist of a batch normalization
    layer and an 1×1 convolutional layer followed by a 2×2 average pooling layer 
    between two contiguous dense blocks.

    Args:
    inputs: A `Tensor` of size [batch, height_in, width_in, channels].
    factor: The subsampling factor.
    scope: Optional variable_scope.

    Returns:
    output: A `Tensor` of size [batch, height_out, width_out, channels] with the
      input, either intact (if factor == 1) or subsampled (if factor > 1).
    """
    with tf.variable_scope(scope):
        if factor == 1:
            return inputs
        else:
            current_depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
            net = slim.batch_norm(inputs, activation_fn=tf.nn.relu)
            net = slim.conv2d(net, theta * current_depth, [1, 1], scope='conv1x1',
                              activation_fn=None, normalizer_fn=None)

            net = slim.avg_pool2d(net, [2, 2], scope='avg_pool', stride=factor)
            return net


@slim.add_arg_scope
def stack_blocks_dense(net, blocks, output_stride=None,
                       outputs_collections=None):
    """Stacks DenseNet `Blocks` and controls output feature density.

    First, this function creates scopes for the DenseNet in the form of
    'block_name/unit_1', 'block_name/unit_2', etc.

    Second, this function allows the user to explicitly control the DenseNet
    output_stride, which is the ratio of the input to output spatial resolution.
    This is useful for dense prediction tasks such as semantic segmentation or
    object detection.

    Most DenseNets consist of 4 DenseNet blocks and subsample the activations by a
    factor of 2 when transitioning between consecutive DenseNet blocks. This results
    to a nominal DenseNet output_stride equal to 8. If we set the output_stride to
    half the nominal network stride (e.g., output_stride=4), then we compute
    responses twice.

    Control of the output feature density is implemented by atrous convolution.

    Args:
      net: A `Tensor` of size [batch, height, width, channels].
      blocks: A list of length equal to the number of DenseNet `Blocks`. Each
        element is a DenseNet `Block` object describing the units in the `Block`.
      output_stride: If `None`, then the output will be computed at the nominal
        network stride. If output_stride is not `None`, it specifies the requested
        ratio of input to output spatial resolution, which needs to be equal to
        the product of unit strides from the start up to some level of the DenseNet.
        For example, if the DenseNet employs units with strides 1, 2, 1, 3, 4, 1,
        then valid values for the output_stride are 1, 2, 6, 24 or None (which
        is equivalent to output_stride=24).
      outputs_collections: Collection to add the DenseNet block outputs.

    Returns:
      net: Output tensor with stride equal to the specified output_stride.

    Raises:
      ValueError: If the target output_stride is not valid.
    """
    # The current_stride variable keeps track of the effective stride of the
    # activations. This allows us to invoke atrous convolution whenever applying
    # the next residual unit would result in the activations having stride larger
    # than the target output_stride.
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1

    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError('The target output_stride cannot be reached.')

                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net, i, rate=rate, **dict(unit, stride=1))
                        rate *= unit.get('stride', 1)
                    else:
                        net = block.unit_fn(net, i, rate=1, **unit)
                        current_stride *= unit.get('stride', 1)

            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')

    return net


def densenet_arg_scope(weight_decay=0.0001,
                       batch_norm_decay=0.997,
                       batch_norm_epsilon=1e-5,
                       batch_norm_scale=True,
                       activation_fn=tf.nn.relu,
                       use_batch_norm=True):
    """Defines the default DenseNet arg scope.

    TODO(gpapan): The batch-normalization related default values above are
      appropriate for use in conjunction with the reference DenseNet models
      released at https://github.com/KaimingHe/deep-residual-networks. When
      training DenseNets from scratch, they might need to be tuned.

    Args:
      weight_decay: The weight decay to use for regularizing the model.
      batch_norm_decay: The moving average decay when estimating layer activation
        statistics in batch normalization.
      batch_norm_epsilon: Small constant to prevent division by zero when
        normalizing activations by their variance in batch normalization.
      batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
        activations in the batch normalization layer.
      activation_fn: The activation function which is used in DenseNet.
      use_batch_norm: Whether or not to use batch normalization.

    Returns:
      An `arg_scope` to use for the DenseNet models.
    """
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'fused': None,  # Use fused batch norm if possible.
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=activation_fn,
            normalizer_fn=slim.batch_norm if use_batch_norm else None,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            # The following implies padding='SAME' for pool1, which makes feature
            # alignment easier for dense prediction tasks. This is also used in
            # https://github.com/facebook/fb.DenseNet.torch. However the accompanying
            # code of 'Deep Residual Learning for Image Recognition' uses
            # padding='VALID' for pool1. You can switch to that choice by setting
            # slim.arg_scope([slim.max_pool2d], padding='VALID').
            with slim.arg_scope([slim.avg_pool2d, slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc
