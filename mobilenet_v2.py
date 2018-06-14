"""MobileNet v2.

MobileNet is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and different
head (for example: embeddings, localization and classification).

As described in https://arxiv.org/abs/1801.04381.

  MobileNets: Inverted Residuals and Linear Bottlenecks: Mobile Networks for
    Classification, Detection and Segmentation
  Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import convnet_builder as cnn

# _CONV_DEFS specifies the MobileNet body
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
InvertedResidual = namedtuple('InvertedResidual', ['kernel', 'stride', 'depth', 'num', 't']) # t is the expension factor
_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=32),
    InvertedResidual(kernel=[3, 3], stride=1, depth=16, num=1, t=1),
    InvertedResidual(kernel=[3, 3], stride=2, depth=24, num=2, t=6),
    InvertedResidual(kernel=[3, 3], stride=2, depth=32, num=3, t=6),
    InvertedResidual(kernel=[3, 3], stride=2, depth=64, num=4, t=6),
    InvertedResidual(kernel=[3, 3], stride=1, depth=96, num=3, t=6),
    InvertedResidual(kernel=[3, 3], stride=2, depth=160, num=3, t=6),
    InvertedResidual(kernel=[3, 3], stride=1, depth=320, num=1, t=6),
    Conv(kernel=[1, 1], stride=1, depth=1280)
]

def _inverted_residual_bottleneck(cnn, depth, stride, expand_ratio, input_layer=None, input_channel=None, scope=None):
  if input_layer is not None:
    cnn.top_layer = input_layer
    cnn.top_size = input_channel

  input_layer = cnn.top_layer
  in_size = cnn.top_size

  cnn.counts['InvertedResidual'] += 1
  with tf.variable_scope(scope, 'InvertedResidual'+ str(cnn.counts['InvertedResidual']), [input_layer]):
    # if depth == in_size:
    #   if stride == 1:
    #     shortcut = input_layer
    #   else:
    #     shortcut = cnn.apool(
    #         1, 1, stride, stride, input_layer=input_layer,
    #         num_channels_in=in_size)
    # else:
    #   shortcut = cnn.conv(
    #       depth, 1, 1, stride, stride, activation=None, use_batch_norm=False,
    #       input_layer=input_layer, num_channels_in=in_size, bias=None, scope='shortcut')

    cnn.conv(expand_ratio * in_size, 1, 1, 1, 1,
             input_layer=input_layer, num_channels_in=in_size,
             use_batch_norm=True, bias=None, activation='relu6', scope='conv')
    cnn.separable_conv(num_out_channels=None, channel_multiplier=1, 
                        k_height=3, k_width=3, d_height=stride, d_width=stride,
                        use_batch_norm=True, bias=None, activation='relu6', scope='depthwise')
    output = cnn.conv(depth, 1, 1, 1, 1,
             use_batch_norm=True, bias=None, activation='linear', scope='pointwise')

    if stride==1 and depth==in_size:
      shortcut = input_layer
      output = shortcut + output
    cnn.top_layer = output
    cnn.top_size = depth
    return output


def mobilenet_v2_base(cnn, inputs_channel=None,
                      final_endpoint='Conv2d_8',
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=None,
                      output_stride=None,
                      scope=None,
                      input_layer=None,
                      input_channel=None):
  if input_layer is not None:
    cnn.top_layer = input_layer
    cnn.top_size = input_channel
  inputs = cnn.top_layer

  depth = lambda d: max(int(d * depth_multiplier), min_depth)
  end_points = {}
  # Used to find thinned depths for each layer.
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')

  if conv_defs is None:
    conv_defs = _CONV_DEFS

  with tf.variable_scope(scope, 'MobilenetV2', [inputs]):
    # The current_stride variable keeps track of the output stride of the
    # activations, i.e., the running product of convolution strides up to the
    # current network layer. This allows us to invoke atrous convolution
    # whenever applying the next convolution would result in the activations
    # having output stride larger than the target output_stride.
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1

    for i, conv_def in enumerate(conv_defs):
      if output_stride is not None and current_stride == output_stride:
        # If we have reached the target output_stride, then we need to employ
        # atrous convolution with stride=1 and multiply the atrous rate by the
        # current unit's stride for use in subsequent layers.
        layer_stride = 1
        layer_rate = rate
        rate *= conv_def.stride
      else:
        layer_stride = conv_def.stride
        layer_rate = 1
        current_stride *= conv_def.stride

      if isinstance(conv_def, Conv):
        end_point = 'Conv2d_%d' % i
        net = cnn.conv(num_out_channels=depth(conv_def.depth), k_height=conv_def.kernel[0], k_width=conv_def.kernel[1], d_height=conv_def.stride, d_width=conv_def.stride,
                        use_batch_norm=True, scope=end_point)
        end_points[end_point] = net
        if end_point == final_endpoint:
          return net, end_points

      elif isinstance(conv_def, InvertedResidual):
        for n in range(conv_def.num):
          end_point = 'InvertedResidual_{}_{}'.format(conv_def.depth, n)
          stride = conv_def.stride if n == 0 else 1
          net = _inverted_residual_bottleneck(cnn, depth(conv_def.depth), stride, conv_def.t, scope=end_point)
          end_points[end_point] = net
          if end_point == final_endpoint:
            return net, end_points
      else:
        raise ValueError('Unknown convolution type %s for layer %d'
                         % (conv_def.ltype, i))
  raise ValueError('Unknown final endpoint %s' % final_endpoint)

def mobilenet_v2(cnn,
                 num_classes=1000,
                 dropout_keep_prob=0.999,
                 is_training=True,
                 min_depth=8,
                 depth_multiplier=1.0,
                 conv_defs=None,
                 prediction_fn=tf.contrib.layers.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='MobilenetV2',
                 global_pool=False):
  """Mobilenet v2 model for classification.

  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer (before dropout)
      are returned instead.
    dropout_keep_prob: the percentage of activation values that are retained.
    is_training: whether is training or not.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    conv_defs: A list of ConvDef namedtuples specifying the net architecture.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape is [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    global_pool: Optional boolean flag to control the avgpooling before the
      logits layer. If false or unset, pooling is done with a fixed window
      that reduces default-sized inputs to 1x1, while larger inputs lead to
      larger outputs. If true, any input size is pooled down to 1x1.

  Returns:
    net: a 2D Tensor with the logits (pre-softmax activations) if num_classes
      is a non-zero integer, or the non-dropped-out input to the logits layer
      if num_classes is 0 or None.
    end_points: a dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: Input rank is invalid.
  """
  inputs = cnn.top_layer
  cnn.regularizer=tf.contrib.layers.l2_regularizer(0.0004)
  with tf.variable_scope(scope, 'MobilenetV2', [inputs], reuse=reuse) as scope:
    net, end_points = mobilenet_v2_base(cnn, scope=scope,
                                        min_depth=min_depth,
                                        depth_multiplier=depth_multiplier,
                                        conv_defs=conv_defs)
    with tf.variable_scope('Logits'):
      if global_pool:
        # Global average pooling.
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      else:
        # Pooling with a fixed kernel size.
        kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
        net = cnn.apool(kernel_size[0], kernel_size[1], mode='VALID')
        end_points['average_pool'] = net
        cnn.dropout(keep_prob=dropout_keep_prob)
        net =  cnn.spatial_mean()
        end_points['features'] = net
      if not num_classes:
        return net, end_points
      # 1 x 1 x 1024
      logits = cnn.affine(num_classes, activation='linear')
    end_points['Logits'] = logits
    if prediction_fn:
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.
  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[2], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
  return kernel_size_out


def wrapped_partial(func, *args, **kwargs):
  partial_func = functools.partial(func, *args, **kwargs)
  functools.update_wrapper(partial_func, func)
  return partial_func


mobilenet_v2_075 = wrapped_partial(mobilenet_v2, depth_multiplier=0.75)
mobilenet_v2_050 = wrapped_partial(mobilenet_v2, depth_multiplier=0.50)
mobilenet_v2_025 = wrapped_partial(mobilenet_v2, depth_multiplier=0.25)