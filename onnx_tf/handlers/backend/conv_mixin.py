import tensorflow as tf

from onnx_tf.common import exception
from onnx_tf.common import get_data_format
from onnx_tf.common import get_perm_from_formats
from onnx_tf.common.tf_helper import tf_shape
from onnx_tf.common import sys_config
from .broadcast_mixin import BroadcastMixin
from .pad_mixin import PadMixin

# Constant string used to indicate that requested padding
# is not natively supported in Tensorflow.
PAD_TF_INCOMPATIBLE = "PAD_TF_INCOMPATIBLE"


class ConvMixin(BroadcastMixin):

  @classmethod
  def conv(cls, node, input_dict, transpose=False):
    """Only supports our Gates use case - special reimplementation
    
    Only supports input tensors in NHWC format, conv_transpose is not
    supported.  This ordering is assumed and not validated."""
    x = input_dict[node.inputs[0]]
    x_rank = len(x.get_shape())

    in_weights = input_dict[node.inputs[1]]
    weights_rank = len(in_weights.get_shape())

    if transpose:
      exception.OP_UNIMPLEMENTED_EXCEPT("Conv transpose is not supported")
    # Translate weights from (M x C x KH x KW) to (KH x KW X C X M)
    perm = list(range(2, weights_rank)) + [1, 0]

    if "kernel_shape" in node.attrs.keys():
      kernel_shape = node.attrs["kernel_shape"]
      if in_weights.get_shape().is_fully_defined():
        assert in_weights.get_shape().as_list()[2:] == kernel_shape, (
            "kernel_shape "
            "attr of convolution does not match the actual weight "
            "passed to this operation, attr {}, actual {}").format(
                kernel_shape,
                in_weights.get_shape().as_list())
    else:
      kernel_shape = tf_shape(in_weights, tf.int32)[2:]

    spatial_size = x_rank - 2

    weights = tf.transpose(in_weights, perm)
    dilations = node.attrs.get("dilations", [1] * spatial_size)
    if tuple(dilations) != tuple([1] * spatial_size) :
      exception.OP_UNIMPLEMENTED_EXCEPT("Non-unit dilation not supported")

    strides = node.attrs.get("strides", [1] * spatial_size)

    pads = node.attrs.get("pads", [0, 0] * spatial_size)

    # Check auto_pad nonexistent or NOTSET first
    if "auto_pad" not in node.attrs or node.attrs["auto_pad"] == "NOTSET":
      if pads != [0, 0] * spatial_size:
        x = PadMixin.get_padding_as_op(x, pads, channels_last=True)
        
      pad_mode = "VALID"
    elif node.attrs["auto_pad"] == "VALID":
      pad_mode = "VALID"
    else:
      exception.OP_UNSUPPORTED_EXCEPT("Conv must have VALID or explicit padding")
    

    group = node.attrs.get("group", 1)
    weight_shape = weights.get_shape().as_list()

    # Is this convolution depthwise we can support?
    depthwise = (x_rank == 4 and len(weight_shape) == 4 and group != 1 and
                 not transpose and not (None in weight_shape))
    if depthwise and x.get_shape().as_list()[3] != None:
      depthwise = bool(group == x.get_shape().as_list()[3])

    if depthwise is True:
      # Depthwise convolution.
      # The convolution kernel layout in tf.depthwise_conv is:
      # [filter_height, filter_width, in_channels, channel_multiplier]
      # Weight is now (KH x KW X C/g X M), or more precisely, (KH x KW X C/g X (g * M/g)),
      # we reshape it to (KH x KW x C x M/g)
      # NOTE: Assuming weight has fixed shape.

      depthwise_filter_shape = weight_shape[0:2] + [
          -1, weight_shape[3] // group
      ]
      weights = tf.reshape(weights, depthwise_filter_shape)

      weight_groups = [weights]
      xs = [x]
    else:
      weight_groups = tf.split(weights, num_or_size_splits=group, axis=-1)
      
      if group == 1:
        xs = [x]
      else:
        xs = tf.split(x, num_or_size_splits=group, axis=-1)

    if depthwise is True:
      strides = [1] + strides + [1]

      convolved = [
          tf.nn.depthwise_conv2d(
            x,
            weight,
            padding=pad_mode,
            strides=strides,
            dilations=None,
            data_format="NHWC"
          )
          for (x, weight) in zip(xs, weight_groups)
      ]

    else:
      convolved = [
          tf.nn.conv2d(
            x,
            weight,
            padding=pad_mode,
            strides=strides,
            dilations=None,
            data_format="NHWC")
          for (x, weight) in zip(xs, weight_groups)
      ]

    if len(convolved) == 1:
      output = convolved[0]
    else:
      output = tf.concat(convolved, axis=-1)
    
    if len(node.inputs) > 2:
      bias = input_dict[node.inputs[2]]
      bias = cls.explicit_broadcast([x, bias], 3)
      output = tf.add(output, bias)

    return [output]
