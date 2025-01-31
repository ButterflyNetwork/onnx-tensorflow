import tensorflow as tf

from onnx_tf_prenatal.handlers.backend_handler import BackendHandler
from onnx_tf_prenatal.handlers.handler import onnx_op


@onnx_op("GlobalMaxPool")
class GlobalMaxPool(BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    dims = tf.range(tf.rank(x))
    _, dim_window = tf.split(dims, [2, tf.size(dims) - 2])
    return [tf.reduce_max(x, axis=dim_window, keepdims=True)]
