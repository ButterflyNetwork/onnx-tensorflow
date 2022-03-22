from onnx_tf.common import get_perm_from_formats
import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("GlobalAveragePool")
class GlobalAveragePool(BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    
    # Use average pool as mean is not reliably supported across dimensions
    # in coreml/nnapi
    result = tf.nn.avg_pool2d(x, x.get_shape().as_list()[1:-1], [1,1], "VALID")

    return [result]
