from onnx_tf_prenatal.handlers.backend_handler import BackendHandler
from onnx_tf_prenatal.handlers.handler import onnx_op


@onnx_op("OptionalGetElement")
class OptionalGetElement(BackendHandler):

  @classmethod
  def version_15(cls, node, **kwargs):
    if len(node.inputs) > 0:
        return [kwargs["tensor_dict"][node.inputs[0]]]
    else:
        raise RuntimeError("No element value!.")
