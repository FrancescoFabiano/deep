import numpy as np
import onnx
from onnx import helper, TensorProto

# Create input tensor info
input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])

# Create output tensor info
output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 224, 224])

# Create a simple identity node (output = input)
node_def = helper.make_node(
    'Identity',    # node op type
    inputs=['input'],
    outputs=['output']
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [input_tensor],
    [output_tensor]
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-example')

# Save the model
onnx.save(model_def, 'model.onnx')

print("Minimal ONNX model 'model.onnx' created.")
