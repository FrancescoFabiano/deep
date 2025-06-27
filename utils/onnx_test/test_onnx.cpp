#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main() {
  try {
    // Initialize ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    // Create ONNX Runtime session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // Path to your ONNX model - change this to your actual model path
    const char* model_path = "model.onnx";

    // Create session
    Ort::Session session(env, model_path, session_options);

    // Print input info
    size_t num_input_nodes = session.GetInputCount();
    std::cout << "Number of inputs = " << num_input_nodes << std::endl;

    Ort::AllocatorWithDefaultOptions allocator;

    // Get input node names and shapes
    for (size_t i = 0; i < num_input_nodes; i++) {
      char* input_name = session.GetInputName(i, allocator);
      Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

      ONNXTensorElementDataType type = tensor_info.GetElementType();
      std::vector<int64_t> input_node_dims = tensor_info.GetShape();

      std::cout << "Input " << i << " : name=" << input_name << " type=" << type << " dims=";
      for (auto dim : input_node_dims) std::cout << dim << " ";
      std::cout << std::endl;

      allocator.Free(input_name);
    }

    // For example, create input tensor data - modify as needed
    std::vector<int64_t> input_shape = {1, 3, 224, 224}; // Example shape
    size_t input_tensor_size = 1;
    for (auto dim : input_shape) input_tensor_size *= dim;

    std::vector<float> input_tensor_values(input_tensor_size);
    for (size_t i = 0; i < input_tensor_size; i++) {
      input_tensor_values[i] = 0.5f; // Example data
    }

    // Create memory info
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Create input tensor object from data values
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_shape.data(), input_shape.size());

    // Specify input and output names (must match your model)
    const char* input_names[] = {"input"};   // Replace with your model's actual input names
    const char* output_names[] = {"output"}; // Replace with your model's actual output names

    // Run inference
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    // Get pointer to output tensor float values
    float* float_array = output_tensors.front().GetTensorMutableData<float>();

    // Get output shape
    auto output_info = output_tensors.front().GetTensorTypeAndShapeInfo();
    auto output_shape = output_info.GetShape();

    std::cout << "Output tensor shape: ";
    for (auto dim : output_shape) std::cout << dim << " ";
    std::cout << std::endl;

    // Optionally print first few output values
    std::cout << "First 10 output values:" << std::endl;
    for (int i = 0; i < 10 && i < output_info.GetElementCount(); i++) {
      std::cout << float_array[i] << " ";
    }
    std::cout << std::endl;

    return 0;
  } catch (const Ort::Exception& e) {
    std::cerr << "[ERROR] ONNX Runtime failed: " << e.what() << std::endl;
    return 1;
  }
}