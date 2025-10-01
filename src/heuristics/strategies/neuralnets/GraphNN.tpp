#include "ExitHandler.h"
#include "GraphNN.h"
#include "TrainingDataset.h"
#include <fstream>
#include <regex>

// --- Singleton instance initialization ---
template <StateRepresentation StateRepr>
GraphNN<StateRepr> *GraphNN<StateRepr>::instance = nullptr;

template <StateRepresentation StateRepr>
GraphNN<StateRepr> &GraphNN<StateRepr>::get_instance() {
  if (!instance) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::GNNInstanceError,
        "GraphNN instance not created. Call create_instance() first.");
    std::exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));
  }
  return *instance;
}

template <StateRepresentation StateRepr>
void GraphNN<StateRepr>::create_instance() {
  if (!instance) {
    instance = new GraphNN();
  }
}

template <StateRepresentation StateRepr> GraphNN<StateRepr>::GraphNN() {
  // Use default StateRepresentation for dataset folder creation
  TrainingDataset<StateRepr>::create_instance();
  m_checking_file_path =
      TrainingDataset<StateRepr>::get_instance().get_folder() +
      "to_predict.dot";

  /// \todo remove this, not needed if inference is done directly
  m_goal_file_path =
      TrainingDataset<StateRepr>::get_instance().get_goal_file_path();

  populate_with_goal();
  initialize_onnx_model();
}

template <StateRepresentation StateRepr>
void GraphNN<StateRepr>::initialize_onnx_model() {
  if (m_model_loaded)
    return;

  try {
    m_session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    /*#ifdef _WIN32
        // Windows way
        _putenv_s("ORT_CUDA_USE_CUDNN", "0");
        _putenv_s("CUDA_LAUNCH_BLOCKING", "1");  // optional, forces sync errors
    #else
        // Linux / WSL / macOS way
        setenv("ORT_CUDA_USE_CUDNN", "0", 1);
        setenv("CUDA_LAUNCH_BLOCKING", "1", 1);  // optional
    #endif*/

    // Add this line to show warnings (2) to complete verbose (0) (only errors
    // and above will be shown)
    if (ArgumentParser::get_instance().get_verbose()) {
      m_session_options.SetLogSeverityLevel(0);
    }

#ifdef USE_CUDA
    try {
      OrtCUDAProviderOptions cuda_options;
      m_session_options.AppendExecutionProvider_CUDA(cuda_options);
      if (ArgumentParser::get_instance().get_verbose()) {
        ArgumentParser::get_instance().get_output_stream()
            << "[ONNX] CUDA execution provider enabled via USE_CUDA."
            << std::endl;
      }
    } catch (const Ort::Exception &e) {
      ArgumentParser::get_instance().get_output_stream()
          << "[WARNING][ONNX] Failed to enable CUDA, defaulting to CPU: "
          << e.what() << std::endl;
    }
#else
    if (ArgumentParser::get_instance().get_verbose()) {
      ArgumentParser::get_instance().get_output_stream()
          << "[ONNX] Compiled without CUDA (USE_CUDA not defined), using CPU."
          << std::endl;
    }
#endif

    m_session = std::make_unique<Ort::Session>(m_env, m_model_path.c_str(),
                                               m_session_options);
    m_allocator = std::make_unique<Ort::AllocatorWithDefaultOptions>();
    m_memory_info = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));

    // Fixed: No allocator argument
    m_input_names = m_session->GetInputNames();
    m_output_names = m_session->GetOutputNames();

    parse_constant_for_normalization();
    m_model_loaded = true;
  } catch (const std::exception &e) {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::GNNModelLoadError,
                                   std::string("Failed to load ONNX model: ") +
                                       e.what());
  }

  if (ArgumentParser::get_instance().get_verbose()) {
    auto &os = ArgumentParser::get_instance().get_output_stream()
               << "[ONNX] Model loaded: " << m_model_path << std::endl;
    os << "[ONNX] Model Constant loaded from: "
       << Configuration::get_instance().get_GNN_constant_path() << std::endl;

    // Print model input and output details
    const auto input_names = m_session->GetInputNames();
    const auto output_names = m_session->GetOutputNames();

    os << "[ONNX] Model Inputs:\n";
    for (size_t i = 0; i < input_names.size(); ++i) {
      const auto &name = input_names[i];
      auto type_info = m_session->GetInputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      const auto element_type = tensor_info.GetElementType();
      auto shape = tensor_info.GetShape();

      os << "  Name: " << name << "\n";
      os << "  Type: " << element_type << "\n";
      os << "  Shape: [";
      for (size_t j = 0; j < shape.size(); ++j) {
        os << shape[j];
        if (j < shape.size() - 1)
          os << ", ";
      }
      os << "]\n";
    }

    os << "[ONNX] Model Outputs:\n";
    for (size_t i = 0; i < output_names.size(); ++i) {
      const auto &name = output_names[i];
      auto type_info = m_session->GetOutputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      auto element_type = tensor_info.GetElementType();
      auto shape = tensor_info.GetShape();

      os << "  Name: " << name << "\n";
      os << "  Type: " << element_type << "\n";
      os << "  Shape: [";
      for (size_t j = 0; j < shape.size(); ++j) {
        os << shape[j];
        if (j < shape.size() - 1)
          os << ", ";
      }
      os << "]\n";
    }
    os << "[ONNX] Model successfully printed." << std::endl;
  }
}

template <StateRepresentation StateRepr>
[[nodiscard]] int GraphNN<StateRepr>::get_score(const State<StateRepr> &state) {
  const auto state_tensor = state_to_tensor_minimal(state.get_representation());

#ifdef DEBUG
  if (ArgumentParser::get_instance().get_verbose()) {
    if (!check_tensor_against_dot(state_tensor, state)) {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::GNNTensorTranslationError,
          "Error while comparing the state/goal in " + m_checking_file_path +
              " or " + m_goal_file_path +
              ". Its tensor representation generated a different dot file.");
    } else {
      ArgumentParser::get_instance().get_output_stream()
          << "[DEBUG] State and goal represented as dot and as tensor match:)"
          << std::endl;
    }

    // std::cout << "[DEBUG] REMOVE THIS" << std::endl;
    // return 1; // Skip actual inference for debug mode

    // compare_predictions(state, run_inference(state_tensor));
  }
#endif

  if (const float inference_result = run_inference(state_tensor);
      inference_result < 0) {
    return 0;
  } else {
    return static_cast<int>(
        std::round((inference_result - m_normalization_intercept) /
                   m_normalization_slope));
  }
}

template <StateRepresentation StateRepr>
float GraphNN<StateRepr>::run_inference(const GraphTensor &tensor) const {
  if (!m_model_loaded) {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::GNNInstanceError,
                                   "[ONNX] Model not loaded before inference.");
  }

  if (ArgumentParser::get_instance().get_dataset_separated()) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::GNNTensorTranslationError,
        "Separated dataset type not supported for GNN inference.");
  }

  auto &session = *m_session;
  const auto &memory_info = *m_memory_info;

  const auto dataset_type = ArgumentParser::get_instance().get_dataset_type();
  const bool is_bitmask = (dataset_type == DatasetType::BITMASK);

  const size_t num_edges = tensor.edge_src.size();
  size_t num_nodes;

  if (is_bitmask) {
    num_nodes = tensor.real_node_ids_bitmask.size() / m_bitmask_size;
  } else {
    num_nodes = tensor.real_node_ids.size();
  }
  Ort::Value real_node_ids_tensor;
  Ort::Value real_node_ids_bitmask_tensor;

  if (is_bitmask) {
    // Shape [num_nodes, bitmask_size]
    const std::array<int64_t, 2> bitmask_shape{
        static_cast<int64_t>(num_nodes), static_cast<int64_t>(m_bitmask_size)};

    // ONNX wants a non-const pointer even though it doesn't mutate it.
    uint8_t *bitmask_ptr =
        const_cast<uint8_t *>(tensor.real_node_ids_bitmask.data());

    real_node_ids_bitmask_tensor = Ort::Value::CreateTensor<uint8_t>(
        memory_info, bitmask_ptr, tensor.real_node_ids_bitmask.size(),
        bitmask_shape.data(), bitmask_shape.size());
  } else {
    // Construct real_node_ids tensor: shape [num_nodes, 1]
    std::vector<float> real_node_ids_float(tensor.real_node_ids.begin(),
                                           tensor.real_node_ids.end());
    const std::array<int64_t, 1> node_ids_shape{
        static_cast<int64_t>(real_node_ids_float.size())};
    real_node_ids_tensor = Ort::Value::CreateTensor<float>(
        memory_info, real_node_ids_float.data(), real_node_ids_float.size(),
        node_ids_shape.data(), node_ids_shape.size());
  }

  // Construct edge_index tensor: shape [2, num_edges]
  std::vector<int64_t> edge_index_data(2 * num_edges);
  for (size_t i = 0; i < num_edges; ++i) {
    edge_index_data[i] = (tensor.edge_src[i]); // First row: edge_src
    edge_index_data[num_edges + i] =
        (tensor.edge_dst[i]); // Second row: edge_dst
  }

  const std::array<int64_t, 2> edge_index_shape{
      2, static_cast<int64_t>(num_edges)};
  Ort::Value edge_index_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, edge_index_data.data(), edge_index_data.size(),
      edge_index_shape.data(), edge_index_shape.size());

  // Construct edge_attr tensor: shape [num_edges, 1]
  std::vector<float> edge_attrs_float(tensor.edge_attrs.begin(),
                                      tensor.edge_attrs.end());
  const std::array<int64_t, 2> edge_attr_shape{static_cast<int64_t>(num_edges),
                                               1};
  Ort::Value edge_attr_tensor = Ort::Value::CreateTensor<float>(
      memory_info, edge_attrs_float.data(), edge_attrs_float.size(),
      edge_attr_shape.data(), edge_attr_shape.size());

  // Construct state_batch tensor: shape [-1]
  std::vector<int64_t> state_batch_data(num_nodes, 0);
  const std::array<int64_t, 1> state_batch_shape{
      static_cast<int64_t>(state_batch_data.size())};
  Ort::Value state_batch_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, state_batch_data.data(), state_batch_data.size(),
      state_batch_shape.data(), state_batch_shape.size());

  // Prepare input tensors
  // Prepare input tensors
  std::vector<Ort::Value> input_tensors;
  if (is_bitmask) {
    input_tensors.emplace_back(std::move(real_node_ids_bitmask_tensor));
  } else {
    input_tensors.emplace_back(std::move(real_node_ids_tensor));
  }
  input_tensors.emplace_back(std::move(edge_index_tensor));
  input_tensors.emplace_back(std::move(edge_attr_tensor));
  input_tensors.emplace_back(std::move(state_batch_tensor));

  // Convert input/output names to const char* arrays
  std::vector<const char *> input_names_cstr;
  input_names_cstr.reserve(m_input_names.size());
  for (const auto &name : m_input_names) {
    input_names_cstr.push_back(name.c_str());
  }
  std::vector<const char *> output_names_cstr;
  output_names_cstr.reserve(m_output_names.size());
  for (const auto &name : m_output_names) {
    output_names_cstr.push_back(name.c_str());
  }

  // Run the model
  auto output_tensors = session.Run(
      Ort::RunOptions{nullptr}, input_names_cstr.data(), input_tensors.data(),
      input_tensors.size(), output_names_cstr.data(), output_names_cstr.size());

  // Get the result (assuming scalar output)
  const auto *output_data = output_tensors[0].GetTensorMutableData<float>();
  const float score = output_data[0];

  return score;
}

/*template <StateRepresentation StateRepr>
void GraphNN<StateRepr>::compare_predictions(const State<StateRepr> &state,
                                             float c_score) {
  std::ofstream ofs(m_checking_file_path);
  if (!ofs) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::GNNFileError,
        "Failed to open file for NN state checking: " + m_checking_file_path);
  }

  state.print_dataset_format(
      ofs, !ArgumentParser::get_instance().get_dataset_mapped(),
      ArgumentParser::get_instance().get_dataset_separated());

  const std::string output_file = "prediction_results.out";
  const std::string shell_script = "lib/RL/run_check.sh";

  const std::string command =
      shell_script + std::string(" ") + m_checking_file_path + " " +
      "lib/RL/models/best.pt " + "lib/RL/models/distance_estimator.onnx ";

  int ret_code = std::system(command.c_str());
  if (ret_code != 0) {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::GNNFileError,
                                   "Shell script failed with exit code " +
                                       std::to_string(ret_code));
  }

  std::ifstream infile(output_file);
  if (!infile.is_open()) {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::GNNFileError,
                                   "Could not open output file: " +
                                       output_file);
  }

  float pytorch_pred = -1.0f;
  float onnx_pred = -1.0f;

  std::string line;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    std::string label;
    float value;

    if (line.rfind("PyTorch:", 0) == 0) {
      if (iss >> label >> value) {
        pytorch_pred = value;
      }
    } else if (line.rfind("ONNX:", 0) == 0) {
      if (iss >> label >> value) {
        onnx_pred = value;
      }
    }
  }

  infile.close();

  // Validate all values are set
  if (pytorch_pred < 0 || onnx_pred < 0) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::GNNFileError,
        "Could not parse prediction values correctly from output file: " +
            output_file);
  }

  // Compare the values
  constexpr float tolerance = 1e-2f; // Define tolerance here

  bool torch_diff = std::abs(c_score - pytorch_pred) > tolerance;
  bool onnx_diff = std::abs(c_score - onnx_pred) > tolerance;
  bool torch_vs_onnx_diff = std::abs(pytorch_pred - onnx_pred) > tolerance;

  if (torch_diff || onnx_diff || torch_vs_onnx_diff) {
    std::ostringstream oss;
    oss << "Predictions differ beyond tolerance (" << tolerance << ")\n";
    oss << "  C++ planner score: " << c_score << "\n";
    oss << "  PyTorch prediction: " << pytorch_pred << "\n";
    oss << "  ONNX prediction: " << onnx_pred << "\n";
    oss << "  |C++ - PyTorch| = " << std::abs(c_score - pytorch_pred) << "\n";
    oss << "  |C++ - ONNX|    = " << std::abs(c_score - onnx_pred) << "\n";
    oss << "  |PyTorch - ONNX| = " << std::abs(pytorch_pred - onnx_pred)
        << "\n";

    ExitHandler::exit_with_message(ExitHandler::ExitCode::GNNFileError,
                                   oss.str());
  }

  std::cout << "[DEBUG] All predictions within tolerance (" << tolerance
            << ")\n";
}*/

/*template <StateRepresentation StateRepr>
[[nodiscard]] short
GraphNN<StateRepr>::get_score_python(const State<StateRepr> &state) {
  // Print the state in the required dataset format to the checking file
  std::ofstream ofs(m_checking_file_path);
  if (!ofs) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::GNNFileError,
        "Failed to open file for NN state checking: " + m_checking_file_path);
  }
  state.print_dataset_format(
      ofs, !ArgumentParser::get_instance().get_dataset_mapped(),
      ArgumentParser::get_instance().get_dataset_separated());

  // Run the external Python script for NN inference
  std::string command = "./lib/RL/run_prediction.sh " + m_checking_file_path +
                        " " + std::to_string(state.get_plan_length()) + " " +
                        m_goal_file_path + " " + m_model_path;

  std::array<char, 128> buffer{};
  std::string result;

  // Open a pipe to the shell
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"),
                                                pclose);
  if (!pipe) {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::GNNScriptError,
                                   "Failed to run Python script");
  }

  // Read the output from the Python script
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }

  try {
    return static_cast<short>(std::stod(result));
  } catch (const std::exception &e) {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::GNNScriptError,
                                   "Error converting prediction to double: " +
                                       std::string(e.what()));
  }

  // Just to please the compiler
  std::exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));
}*/

template <StateRepresentation StateRepr>
size_t GraphNN<StateRepr>::get_symbolic_id(const size_t node,
                                           const KripkeWorldPointer &kworld) {

  // Since we pass the kworld, we are sure this is a state node and not a goal
  // node

  if (!m_node_to_symbolic.contains(node)) {
    m_node_to_symbolic[node] = m_symbolic_id++;

    // Not converting to bitmask if:
    //  1. Not bitmask dataset
    if (ArgumentParser::get_instance().get_dataset_type() !=
        DatasetType::BITMASK) {
      m_real_node_ids.push_back(node);
    } else {
      // Convert from kworld id
      const std::string bitmask_str = HelperPrint::kworld_to_bitmask(
          kworld, !ArgumentParser::get_instance().get_dataset_separated(),
          Domain::get_instance().get_positive_fluents());

      if (m_bitmask_size == 0) {
        m_bitmask_size = bitmask_str.size();
      }

      // Append new row into flattened storage
      const size_t start = m_real_node_ids_bitmask.size();
      m_real_node_ids_bitmask.resize(start + m_bitmask_size);

      for (size_t j = 0; j < m_bitmask_size; ++j) {
        char c = bitmask_str[j];
        if (c == '0') {
          m_real_node_ids_bitmask[start + j] = 0;
        } else if (c == '1') {
          m_real_node_ids_bitmask[start + j] = 1;
        } else {
          ExitHandler::exit_with_message(
              ExitHandler::ExitCode::GNNBitmaskGOALError,
              "Invalid character in bitmask string.");
        }
      }
    }
  }
  return m_node_to_symbolic[node];
}

template <StateRepresentation StateRepr>
size_t GraphNN<StateRepr>::get_symbolic_id(const size_t node) {

  // Since we do not pass information on the kworld, this is a goal node

  if (!m_node_to_symbolic.contains(node)) {
    m_node_to_symbolic[node] = m_symbolic_id++;

    // Not converting to bitmask if:
    //  1. Not bitmask dataset
    //  2. Bitmask dataset but separated. This happens if we are doing bitmask
    //  but the goal is printed out separately in normal mode.
    if (ArgumentParser::get_instance().get_dataset_type() !=
            DatasetType::BITMASK ||
        ArgumentParser::get_instance().get_dataset_separated()) {
      m_real_node_ids.push_back(node);
    } else {
      const size_t start = m_real_node_ids_bitmask.size();
      m_real_node_ids_bitmask.resize(start + BITMASK_DIM);

      // [0] = MSB, [BITMASK_DIM-1] = LSB
      // Fill from LSB upward in the integer while placing into MSB..LSB
      // positions
      uint64_t val = static_cast<uint64_t>(node);
      for (size_t i = 0; i < BITMASK_DIM; ++i) {
        // Take bit i (LSB-first), store at position (BITMASK_DIM-1 - i) to get
        // MSB-first layout
        m_real_node_ids_bitmask[start + (BITMASK_DIM - 1 - i)] =
            static_cast<uint8_t>(val & 1ULL);
        val >>= 1ULL;
      }
    }
  }
  return m_node_to_symbolic[node];
}

template <StateRepresentation StateRepr>
void GraphNN<StateRepr>::add_edge(const size_t src,
                                  const KripkeWorldPointer &src_kworld,
                                  const size_t dst,
                                  const KripkeWorldPointer &dst_kworld,
                                  const int64_t label) {

  // Both src and dst have kworld information because they are states kworlds
  // and not goal nodes
  m_edge_src.push_back(get_symbolic_id(src, src_kworld));
  m_edge_dst.push_back(get_symbolic_id(dst, dst_kworld));
  m_edge_labels.push_back(label);
}

template <StateRepresentation StateRepr>
void GraphNN<StateRepr>::add_edge(const size_t src, const size_t dst,
                                  const KripkeWorldPointer &dst_kworld,
                                  const int64_t label) {
  // Only dst has kworld information because it is a state kworld and not a goal
  // node
  m_edge_src.push_back(get_symbolic_id(src));
  m_edge_dst.push_back(get_symbolic_id(dst, dst_kworld));
  m_edge_labels.push_back(label);
}

template <StateRepresentation StateRepr>
void GraphNN<StateRepr>::add_edge(const size_t src, const size_t dst,
                                  const int64_t label) {
  m_edge_src.push_back(get_symbolic_id(src));
  m_edge_dst.push_back(get_symbolic_id(dst));
  m_edge_labels.push_back(label);
}

template <StateRepresentation StateRepr>
void GraphNN<StateRepr>::parse_constant_for_normalization() {
  std::string filename = Configuration::get_instance().get_GNN_constant_path();
  std::ifstream infile(filename);
  if (!infile) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::GNNFileError,
        "Failed to open normalization constant file: " + filename);
  }

  std::string line;

  std::regex pattern_slope(R"(slope\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?))");
  std::regex pattern_intercept(
      R"(intercept\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?))");

  bool slope_found = false;
  bool intercept_found = false;

  while (std::getline(infile, line)) {
    std::smatch match;

    if (!slope_found && std::regex_search(line, match, pattern_slope) &&
        match.size() == 2) {
      try {
        m_normalization_slope = std::stof(match[1].str());
        slope_found = true;
      } catch (const std::exception &e) {
        ExitHandler::exit_with_message(ExitHandler::ExitCode::GNNFileError,
                                       "Failed to parse normalization slope: " +
                                           std::string(e.what()));
      }
    }

    if (!intercept_found && std::regex_search(line, match, pattern_intercept) &&
        match.size() == 2) {
      try {
        m_normalization_intercept = std::stof(match[1].str());
        intercept_found = true;
      } catch (const std::exception &e) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::GNNFileError,
            "Failed to parse normalization intercept: " +
                std::string(e.what()));
      }
    }

    if (slope_found && intercept_found) {
      break; // both found, can stop early
    }
  }

  if (!slope_found) {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::GNNFileError,
                                   "Normalization slope not found in file: " +
                                       filename);
  }
  if (!intercept_found) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::GNNFileError,
        "Normalization intercept not found in file: " + filename);
  }
}

template <StateRepresentation StateRepr>
void GraphNN<StateRepr>::populate_with_goal() {
  std::string goal_graph_string;

  if (ArgumentParser::get_instance().get_dataset_type() ==
      DatasetType::BITMASK) {
    goal_graph_string =
        TrainingDataset<StateRepr>::get_instance().get_goal_forced_string();
  } else {
    goal_graph_string =
        TrainingDataset<StateRepr>::get_instance().get_goal_string();
  }

  const std::regex pattern(R"((\d+)\s*->\s*(\d+)\s*\[label=\"(\d+)\"\];)");
  const std::sregex_iterator begin(goal_graph_string.begin(),
                                   goal_graph_string.end(), pattern);
  const std::sregex_iterator end;

  if (!ArgumentParser::get_instance().get_dataset_separated()) {
    constexpr auto epsilon_id =
        TrainingDataset<KripkeState>::get_epsilon_node_id_int();
    constexpr auto goal_parent_id =
        TrainingDataset<KripkeState>::get_goal_parent_id_int();

    add_edge(epsilon_id, goal_parent_id,
             TrainingDataset<KripkeState>::get_to_goal_edge_id_int());
  } else {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::GNNTensorTranslationError,
        "Separated dataset type not supported for GNN inference.");
  }

  for (auto it = begin; it != end; ++it) {
    const size_t src = std::stoul((*it)[1]);
    const size_t dst = std::stoul((*it)[2]);
    const size_t label = std::stoul((*it)[3]);

    add_edge(src, dst, label);
  }

  if (ArgumentParser::get_instance().get_dataset_separated()) {
    fill_graph_tensor(m_goal_graph_tensor);

    m_edge_dst.clear();
    m_edge_src.clear();
    m_edge_labels.clear();
    m_real_node_ids.clear();
    m_real_node_ids_bitmask.clear();
    m_node_to_symbolic.clear();
    m_symbolic_id = 0;

    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::GNNTensorTranslationError,
        "Separated dataset type not supported for GNN inference.");
  }

  m_edges_initial_size = m_edge_labels.size();
  m_node_ids_initial_size = m_real_node_ids.size();
  m_real_node_ids_bitmask_initial_size = m_real_node_ids_bitmask.size();
  m_starting_symbolic_id = m_symbolic_id;
}

template <StateRepresentation StateRepr>
GraphTensor
GraphNN<StateRepr>::state_to_tensor_minimal(const KripkeState &kstate) {

  const auto m_node_to_symbolic_original = m_node_to_symbolic;

  const auto &training_dataset = TrainingDataset<KripkeState>::get_instance();
  const bool is_merged =
      !ArgumentParser::get_instance().get_dataset_separated();

  std::unordered_map<KripkeWorldId, KripkeWorldId> world_map;
  const auto dataset_type = ArgumentParser::get_instance().get_dataset_type();
  int world_counter = training_dataset.get_shift_state_ids();

  if (is_merged) {
    const auto state_parent = kstate.get_pointed();
    const auto state_parent_id = state_parent.get_id();

    add_edge(TrainingDataset<KripkeState>::get_epsilon_node_id_int(),
             state_parent_id, state_parent,
             TrainingDataset<KripkeState>::get_to_state_edge_id_int());
  } else {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::GNNTensorTranslationError,
        "Separated dataset type not supported for GNN inference.");
  }

  // Assign IDs ///\todo remove this for efficiency. The hash can be used
  // directly
  for (const auto &pw : kstate.get_worlds()) {
    if (const auto hash = pw.get_id(); !world_map.contains(hash)) {
      switch (dataset_type) {
      case DatasetType::HASHED:
      case DatasetType::BITMASK:
        world_map[hash] = hash;
        break;
      case DatasetType::MAPPED: {
        world_map[hash] = world_counter++;
        break;
      }
      default: {
        ExitHandler::exit_with_message(ExitHandler::ExitCode::ArgParseError,
                                       "Invalid Dataset Type specified");
      }
      }
    }
  }

  for (const auto &[from_pw, from_map] : kstate.get_beliefs()) {
    const size_t src = world_map[from_pw.get_id()];

    for (const auto &[agent, to_set] : from_map) {
      const auto label = static_cast<int64_t>(
          training_dataset.get_unique_a_id_from_map(agent));

      for (const auto &to_pw : to_set) {
        const size_t dst = world_map[to_pw.get_id()];

        add_edge(src, from_pw, dst, to_pw, label);
      }
    }
  }

  GraphTensor ret;
  fill_graph_tensor(ret);

  // Erase only the newly inserted elements
  m_edge_src.erase(m_edge_src.begin() + m_edges_initial_size, m_edge_src.end());
  m_edge_dst.erase(m_edge_dst.begin() + m_edges_initial_size, m_edge_dst.end());
  m_edge_labels.erase(m_edge_labels.begin() + m_edges_initial_size,
                      m_edge_labels.end());
  m_real_node_ids.erase(m_real_node_ids.begin() + m_node_ids_initial_size,
                        m_real_node_ids.end());
  m_real_node_ids_bitmask.erase(m_real_node_ids_bitmask.begin() +
                                    m_real_node_ids_bitmask_initial_size,
                                m_real_node_ids_bitmask.end());
  m_node_to_symbolic = m_node_to_symbolic_original;
  m_symbolic_id = m_starting_symbolic_id;

  return ret;
}

template <StateRepresentation StateRepr>
void GraphNN<StateRepr>::fill_graph_tensor(GraphTensor &tensor) const {
  tensor.edge_src = m_edge_src;
  tensor.edge_dst = m_edge_dst;
  tensor.edge_attrs = m_edge_labels;
  tensor.real_node_ids = m_real_node_ids;
  tensor.real_node_ids_bitmask = m_real_node_ids_bitmask;
}

template <StateRepresentation StateRepr>
bool GraphNN<StateRepr>::check_tensor_against_dot(
    const GraphTensor &state_tensor, const State<StateRepr> &state) const {
  // Write the state's dataset format to m_checking_file_path
  std::ofstream ofs_orig(m_checking_file_path);
  if (!ofs_orig) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::GNNFileError,
        "Failed to open file for NN state checking: " + m_checking_file_path);
  }
  state.print_dataset_format(ofs_orig);
  ofs_orig.close();

  // Prepare modified path string
  bool ret = write_and_compare_tensor_to_dot(m_checking_file_path, state_tensor,
                                             false);
  if (ArgumentParser::get_instance().get_dataset_separated() && ret) {
    ret = write_and_compare_tensor_to_dot(m_goal_file_path, m_goal_graph_tensor,
                                          true) &&
          ret;
  }

  return ret;
}

template <StateRepresentation StateRepr>
bool GraphNN<StateRepr>::write_and_compare_tensor_to_dot(
    const std::string &origin_filename, const GraphTensor &state_tensor,
    bool is_goal) const {
  // Prepare modified path string
  std::string modified_path = origin_filename; // copy original

  if (size_t pos = modified_path.rfind(".dot"); pos != std::string::npos) {
    modified_path.insert(pos, "_compare");
  } else {
    modified_path += "_compare.dot";
  }

  std::ofstream ofs(modified_path);
  if (!ofs) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::GNNFileError,
        "Failed to open file for NN state/goal comparison: " + modified_path);
  }

  ofs << "digraph G {\n";

  // edge_ids shape: [2, num_edges]
  auto edge_attrs = state_tensor.edge_attrs;
  auto edge_src = state_tensor.edge_src;
  auto edge_dst = state_tensor.edge_dst;
  auto real_node_ids = state_tensor.real_node_ids;
  auto real_node_ids_bitmask = state_tensor.real_node_ids_bitmask;

  const auto num_edges = edge_src.size();
  if (num_edges != edge_dst.size()) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::GNNTensorTranslationError,
        "Inconsistent edge data in GraphTensor: "
        "edge_src and edge_dst must have the same number of edges.");
  }

  // Print edges with optional edge_attrs label
  for (size_t e = 0; e < num_edges; ++e) {
    int64_t src_symbolic = edge_src[e];
    int64_t dst_symbolic = edge_dst[e];

    int64_t attr = edge_attrs[e];
    std::string src_str, dst_str;

    if (ArgumentParser::get_instance().get_dataset_type() ==
            DatasetType::BITMASK &&
        !is_goal) {
      src_str = "";
      dst_str = "";
      size_t iteration = 0;
      while (iteration < m_bitmask_size) {
        src_str += std::to_string(
            real_node_ids_bitmask[src_symbolic * m_bitmask_size + iteration]);
        dst_str += std::to_string(
            real_node_ids_bitmask[dst_symbolic * m_bitmask_size + iteration]);
        ++iteration;
      }
    } else {
      uint64_t src = real_node_ids[src_symbolic];
      uint64_t dst = real_node_ids[dst_symbolic];
      src_str = std::to_string(src);
      dst_str = std::to_string(dst);
    }

    ofs << "  " << src_str << " -> " << dst_str << " [label=\""
        << std::to_string(attr) << "\"]" << ";\n";
  }

  ofs << "}\n";
  ofs.close();

  auto &os = ArgumentParser::get_instance().get_output_stream();

  // Now compare the two files line-by-line
  std::ifstream file1(origin_filename);
  std::ifstream file2(modified_path);

  if (!file1 || !file2) {
    os << "[ERROR] Problem in opening files for comparison (the files are: "
       << origin_filename << " and " << modified_path << ")." << std::endl;
    return false;
  }

  std::string line1, line2;
  size_t line_number = 1;

  while (true) {
    bool file1_eof = !std::getline(file1, line1);
    bool file2_eof = !std::getline(file2, line2);

    if (file1_eof && file2_eof) {
      // Both files ended, all lines matched
      return true;
    }

    if (file1_eof != file2_eof) {
      os << "[ERROR] Files differ in length at line " << line_number << "."
         << std::endl;
      return false;
    }

    if (line1 != line2) {
      os << "[ERROR] Difference found at line " << line_number << ":\n";
      os << "File1 (" << origin_filename << "): " << line1 << "\n";
      os << "File2 (" << modified_path << "): " << line2 << "\n";

      // Optional: show column of first mismatch
      size_t col = 0;
      while (col < line1.size() && col < line2.size() &&
             line1[col] == line2[col]) {
        ++col;
      }
      os << "First difference at column " << col + 1 << std::endl;

      return false;
    }

    ++line_number;
  }
}
