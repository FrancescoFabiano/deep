#include "ExitHandler.h"
#include "FringeEvalRL.h"
#include <fstream>
#include <regex>

// --- Singleton instance initialization ---
// template <StateRepresentation StateRepr>
// FringeEvalRL<StateRepr> *FringeEvalRL<StateRepr>::instance = nullptr;

template <StateRepresentation StateRepr>
FringeEvalRL<StateRepr> &FringeEvalRL<StateRepr>::get_instance() {
  thread_local FringeEvalRL<StateRepr> instance;
  return instance;
}

/*template <StateRepresentation StateRepr>
FringeEvalRL<StateRepr> &FringeEvalRL<StateRepr>::get_instance() {
  if (!instance) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::FringeEvalInstanceError,
        "FringeEvalRL instance not created. Call create_instance() first.");
    std::exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));
  }
  return *instance;
}*/

// template <StateRepresentation StateRepr>
// void FringeEvalRL<StateRepr>::create_instance() {
//   if (!instance) {
//     instance = new FringeEvalRL();
//   }
// }

template <StateRepresentation StateRepr>
FringeEvalRL<StateRepr>::FringeEvalRL() {
  // Create GNN if not created yet
  // GraphNN<StateRepr>::create_instance();
  initialize_onnx_model();
}

template <StateRepresentation StateRepr>
void FringeEvalRL<StateRepr>::initialize_onnx_model() {
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

    // The exported RL ONNX currently emits fixed-size logits [F] where F is
    // the export-time frontier size (normally 32). Keep runtime config aligned.
    if (!m_output_names.empty()) {
      auto output_type_info = m_session->GetOutputTypeInfo(0);

      if (output_type_info.GetONNXType() != ONNXType::ONNX_TYPE_TENSOR) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::FringeEvalModelLoadError,
            "ONNX output 0 is not a tensor.");
      }

      auto output_info = output_type_info.GetTensorTypeAndShapeInfo();
      auto output_shape = output_info.GetShape();

      if (output_shape.empty()) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::FringeEvalModelLoadError,
            "ONNX output 0 has empty shape.");
      }

      // Use the last dim if your model exports [1, F], or the only dim if it
      // exports [F].
      int64_t frontier_dim = output_shape.back();
      if (frontier_dim <= 0) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::FringeEvalModelLoadError,
            "ONNX output 0 has invalid/dynamic frontier dimension.");
      }

      const auto model_frontier_size = static_cast<size_t>(frontier_dim);
      const auto configured_frontier_size = static_cast<size_t>(
          ArgumentParser::get_instance().get_RL_fringe_size());

      if (model_frontier_size != configured_frontier_size) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::FringeEvalModelLoadError,
            "RL fringe size mismatch: ONNX logits length is " +
                std::to_string(model_frontier_size) +
                " but --RL_fringe_size is " +
                std::to_string(configured_frontier_size) + ".");
      }
    }

    m_model_loaded = true;
  } catch (const std::exception &e) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::FringeEvalModelLoadError,
        std::string("Failed to create ONNX model: ") + e.what());
  }

  if (ArgumentParser::get_instance().get_verbose()) {
    auto &os = ArgumentParser::get_instance().get_output_stream()
               << "[ONNX] Model loaded: " << m_model_path << std::endl;

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
FringeTensor FringeEvalRL<StateRepr>::fringe_to_tensor_minimal(
    const std::vector<State<StateRepr>> &states) {
  switch (ArgumentParser::get_instance().get_dataset_type()) {
  case DatasetType::HASHED:
    break;
  case DatasetType::BITMASK:
  case DatasetType::MAPPED:
    // Here pay attention to scale the id of the nodes by the number of previous
    // inserted nodes
  default: {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::FringeNotImplementedError,
        "This datatypes for FringeTensor have not been implemented yet");
  }
  }

  FringeTensor fringe_tensor_ret;

  auto node_offset = 0;
  auto state_number = 0;
#ifdef DEBUG
  if (static_cast<size_t>(ArgumentParser::get_instance().get_RL_fringe_size()) <
          states.size() ||
      states.size() > fringe_tensor_ret.active_states.size()) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::FringeEvalInstanceError,
        "The number of states in the fringe exceeds the maximum allowed size "
        "for RL evaluation. Please check the configuration.");
  }
#endif

  for (auto state : states) {
    const auto state_tensor = state.get_tensor_representation();

    const auto state_real_nodes_ids = state_tensor.real_node_ids;
    const auto number_of_nodes = state_real_nodes_ids.size();
    fringe_tensor_ret.real_node_ids.insert(
        fringe_tensor_ret.real_node_ids.end(), state_real_nodes_ids.begin(),
        state_real_nodes_ids.end());
    fringe_tensor_ret.membership.insert(fringe_tensor_ret.membership.end(),
                                        static_cast<int64_t>(number_of_nodes),
                                        static_cast<int64_t>(state_number));

    const auto state_edges_src = state_tensor.edge_src;
    const auto state_edges_dst = state_tensor.edge_dst;
    const auto state_edges_attrs = state_tensor.edge_attrs;

    for (size_t e = 0; e < state_edges_src.size(); ++e) {
      fringe_tensor_ret.edge_src.push_back(state_edges_src[e] + node_offset);
      fringe_tensor_ret.edge_dst.push_back(state_edges_dst[e] + node_offset);
      fringe_tensor_ret.edge_attrs.push_back(state_edges_attrs[e]);
    }

    fringe_tensor_ret.active_states[state_number] = 1;

    node_offset += number_of_nodes;
    state_number += 1;
  }

  fringe_tensor_ret.candidate_batch.resize(state_number, 0);

  return fringe_tensor_ret;
}

template <StateRepresentation StateRepr>
std::vector<float> FringeEvalRL<StateRepr>::get_score(
    const std::vector<State<StateRepr>> &states,
    std::vector<float> *raw_scores) {
  const auto fringe_tensor = fringe_to_tensor_minimal(states);

  if (!m_model_loaded) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::FringeEvalInstanceError,
        "[ONNX] Model not loaded before inference.");
  }

  auto &session = *m_session;
  const auto &memory_info = *m_memory_info;

  const size_t num_edges = fringe_tensor.edge_src.size();
  // size_t num_nodes = fringe_tensor.real_node_ids.size();

  // Construct real_node_ids tensor: shape [num_nodes, 1]
  std::vector<int64_t> node_ids(fringe_tensor.real_node_ids.begin(),
                                fringe_tensor.real_node_ids.end());
  const std::array<int64_t, 1> node_ids_shape{
      static_cast<int64_t>(node_ids.size())};
  Ort::Value node_ids_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, node_ids.data(), node_ids.size(), node_ids_shape.data(),
      node_ids_shape.size());

  // Construct edge_index tensor: shape [2, num_edges]
  std::vector<int64_t> edge_index_data(2 * num_edges);
  for (size_t i = 0; i < num_edges; ++i) {
    edge_index_data[i] = (fringe_tensor.edge_src[i]); // First row: edge_src
    edge_index_data[num_edges + i] =
        (fringe_tensor.edge_dst[i]); // Second row: edge_dst
  }

  const std::array<int64_t, 2> edge_index_shape{
      2, static_cast<int64_t>(num_edges)};

  Ort::Value edge_index_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, edge_index_data.data(), edge_index_data.size(),
      edge_index_shape.data(), edge_index_shape.size());

  // Construct edge_attr tensor: shape [num_edges, 1]
  std::vector<int64_t> edge_attrs(fringe_tensor.edge_attrs.begin(),
                                  fringe_tensor.edge_attrs.end());
  const std::array<int64_t, 1> edge_attr_shape{static_cast<int64_t>(num_edges)};
  Ort::Value edge_attr_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, edge_attrs.data(), edge_attrs.size(), edge_attr_shape.data(),
      edge_attr_shape.size());

  // Construct membership tensor: shape [num_edges, 1]
  std::vector<int64_t> membership(fringe_tensor.membership.begin(),
                                  fringe_tensor.membership.end());
  const std::array<int64_t, 1> membership_shape{
      static_cast<int64_t>(membership.size())};
  Ort::Value membership_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, membership.data(), membership.size(),
      membership_shape.data(), membership_shape.size());

  // Construct active_states tensor: shape [num_edges]
  std::vector<uint8_t> active_states(fringe_tensor.active_states.begin(),
                                     fringe_tensor.active_states.end());
  const std::array<int64_t, 1> active_states_shape{
      static_cast<int64_t>(active_states.size())};
  Ort::Value active_states_tensor = Ort::Value::CreateTensor<uint8_t>(
      memory_info, active_states.data(), active_states.size(),
      active_states_shape.data(), active_states_shape.size());

  /*// Construct state_batch tensor
  std::vector<int64_t> state_batch_data = fringe_tensor.candidate_batch;
  const std::array<int64_t, 1> state_batch_shape{
      static_cast<int64_t>(state_batch_data.size())
  };
  Ort::Value state_batch_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, state_batch_data.data(), state_batch_data.size(),
      state_batch_shape.data(), state_batch_shape.size());*/

  // Prepare input tensors
  std::vector<Ort::Value> input_tensors;

  input_tensors.emplace_back(std::move(node_ids_tensor));
  input_tensors.emplace_back(std::move(edge_index_tensor));
  input_tensors.emplace_back(std::move(edge_attr_tensor));
  input_tensors.emplace_back(std::move(membership_tensor));
  // input_tensors.emplace_back(std::move(state_batch_tensor));

  if (ArgumentParser::get_instance().get_dataset_separated()) {
    if (!m_goal_tensors_computed) {
      const auto goal_tensor =
          GraphNN<StateRepr>::get_instance().get_goal_tensor();

      m_real_node_ids_goal_data = goal_tensor.real_node_ids;

      const size_t num_goal_edges = goal_tensor.edge_src.size();
      m_edge_index_goal_data.resize(2 * num_goal_edges);
      for (size_t i = 0; i < num_goal_edges; ++i) {
        m_edge_index_goal_data[i] =
            static_cast<int64_t>(goal_tensor.edge_src[i]);
        m_edge_index_goal_data[num_goal_edges + i] =
            static_cast<int64_t>(goal_tensor.edge_dst[i]);
      }

      m_edge_attrs_goal_data = goal_tensor.edge_attrs;
      m_state_batch_goal_data.assign(goal_tensor.real_node_ids.size(), 0);

      m_goal_tensors_computed = true;
    }

    // I can only move the tensor and not copy them, this is needed

    std::vector<int64_t> goal_real_node_ids_shape{
        static_cast<int64_t>(m_real_node_ids_goal_data.size())};

    std::vector<int64_t> goal_edge_index_shape{
        2, static_cast<int64_t>(m_edge_index_goal_data.size() / 2)};

    std::vector<int64_t> goal_edge_attr_shape{
        static_cast<int64_t>(m_edge_attrs_goal_data.size())};

    std::vector<int64_t> goal_state_batch_shape{
        static_cast<int64_t>(m_state_batch_goal_data.size())};

    Ort::Value goal_real_node_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        *m_memory_info, m_real_node_ids_goal_data.data(),
        m_real_node_ids_goal_data.size(), goal_real_node_ids_shape.data(),
        goal_real_node_ids_shape.size());

    Ort::Value goal_edge_index_tensor = Ort::Value::CreateTensor<int64_t>(
        *m_memory_info, m_edge_index_goal_data.data(),
        m_edge_index_goal_data.size(), goal_edge_index_shape.data(),
        goal_edge_index_shape.size());

    Ort::Value goal_edge_attr_tensor = Ort::Value::CreateTensor<int64_t>(
        *m_memory_info, m_edge_attrs_goal_data.data(),
        m_edge_attrs_goal_data.size(), goal_edge_attr_shape.data(),
        goal_edge_attr_shape.size());

    Ort::Value goal_state_batch_tensor = Ort::Value::CreateTensor<int64_t>(
        *m_memory_info, m_state_batch_goal_data.data(),
        m_state_batch_goal_data.size(), goal_state_batch_shape.data(),
        goal_state_batch_shape.size());

    input_tensors.emplace_back(std::move(goal_real_node_ids_tensor));
    input_tensors.emplace_back(std::move(goal_edge_index_tensor));
    input_tensors.emplace_back(std::move(goal_edge_attr_tensor));
    input_tensors.emplace_back(std::move(goal_state_batch_tensor));
  }

  // Keep mask as the last input to match ONNX export order:
  // node/edge/membership, optional goal_*, then mask.
  input_tensors.emplace_back(std::move(active_states_tensor));

  if (input_tensors.size() != m_input_names.size()) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::FringeEvalModelLoadError,
        "ONNX input count mismatch: model expects " +
            std::to_string(m_input_names.size()) +
            " input tensors but C++ "
            "prepared " +
            std::to_string(input_tensors.size()) + ".");
  }

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
  // Get the result
  const float *output_data = output_tensors[0].template GetTensorData<float>();
  const auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();

  if (raw_scores != nullptr) {
    raw_scores->assign(output_data, output_data + states.size());
  }

  return rankScores(output_data, states.size());
}

template <StateRepresentation StateRepr>
std::vector<float> FringeEvalRL<StateRepr>::rankScores(const float *scores,
                                                       const size_t n) const {
  // Pair each score with its original index
  std::vector<std::pair<float, size_t>> paired;
  paired.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    paired.emplace_back(scores[i], i);
  }

  // Sort by score descending (higher score = better rank)
  std::sort(paired.begin(), paired.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  // Create result array
  std::vector<float> ranks(n);

  // Assign ranks
  for (size_t i = 0; i < n; ++i) {
    ranks[paired[i].second] = static_cast<float>(i);
  }

  /*#ifdef DEBUG
    std::cout << "[";
    for (size_t i = 0; i < n; ++i) {
      std::cout << scores[i] << ", " << ranks[i];
      if (i + 1 < n)
        std::cout << " -- ";
    }
    std::cout << "]\n";
  #endif*/
  return ranks;
}
