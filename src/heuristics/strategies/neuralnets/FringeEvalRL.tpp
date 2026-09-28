#include "ExitHandler.h"
#include "FringeEvalRL.h"
#include <algorithm>
#include <cmath>
#include <limits>

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
PackedGraph FringeEvalRL<StateRepr>::fringe_to_tensor_minimal(
    std::vector<State<StateRepr>> &states) {
#ifdef DEBUG
  if (static_cast<size_t>(ArgumentParser::get_instance().get_RL_fringe_size()) <
      states.size()) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::FringeEvalInstanceError,
        "The number of states in the fringe exceeds the maximum allowed size "
        "for RL evaluation. Please check the configuration.");
  }
#endif

  // The tensor is cached in each state, so read it in place (a copy of the
  // state would recompute it at every evaluation).
  std::vector<const GraphTensor *> graphs;
  graphs.reserve(states.size());
  for (auto &state : states) {
    graphs.push_back(&state.get_tensor_representation());
  }

  return OnnxInputs::pack(
      graphs,
      ArgumentParser::get_instance().get_dataset_type() == DatasetType::BITMASK,
      GraphNN<StateRepr>::get_instance().get_bitmask_size());
}

template <StateRepresentation StateRepr>
std::vector<float>
FringeEvalRL<StateRepr>::get_score(std::vector<State<StateRepr>> &states,
                                   std::vector<float> *raw_scores) {
  if (!m_model_loaded) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::FringeEvalInstanceError,
        "[ONNX] Model not loaded before inference.");
  }

  // Inputs, in the ONNX export order: nodes, edge_index, edge_attr [E],
  // membership, pointed_ids, optional goal_* (separated), then the mask.
  // Nodes are uint8 [N, bits] under BITMASK, int64 [N] otherwise; the goal
  // (separated) always uses int64 ids.
  const bool is_bitmask =
      ArgumentParser::get_instance().get_dataset_type() == DatasetType::BITMASK;
  const auto fringe_packed = fringe_to_tensor_minimal(states);
  OnnxInputs inputs(*m_memory_info);
  inputs.add_graph(fringe_packed, is_bitmask,
                   GraphNN<StateRepr>::get_instance().get_bitmask_size(), false,
                   true);

  if (ArgumentParser::get_instance().get_dataset_separated()) {
    inputs.add_graph(GraphNN<StateRepr>::get_instance().get_goal_packed(),
                     false, 0, false, false);
  }

  // Active-state mask: 1 for each occupied fringe slot.
  std::vector<uint8_t> active_states(
      ArgumentParser::get_instance().get_RL_fringe_size(), 0);
  std::fill_n(active_states.begin(),
              std::min(states.size(), active_states.size()), 1);
  inputs.add_owned(std::move(active_states));

  const auto outputs =
      inputs.run(*m_session, m_input_names, m_output_names,
                 ExitHandler::ExitCode::FringeEvalModelLoadError);

  const float *output_data = outputs[0].template GetTensorData<float>();

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
    float score = scores[i];
    // NaN/inf would break the sort ordering: demote them to the worst score
    if (!std::isfinite(score)) {
      ArgumentParser::get_instance().get_output_stream()
          << "[WARNING] FringeEvalRL model returned a non-finite score ("
          << score << ") at fringe slot " << i
          << ", treating it as the worst score." << std::endl;
      score = std::numeric_limits<float>::lowest();
    }
    paired.emplace_back(score, i);
  }

  // Sort by score descending (higher score = better rank); ties broken by
  // original index so the ranking is reproducible.
  std::sort(paired.begin(), paired.end(), [](const auto &a, const auto &b) {
    return a.first != b.first ? a.first > b.first : a.second < b.second;
  });

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
