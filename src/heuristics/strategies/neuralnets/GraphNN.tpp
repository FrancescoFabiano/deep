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
    m_session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_CUDA
    try {
      OrtCUDAProviderOptions cuda_options;
      m_session_options.AppendExecutionProvider_CUDA(cuda_options);
      if (ArgumentParser::get_instance().get_verbose()) {
        ArgumentParser::get_instance().get_output_stream() << "[ONNX] CUDA execution provider enabled via USE_CUDA." << std::endl;
      }
    } catch (const Ort::Exception &e) {
      ArgumentParser::get_instance().get_output_stream() << "[WARNING][ONNX] Failed to enable CUDA, defaulting to CPU: " << e.what() << std::endl;
    }
#else
    if (ArgumentParser::get_instance().get_verbose()) {
      ArgumentParser::get_instance().get_output_stream() << "[ONNX] Compiled without CUDA (USE_CUDA not defined), using CPU." << std::endl;
    }
#endif

    m_session = std::make_unique<Ort::Session>(m_env, model_path.c_str(), m_session_options);
    m_allocator = std::make_unique<Ort::AllocatorWithDefaultOptions>();
    m_memory_info = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));

    m_input_names.push_back(m_session->GetInputName(0, *m_allocator));
    m_output_names.push_back(m_session->GetOutputName(0, *m_allocator));

    m_model_loaded = true;
  } catch (const std::exception &e) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::GNNModelLoadError,
        std::string("Failed to load ONNX model: ") + e.what());
    std::exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));

    if (ArgumentParser::get_instance().get_verbose()) {
      ArgumentParser::get_instance().get_output_stream() << "[ONNX] Model loaded: " << m_model_path << std::endl;
    }}
}


template <StateRepresentation StateRepr>
[[nodiscard]] short
GraphNN<StateRepr>::get_score(const State<StateRepr> &state) {
  const auto state_tensor = state_to_tensor_minimal(state.get_representation());

#ifdef DEBUG
  if (ArgumentParser::get_instance().get_verbose()) {
    if (!check_tensor_against_dot(state_tensor, state)) {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::GNNTensorTranslationError,
          "[ERROR] Error while comparing the state/goal in " +
              m_checking_file_path + "/" + m_goal_file_path +
              ". Its tensor representation generated a different dot file.");
    } else {
      ArgumentParser::get_instance().get_output_stream()
          << "[DEBUG] State and goal represented as dot and as tensor match:)"
          << std::endl;
    }
  }
#endif

  /// load onnx

  /// give state_tensor as input (and goal if is not merged)
  /// get result from onn + input

  return 1;
}


template <StateRepresentation StateRepr>
float GraphNN<StateRepr>::run_inference(const GraphTensor &tensor) {
  if (!m_model_loaded) {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::GNNInstanceError,
                                   "[ONNX] Model not loaded before inference.");
  }

  auto &session = *m_session;
  auto &allocator = *m_allocator;
  auto &memory_info = *m_memory_info;

  const size_t num_edges = tensor.edge_src.size();
  const size_t num_nodes = tensor.real_node_ids.size();

  // Construct edge_index tensor: shape [2, num_edges]
  std::vector<int64_t> edge_index_data;
  edge_index_data.reserve(2 * num_edges);
  for (size_t i = 0; i < num_edges; ++i) {
    edge_index_data.push_back(tensor.edge_src[i]);
    edge_index_data.push_back(tensor.edge_dst[i]);
  }

  std::array<int64_t, 2> edge_index_shape{2, static_cast<int64_t>(num_edges)};
  Ort::Value edge_index_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, edge_index_data.data(), edge_index_data.size(),
      edge_index_shape.data(), edge_index_shape.size());

  // Construct edge_attr tensor: shape [num_edges, 1]
  std::array<int64_t, 2> edge_attr_shape{static_cast<int64_t>(num_edges), 1};
  Ort::Value edge_attr_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, const_cast<int64_t *>(tensor.edge_attrs.data()), tensor.edge_attrs.size(),
      edge_attr_shape.data(), edge_attr_shape.size());

  // Construct real_node_ids tensor: shape [num_nodes, 1]
  std::array<int64_t, 2> node_ids_shape{static_cast<int64_t>(num_nodes), 1};
  Ort::Value real_node_ids_tensor = Ort::Value::CreateTensor<uint64_t>(
      memory_info, const_cast<uint64_t *>(tensor.real_node_ids.data()), tensor.real_node_ids.size(),
      node_ids_shape.data(), node_ids_shape.size());

  // Prepare input names (in order as expected by model)
  std::vector<Ort::Value> input_tensors;
  std::vector<const char *> input_names = {
      m_input_names[0], // assume order: edge_index
      m_input_names[1], // edge_attr
      m_input_names[2]  // real_node_ids
  };

  input_tensors.emplace_back(std::move(edge_index_tensor));
  input_tensors.emplace_back(std::move(edge_attr_tensor));
  input_tensors.emplace_back(std::move(real_node_ids_tensor));

  // Run the model
  auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                    input_names.data(), input_tensors.data(),
                                    input_tensors.size(),
                                    m_output_names.data(), m_output_names.size());

  // Get the result (assuming scalar output)
  float *output_data = output_tensors[0].GetTensorMutableData<float>();
  float score = output_data[0];

  return score;
}


template <StateRepresentation StateRepr>
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
      ArgumentParser::get_instance().get_dataset_merged());

  // Run the external Python script for NN inference
  /// todo Replace this with C++ call that opens the model and runs the
  /// inference
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
}

template <StateRepresentation StateRepr>
size_t GraphNN<StateRepr>::get_symbolic_id(const size_t node) {
  if (!m_node_to_symbolic.contains(node)) {
    m_node_to_symbolic[node] = m_symbolic_id++;
    m_real_node_ids.push_back(node);
  }
  return m_node_to_symbolic[node];
}

template <StateRepresentation StateRepr>
void GraphNN<StateRepr>::add_edge(const int64_t src, const int64_t dst,
                                  const int64_t label) {
  m_edge_src.push_back(get_symbolic_id(src));
  m_edge_dst.push_back(get_symbolic_id(dst));
  m_edge_labels.push_back(label);
}

template <StateRepresentation StateRepr>
void GraphNN<StateRepr>::populate_with_goal() {
  auto const goal_graph_string =
      TrainingDataset<StateRepr>::get_instance().get_goal_string();
  const std::regex pattern(R"((\d+)\s*->\s*(\d+)\s*\[label=\"(\d+)\"\];)");
  const std::sregex_iterator begin(goal_graph_string.begin(),
                                   goal_graph_string.end(), pattern);
  const std::sregex_iterator end;

  if (ArgumentParser::get_instance().get_dataset_merged()) {
    constexpr auto epsilon_id =
        TrainingDataset<KripkeState>::get_epsilon_node_id_int();
    constexpr auto goal_parent_id =
        TrainingDataset<KripkeState>::get_goal_parent_id_int();

    add_edge(epsilon_id, goal_parent_id,
             TrainingDataset<KripkeState>::get_to_goal_edge_id_int());
  }

  for (auto it = begin; it != end; ++it) {
    const size_t src = std::stoul((*it)[1]);
    const size_t dst = std::stoul((*it)[2]);
    const size_t label = std::stoul((*it)[3]);

    add_edge(src, dst, label);
  }

  if (!ArgumentParser::get_instance().get_dataset_merged()) {
    fill_graph_tensor(m_goal_graph_tensor);

    m_edge_dst.clear();
    m_edge_src.clear();
    m_edge_labels.clear();
    m_real_node_ids.clear();
    m_node_to_symbolic.clear();
    m_symbolic_id = 0;
  }

  m_edges_initial_size = m_edge_labels.size();
  m_node_ids_initial_size = m_real_node_ids.size();
  m_starting_symbolic_id = m_symbolic_id;
}

template <StateRepresentation StateRepr>
GraphTensor
GraphNN<StateRepr>::state_to_tensor_minimal(const KripkeState &kstate) {
#ifdef DEBUG
  if (ArgumentParser::get_instance().get_dataset_mapped()) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::GNNMappedNotSupportedError,
        "We do not support the mapped version with the inference in C++ "
        "(useless, hashing should be better).");
  }
#endif

  const auto m_node_to_symbolic_original = m_node_to_symbolic;

  if (ArgumentParser::get_instance().get_dataset_merged()) {
    const auto state_parent_id = kstate.get_pointed().get_id();

    add_edge(TrainingDataset<KripkeState>::get_epsilon_node_id_int(),
             state_parent_id,
             TrainingDataset<KripkeState>::get_to_state_edge_id_int());
  }

  const auto &training_dataset = TrainingDataset<KripkeState>::get_instance();

  for (const auto &[from_pw, from_map] : kstate.get_beliefs()) {
    const size_t src = from_pw.get_id();

    for (const auto &[agent, to_set] : from_map) {
      const int64_t label = static_cast<int64_t>(
          training_dataset.get_unique_a_id_from_map(agent));

      for (const auto &to_pw : to_set) {
        const size_t dst = to_pw.get_id();

        add_edge(src, dst, label);
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
  m_node_to_symbolic = m_node_to_symbolic_original;
  m_symbolic_id = m_starting_symbolic_id;

  return ret;
}

template <StateRepresentation StateRepr>
void GraphNN<StateRepr>::fill_graph_tensor(GraphTensor &tensor) {
  tensor.edge_src = m_edge_src;
  tensor.edge_dst = m_edge_dst;
  tensor.edge_attrs = m_edge_labels;
  tensor.real_node_ids = m_real_node_ids;
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
  state.print_dataset_format(
      ofs_orig, !ArgumentParser::get_instance().get_dataset_mapped(),
      ArgumentParser::get_instance().get_dataset_merged());
  ofs_orig.close();

  // Prepare modified path string
  bool ret =
      write_and_compare_tensor_to_dot(m_checking_file_path, state_tensor);
  if (!ArgumentParser::get_instance().get_dataset_mapped() && ret) {
    ret = ret && write_and_compare_tensor_to_dot(m_goal_file_path,
                                                 m_goal_graph_tensor);
  }

  return ret;
}

template <StateRepresentation StateRepr>
bool GraphNN<StateRepr>::write_and_compare_tensor_to_dot(
    const std::string &origin_filename, const GraphTensor &state_tensor) {
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
  auto edge_ids = state_tensor.edge_ids;
  auto edge_attrs = state_tensor.edge_attrs;

  int64_t num_edges = edge_ids.size(1);

  // Print edges with optional edge_attrs label
  for (int64_t e = 0; e < num_edges; ++e) {
    int64_t src_symbolic = edge_ids[0][e].item<int64_t>();
    int64_t dst_symbolic = edge_ids[1][e].item<int64_t>();

    auto accessor = state_tensor.real_node_ids.accessor<uint64_t, 2>();

    uint64_t src = accessor[src_symbolic][0];
    uint64_t dst = accessor[dst_symbolic][0];

    // Get edge attribute if exists
    std::string label_str = "";
    if (edge_attrs.defined() && edge_attrs.numel() > e) {
      int64_t attr =
          edge_attrs[e][0].item<int64_t>(); // assuming shape [num_edges,1]
      label_str = " [label=\"" + std::to_string(attr) + "\"]";
    }

    ofs << "  " << src << " -> " << dst << label_str << ";\n";
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

  // Jut To please the compiler
  exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));
}
