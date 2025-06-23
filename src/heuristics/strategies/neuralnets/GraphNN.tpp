#include "ExitHandler.h"
#include "GraphNN.h"
#include "TrainingDataset.h"
#include <fstream>

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
  m_goal_file_path =
      TrainingDataset<StateRepr>::get_instance().get_goal_file_path();
}

/**
 * \brief Get the score for a given state using the neural network heuristic.
 * \tparam StateRepr The state representation type.
 * \param state The state to evaluate.
 * \return The heuristic score for the state.
 */
template <StateRepresentation StateRepr>
[[nodiscard]] short
GraphNN<StateRepr>::get_score(const State<StateRepr> &state) {
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
GraphTensor
GraphNN<StateRepr>::kripke_to_tensor_minimal(const KripkeState &kstate) {

  /*
   * \todo 1. Add merged and hash distinction, once decision is taken, keep only
   * one in release (which will be most likely hashed and merged) \todo 2. Make
   * sure no negative indexes exist. Start form zero and move forward. Maybe
   * start from 10 with the meaningful ones
   */

  const auto &training_dataset = TrainingDataset<KripkeState>::get_instance();

  std::unordered_map<KripkeWorldId, int64_t> node_index_map;
  std::vector<size_t> node_ids;
  int64_t next_index = 0;

  // Collect nodes
  for (const auto &world : kstate.get_worlds()) {
    const auto hash = world.get_id();
    if (!node_index_map.contains(hash)) {
      node_index_map[hash] = next_index++;
      node_ids.push_back(hash);
    }
  }

  // Collect edges
  std::vector<int64_t> edge_src;
  std::vector<int64_t> edge_dst;
  std::vector<int64_t> edge_labels;

  for (const auto &[from_pw, from_map] : kstate.get_beliefs()) {
    size_t from_hash = from_pw.get_id();
    int64_t from_idx = node_index_map.at(from_hash);

    for (const auto &[agent, to_set] : from_map) {
      int64_t label = static_cast<int64_t>(
          training_dataset.get_unique_a_id_from_map(agent));

      for (const auto &to_pw : to_set) {
        size_t to_hash = to_pw.get_id();
        int64_t to_idx = node_index_map.at(to_hash);

        edge_src.push_back(from_idx);
        edge_dst.push_back(to_idx);
        edge_labels.push_back(label);
      }
    }
  }

  // Build tensors
  torch::Tensor edge_index =
      torch::stack({torch::from_blob(edge_src.data(),
                                     {static_cast<int64_t>(edge_src.size())},
                                     torch::kInt64),
                    torch::from_blob(edge_dst.data(),
                                     {static_cast<int64_t>(edge_dst.size())},
                                     torch::kInt64)})
          .clone(); // clone for safety

  torch::Tensor edge_attr =
      torch::from_blob(edge_labels.data(),
                       {static_cast<int64_t>(edge_labels.size()), 1},
                       torch::kInt64)
          .clone();

  return GraphTensor{
      .edge_index = edge_index, .edge_attr = edge_attr, .node_ids = node_ids};
}