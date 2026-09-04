#include "ArgumentParser.h"
#include "Configuration.h"
#include "Domain.h"
#include "ExitHandler.h"
#include "HelperPrint.h"
#include "HeuristicsManager.h"
#include "TrainingDataset.h"
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip> // Make sure this is included at the top of your file
#include <memory>
#include <map>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

/*
// --- Singleton implementation ---
template <StateRepresentation StateRepr>
TrainingDataset<StateRepr> &TrainingDataset<StateRepr>::get_instance() {
  if (!instance) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::NNInstanceError,
        "GNN instance not created. Call create_instance() first.");
    // Just to please the compiler
    std::exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));
  }
  return *instance;
}

template <StateRepresentation StateRepr>
void TrainingDataset<StateRepr>::create_instance() {
  if (!instance) {
    instance = new TrainingDataset();
  }
}

*/

template <StateRepresentation StateRepr>
TrainingDataset<StateRepr> &TrainingDataset<StateRepr>::get_instance() {
  static TrainingDataset<StateRepr> instance;
  return instance;
}

template <StateRepresentation StateRepr>
const std::string &TrainingDataset<StateRepr>::get_folder() const {
  return m_folder;
}

template <StateRepresentation StateRepr>
const std::string &TrainingDataset<StateRepr>::get_to_goal_edge_id_string() {
  return m_to_goal_edge_id;
}

template <StateRepresentation StateRepr>
const std::string &TrainingDataset<StateRepr>::get_to_state_edge_id_string() {
  return m_to_state_edge_id;
}

template <StateRepresentation StateRepr>
const std::string &TrainingDataset<StateRepr>::get_epsilon_node_id_string() {
  return m_epsilon_node_id;
}

template <StateRepresentation StateRepr>
const std::string &TrainingDataset<StateRepr>::get_goal_parent_id_string() {
  return m_goal_parent_id;
}

template <StateRepresentation StateRepr>
constexpr int TrainingDataset<StateRepr>::get_to_goal_edge_id_int() {
  return m_to_goal_edge_id_int;
}

template <StateRepresentation StateRepr>
constexpr int TrainingDataset<StateRepr>::get_to_state_edge_id_int() {
  return m_to_state_edge_id_int;
}

template <StateRepresentation StateRepr>
constexpr int TrainingDataset<StateRepr>::get_epsilon_node_id_int() {
  return m_epsilon_node_id_int;
}

template <StateRepresentation StateRepr>
constexpr int TrainingDataset<StateRepr>::get_goal_parent_id_int() {
  return m_goal_parent_id_int;
}

template <StateRepresentation StateRepr>
constexpr const std::string &
TrainingDataset<StateRepr>::get_goal_string() const {
  return m_goal_string;
}

template <StateRepresentation StateRepr>
constexpr const std::string &
TrainingDataset<StateRepr>::get_goal_forced_string() const {
  return m_goal_forced_string;
}

template <StateRepresentation StateRepr>
int TrainingDataset<StateRepr>::get_shift_state_ids() const {
  return m_shift_state_ids;
}

template <StateRepresentation StateRepr>
std::string
TrainingDataset<StateRepr>::make_unique_folder(const std::string &base_path,
                                               const std::string &domain_name) {
  const std::filesystem::path folder_path =
      std::filesystem::path(base_path) / domain_name;
  std::string unique_path = folder_path.string();

  int counter = 1;
  while (std::filesystem::exists(unique_path)) {
    std::ostringstream oss;
    oss << folder_path.string() << "_" << counter;
    unique_path = oss.str();
    ++counter;
  }

  return unique_path + "/"; // ensure trailing slash if you want it
}

template <StateRepresentation StateRepr>
std::string TrainingDataset<StateRepr>::create_complete_path() const {
  std::string dataset_type;

  switch (ArgumentParser::get_instance().get_dataset_type()) {
  case DatasetType::HASHED: {
    dataset_type = std::string(OutputPaths::DATASET_NN_DATASET_HASHED);
    break;
  }
  case DatasetType::MAPPED: {
    dataset_type = std::string(OutputPaths::DATASET_NN_DATASET_MAPPED);
    break;
  }
  case DatasetType::BITMASK: {
    dataset_type = std::string(OutputPaths::DATASET_NN_DATASET_BITMASK);
    break;
  }
  default: {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::ArgParseError,
                                   "Invalid Dataset Type specified");
  }
  }

  if (ArgumentParser::get_instance().get_dataset_separated()) {
    return m_training_raw_files_folder + dataset_type + "_" +
           std::string(OutputPaths::DATASET_NN_DATASET_SEPARATED) + "/";
  } else {
    return m_training_raw_files_folder + dataset_type + "_" +
           std::string(OutputPaths::DATASET_NN_DATASET_MERGED) + "/";
  }
}

template <StateRepresentation StateRepr>
std::string
TrainingDataset<StateRepr>::to_binary_string(const bool force_non_binary_ids,
                                             const std::string &str_value) {

  if (ArgumentParser::get_instance().get_dataset_type() !=
          DatasetType::BITMASK ||
      ArgumentParser::get_instance().get_dataset_separated() ||
      force_non_binary_ids) {
    return str_value;
  }

  int value{};
  try {
    value = std::stoi(str_value);
  } catch (const std::exception &e) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::GNNBitmaskGOALError,
        "Wrong integer conversion for ID in the goal encoding. Exception:" +
            std::string(e.what()));
  }
  return to_binary_string(force_non_binary_ids, value);
}

template <StateRepresentation StateRepr>
std::string
TrainingDataset<StateRepr>::to_binary_string(const bool force_non_binary_ids,
                                             const size_t value) {

  if (ArgumentParser::get_instance().get_dataset_type() !=
          DatasetType::BITMASK ||
      ArgumentParser::get_instance().get_dataset_separated() ||
      force_non_binary_ids) {
    return std::to_string(value);
  }

  // Check if the value fits in the given number of bits
  auto uvalue = static_cast<unsigned int>(value);
  if (uvalue >= (1u << GOAL_ENCODING_BITS)) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::GNNBitmaskGOALError,
        "The number of bits is not enough to encode all the goal information. "
        "Increase GOAL_ENCODING_BITS in define.h, and ensure that all "
        "training data "
        "uses the same padding values. Verify that this value is "
        "consistently passed "
        "to the GNN during training and correctly applied during "
        "inference.");
  }

  auto bit_width = GOAL_ENCODING_BITS;
  if (!ArgumentParser::get_instance().get_dataset_separated()) {
    bit_width += MAX_REPETITION_BITS + MAX_FLUENT_NUMBER;
  }

  std::string binary(bit_width, '0');
  for (size_t i = 0; i < bit_width; ++i) {
    binary[bit_width - 1 - i] = (uvalue & 1) ? '1' : '0';
    uvalue >>= 1;
  }
  return binary;
}

template <StateRepresentation StateRepr>
void TrainingDataset<StateRepr>::update_binary_ids() {
  m_epsilon_node_id = to_binary_string(false, m_epsilon_node_id);
  m_goal_parent_id = to_binary_string(false, m_goal_parent_id);
}

template <StateRepresentation StateRepr>
TrainingDataset<StateRepr>::TrainingDataset() {
  /// \brief Mersenne Twister random number generator, seeded with rd if no seed
  /// provided.
  m_seed = ArgumentParser::get_instance().get_dataset_seed();
  if (m_seed < 0) {
    m_seed = std::random_device{}(); // Use random device if seed is negative
  }
  m_gen.seed(m_seed);

  const std::string domain_name = Domain::get_instance().get_name();

  if (ArgumentParser::get_instance().get_dataset_mode()) {
    std::string generation_name =
        ArgumentParser::get_instance().get_dataset_generation_type_string();

    for (char &c : generation_name) {
      if (!std::isalnum(static_cast<unsigned char>(c))) {
        c = '_';
      }
    }

    // Avoid repeated underscores produced by strings such as "HFS (SUBGOALS)".
    while (generation_name.find("__") != std::string::npos) {
      generation_name.replace(generation_name.find("__"), 2, "_");
    }

    const std::string filename =
        domain_name + "_" + generation_name + "_depth_" +
        std::to_string(ArgumentParser::get_instance().get_dataset_depth()) +
        ".csv";

    m_folder = make_unique_folder(
        OutputPaths::DATASET_TRAINING_FOLDER,
        domain_name + "_" + generation_name);

    m_training_raw_files_folder = m_folder + "RawFiles/";
    m_filepath_csv = m_folder + filename;
    // Use std::filesystem for directory creation (C++17+)
    try {
      std::filesystem::create_directories(m_folder);
      std::filesystem::create_directories(m_training_raw_files_folder);
      std::filesystem::create_directories(create_complete_path());
    } catch (const std::filesystem::filesystem_error &e) {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::NNDirectoryCreationError,
          std::string("Error creating directories: ") + e.what());
    }
  } else {
    m_folder = std::string(OutputPaths::DATASET_INFERENCE_FOLDER) + "/" +
               domain_name + "/";
    try {
      std::filesystem::create_directories(m_folder);
    } catch (const std::filesystem::filesystem_error &e) {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::NNDirectoryCreationError,
          std::string("Error creating directories: ") + e.what());
    }
  }

  m_action_to_id["no-op"] = 0;
  int action_counter = 1;
  for (const auto &act : Domain::get_instance().get_actions()) {
    m_action_to_id[act.get_name()] = action_counter++;
  }

  m_goal_file_path = m_folder + "goal_tree.dot";

  m_shift_state_ids =
      m_to_state_edge_id_int + 1; // This is used to shifts the goals id
  m_shift_state_ids +=
      Domain::get_instance().get_goal_description().size() + 1 + 1;
  // We will also generate the goal edges and shift for them as well. Done in
  // goal generation
  populate_agent_ids(m_shift_state_ids);
  m_shift_state_ids += static_cast<int>(m_agent_to_id.size()) + 1;
  populate_fluent_ids(m_shift_state_ids);
  m_shift_state_ids += static_cast<int>(m_fluent_to_id.size()) + 1;

  if (ArgumentParser::get_instance().get_dataset_type() ==
          DatasetType::BITMASK &&
      !ArgumentParser::get_instance().get_dataset_separated()) {
    const size_t original_shift_id = m_shift_state_ids;
    generate_goal_tree_subgraph(true);
    m_shift_state_ids = original_shift_id;
  }

  update_binary_ids();
  // This stores the goal tree in a string for efficient printing
  generate_goal_tree_subgraph(false);

  if (ArgumentParser::get_instance().get_dataset_separated()) {
    print_goal_tree(); // Only needed if we do not use the goal and state merged
                       // together
  }
}

template <StateRepresentation StateRepr>
bool TrainingDataset<StateRepr>::generate_dataset() {
  std::ofstream result(m_filepath_csv);
  if (!result.is_open()) {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::NNTrainingFileError,
                                   "Error opening file: " + m_filepath_csv);
    std::exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));
  }
  result
      << "File Path,Depth,Distance From Goal,Goal,File Path Predecessor,Action"
      << std::endl;
  result.close();

  return search_space_exploration();
}

template <StateRepresentation StateRepr>
const std::string &TrainingDataset<StateRepr>::get_goal_file_path() const {
  return m_goal_file_path;
}

template <StateRepresentation StateRepr>
void TrainingDataset<StateRepr>::print_goal_tree() const {
  std::ofstream dot_file(m_goal_file_path);
  if (!dot_file.is_open()) {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::NNTrainingFileError,
                                   "Error opening file: " + m_goal_file_path);
  }

  dot_file << "digraph G {\n";

  dot_file << m_goal_string;

  dot_file << "}\n";
  dot_file.close();
}

template <StateRepresentation StateRepr>
void TrainingDataset<StateRepr>::generate_goal_tree_subgraph(
    const bool force_non_binary_ids) {
  std::stringstream string_goals_graph;
  const auto goal_list = Domain::get_instance().get_goal_description();
  size_t goal_counter = m_to_state_edge_id_int + 1;

  size_t next_id = m_shift_state_ids;

  // This is the root node of the root node of goals connected to the parent
  // when exists (which has m_failed_state as label)
  for (const auto &goal : goal_list) {
    generate_goal_subtree(goal, ++goal_counter, next_id, m_goal_parent_id,
                          string_goals_graph, force_non_binary_ids);
  }

  m_shift_state_ids += +1;
  // The final value of shits so that state, when mapped starts from the latest
  // node generated for the goals + 1
  if (force_non_binary_ids) {
    m_goal_forced_string = string_goals_graph.str();
  } else {
    m_goal_string = string_goals_graph.str();
  }
}

template <StateRepresentation StateRepr>
size_t TrainingDataset<StateRepr>::get_id_from_map(
    const std::unordered_map<boost::dynamic_bitset<>, size_t> &id_map,
    const boost::dynamic_bitset<> &key, const std::string &type_name) {
  if (const auto it = id_map.find(key); it != id_map.end()) {
    return it->second;
  }
  ExitHandler::exit_with_message(ExitHandler::ExitCode::NNMappingError,
                                 "Error accessing a key in " + type_name +
                                     " map. Key not found.");
  // Jut to please the compiler
  std::exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));
}

template <StateRepresentation StateRepr>
void TrainingDataset<StateRepr>::populate_ids_from_bitset(
    const std::set<boost::dynamic_bitset<>> &keys_set,
    std::unordered_map<boost::dynamic_bitset<>, size_t> &id_map,
    const size_t start_id) {
  size_t current_id = start_id;
  for (const auto &key : keys_set) {
    id_map[key] = current_id++;
  }
}

template <StateRepresentation StateRepr>
size_t
TrainingDataset<StateRepr>::get_unique_f_id_from_map(const Fluent &fl) const {
  return get_id_from_map(m_fluent_to_id, fl, "Fluent");
}

template <StateRepresentation StateRepr>
size_t
TrainingDataset<StateRepr>::get_unique_a_id_from_map(const Agent &ag) const {
  return get_id_from_map(m_agent_to_id, ag, "Agent");
}

template <StateRepresentation StateRepr>
void TrainingDataset<StateRepr>::populate_fluent_ids(const size_t start_id) {
  populate_ids_from_bitset(Domain::get_instance().get_fluents(), m_fluent_to_id,
                           start_id);
}

template <StateRepresentation StateRepr>
void TrainingDataset<StateRepr>::populate_agent_ids(const size_t start_id) {
  populate_ids_from_bitset(Domain::get_instance().get_agents(), m_agent_to_id,
                           start_id);
}

template <StateRepresentation StateRepr>
void TrainingDataset<StateRepr>::generate_goal_subtree(
    const BeliefFormula &to_print, const size_t goal_counter, size_t &next_id,
    const std::string &parent_node, std::ostream &os,
    const bool force_non_binary_ids) {
  size_t current_node_id = ++next_id;
  std::string node_name;

  switch (to_print.get_formula_type()) {
  case BeliefFormulaType::FLUENT_FORMULA: {
    std::string m_parent_node = parent_node;
    if (to_print.get_fluent_formula().size() > 1) {
      // REMOVE LETTERS node_name = "F_OR" + std::to_string(current_node_id);
      node_name = std::to_string(current_node_id);
      current_node_id = ++next_id;
      // ofs << "  " << node_name << " [label=\"" << current_node_id <<
      // "\"];\n";
      os << "  " << to_binary_string(force_non_binary_ids, parent_node)
         << " -> " << to_binary_string(force_non_binary_ids, node_name)
         << " [label=\"" << goal_counter << "\"];\n";
      m_parent_node = node_name;
    }

    for (const auto &fls_set : to_print.get_fluent_formula()) {
      std::string m_m_parent_node = m_parent_node;

      if (fls_set.size() > 1) {
        // REMOVE LETTERS node_name = "F_AND" + std::to_string(current_node_id);
        node_name = std::to_string(current_node_id);
        current_node_id = ++next_id;
        // ofs << "  " << node_name << " [label=\"" << current_node_id <<
        // "\"];\n";
        os << "  " << to_binary_string(force_non_binary_ids, m_parent_node)
           << " -> " << to_binary_string(force_non_binary_ids, node_name)
           << " [label=\"" << goal_counter << "\"];\n";
        m_m_parent_node = node_name;
      }

      for (const auto &fl : fls_set) {
        // REMOVE LETTERS ofs << "  " << m_m_parent_node << " -> F" <<
        // get_unique_f_id_from_map(fl) << " [label=\"" << goal_counter <<
        // "\"];\n";
        os << "  " << to_binary_string(force_non_binary_ids, m_m_parent_node)
           << " -> "
           << to_binary_string(force_non_binary_ids,
                               get_unique_f_id_from_map(fl))
           << " [label=\"" << goal_counter << "\"];\n";
      }
    }
    break;
  }

  case BeliefFormulaType::BELIEF_FORMULA: {
    // REMOVE LETTERS node_name = "B" + std::to_string(current_node_id);
    node_name = std::to_string(current_node_id);
    // ofs << "  " << node_name << " [label=\"" << current_node_id << "\"];\n";
    os << "  " << to_binary_string(force_non_binary_ids, parent_node) << " -> "
       << to_binary_string(force_non_binary_ids, node_name) << " [label=\""
       << goal_counter << "\"];\n";
    // REMOVE LETTERS ofs << "  " << node_name << " -> A" <<
    // get_unique_a_id_from_map(to_print.get_agent()) << " [label=\"" <<
    // goal_counter << "\"];\n"; REMOVE LETTERS ofs << "  A" <<
    // get_unique_a_id_from_map(to_print.get_agent()) << " -> " << node_name <<
    // " [label=\"" << goal_counter << "\"];\n";
    os << "  " << to_binary_string(force_non_binary_ids, node_name) << " -> "
       << to_binary_string(force_non_binary_ids,
                           get_unique_a_id_from_map(to_print.get_agent()))
       << " [label=\"" << goal_counter << "\"];\n";
    os << "  "
       << to_binary_string(force_non_binary_ids,
                           get_unique_a_id_from_map(to_print.get_agent()))
       << " -> " << to_binary_string(force_non_binary_ids, node_name)
       << " [label=\"" << goal_counter << "\"];\n";

    generate_goal_subtree(to_print.get_bf1(), goal_counter, next_id, node_name,
                          os, force_non_binary_ids);
    break;
  }

  case BeliefFormulaType::C_FORMULA: {
    // REMOVE LETTERS node_name = "C" + std::to_string(current_node_id);
    node_name = std::to_string(current_node_id);
    // ofs << "  " << node_name << " [label=\"" << current_node_id << "\"];\n";
    os << "  " << to_binary_string(force_non_binary_ids, parent_node) << " -> "
       << to_binary_string(force_non_binary_ids, node_name) << " [label=\""
       << goal_counter << "\"];\n";

    for (const auto &ag : to_print.get_group_agents()) {
      // REMOVE LETTERS ofs << "  " << node_name << " -> A" <<
      // get_unique_a_id_from_map(ag) << " [label=\"" << goal_counter <<
      // "\"];\n"; REMOVE LETTERS ofs << "  A" << get_unique_a_id_from_map(ag)
      // << " -> " << node_name << " [label=\"" << goal_counter << "\"];\n";
      os << "  " << to_binary_string(force_non_binary_ids, node_name) << " -> "
         << to_binary_string(force_non_binary_ids, get_unique_a_id_from_map(ag))
         << " [label=\"" << goal_counter << "\"];\n";
      os << "  "
         << to_binary_string(force_non_binary_ids, get_unique_a_id_from_map(ag))
         << " -> " << to_binary_string(force_non_binary_ids, node_name)
         << " [label=\"" << goal_counter << "\"];\n";
    }

    generate_goal_subtree(to_print.get_bf1(), goal_counter, next_id, node_name,
                          os, force_non_binary_ids);
    break;
  }

  case BeliefFormulaType::PROPOSITIONAL_FORMULA: {
    switch (to_print.get_operator()) {
    case BeliefFormulaOperator::BF_NOT:
    case BeliefFormulaOperator::BF_AND:
    case BeliefFormulaOperator::BF_OR: {
      break;
    }
    case BeliefFormulaOperator::BF_FAIL:
    default: {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::BeliefFormulaOperatorUnset,
          "Error in reading a Belief Formula during the GOAL dot generation.");
      break;
    }
    }

    // REMOVE LETTERS node_name = node_name + std::to_string(current_node_id);
    node_name = std::to_string(current_node_id);
    // ofs << "  " << node_name << " [label=\"" << current_node_id << "\"];\n";
    os << "  " << to_binary_string(force_non_binary_ids, parent_node) << " -> "
       << to_binary_string(force_non_binary_ids, node_name) << " [label=\""
       << goal_counter << "\"];\n";
    generate_goal_subtree(to_print.get_bf1(), goal_counter, next_id, node_name,
                          os, force_non_binary_ids);

    if (!to_print.is_bf2_null()) {
      generate_goal_subtree(to_print.get_bf2(), goal_counter, next_id,
                            node_name, os, force_non_binary_ids);
    }

    break;
  }

  case BeliefFormulaType::BF_EMPTY:
  case BeliefFormulaType::BF_TYPE_FAIL:
  default: {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::BeliefFormulaTypeUnset,
        "Error in reading a Belief Formula during the GOAL dot generation.");
    break;
  }
  }
}

template <StateRepresentation StateRepr>
bool TrainingDataset<StateRepr>::search_space_exploration() {
  State<StateRepr> initial_state;
  initial_state.build_initial();

  if (Configuration::get_instance().get_bisimulation()) {
    initial_state.contract_with_bisimulation();
  }

  ActionsSet actions = Domain::get_instance().get_actions();

  const auto start_time = std::chrono::system_clock::now();

  bool result = false;

  const auto max_depth = ArgumentParser::get_instance().get_dataset_depth();
  m_visited_states.clear();

  if (const size_t branching_factor = actions.size(); branching_factor <= 1) {
    m_total_possible_nodes_log = log(max_depth + 1);
  } else {
    // Calculate expected log of total nodes
    const double numerator_log = (max_depth + 1) * std::log(branching_factor);
    const double denominator_log = std::log(branching_factor - 1);
    m_total_possible_nodes_log = numerator_log - denominator_log;
  }

  auto &os = ArgumentParser::get_instance().get_output_stream();

  m_threshold_node_generation =
      ArgumentParser::get_instance().get_generation_threshold();
  m_threshold_node_generation_log = std::log(m_threshold_node_generation * 3);
  m_max_threshold_node_creation =
      ArgumentParser::get_instance().get_max_creation_threshold();
  m_min_threshold_node_creation =
      ArgumentParser::get_instance().get_min_creation_threshold();

  os << "Total possible nodes exceed threshold." << std::endl;
  os << "Approximate number of nodes (exp(log)) = "
     << std::exp(m_total_possible_nodes_log) << std::endl;
  os << "Threshold number of nodes = " << m_threshold_node_generation
     << std::endl;

  const auto dataset_generation_type = ArgumentParser::get_instance().get_dataset_generation_type();
  const auto dataset_generation_type_string = ArgumentParser::get_instance().get_dataset_generation_type_string();

  os << "Using " << dataset_generation_type_string << " as dataset generation strategy." << std::endl;
  if (m_total_possible_nodes_log < m_threshold_node_generation_log) {
    os << "Switching to non-stochastic DFS (i.e., complete) because the number of nodes the search space is too low." << std::endl;
  }
  os << "Seed = " << m_seed << std::endl;

  switch (dataset_generation_type) {
    case DatasetGenerationType::BFS:
      result = bfs_exploration(initial_state, &actions);
      break;
    case DatasetGenerationType::DFS:
      result = dfs_exploration(initial_state, &actions, false);
      break;
    case DatasetGenerationType::S_DFS:
      result = dfs_exploration(initial_state, &actions, true);
      break;
    case DatasetGenerationType::HFS:
      result = hfs_exploration(initial_state, &actions);
      break;
    default:
      ExitHandler::exit_with_message(
        ExitHandler::ExitCode::DatasetGenerationTypeWrong,
        "Error in the Dataset Generation type.");
      break;
  }

  if (m_goal_founds > 0) {
    os << "Number of goals found: " << m_goal_founds << std::endl;
  } else {
    os << "[WARNING] No goals found with " << dataset_generation_type_string << " as exploration strategy, this is not a good training set (recreate it with more nodes for exploration, a different seed (if stochastic in particular), mode depth, or a different strategy altogether)."
       << std::endl;
  }


  const auto end_time = std::chrono::system_clock::now();
  const std::chrono::duration<double> elapsed = end_time - start_time;
  //auto &os = ArgumentParser::get_instance().get_output_stream();
  os << "\nDataset Generated in " << elapsed.count() << " seconds."
     << std::endl;
  os << "Dataset stored in " << m_folder << " folder." << std::endl;

  return result;
}

template <StateRepresentation StateRepr>
bool TrainingDataset<StateRepr>::bfs_exploration(
    State<StateRepr> &initial_state, const ActionsSet *actions) {
  return priority_exploration(initial_state, actions, false);
}

template <StateRepresentation StateRepr>
bool TrainingDataset<StateRepr>::hfs_exploration(
    State<StateRepr> &initial_state, const ActionsSet *actions) {
  return priority_exploration(initial_state, actions, true);
}

template <StateRepresentation StateRepr>
bool TrainingDataset<StateRepr>::priority_exploration(
    State<StateRepr> &initial_state, const ActionsSet *actions,
    const bool use_heuristic) {

  using StateType = State<StateRepr>;
  using StatePtr = const StateType *;

  struct QueueEntry {
    StatePtr state;
    size_t depth;
    std::string filename;
    int priority;
    size_t sequence;
  };

  struct Compare {
    bool operator()(const QueueEntry &lhs,
                    const QueueEntry &rhs) const {
      if (lhs.priority != rhs.priority) {
        return lhs.priority > rhs.priority;
      }
      return lhs.sequence > rhs.sequence;
    }
  };

  struct ExplorationInfo {
    std::vector<StatePtr> predecessors;
    std::streamoff score_position = -1;
    int score;
  };

  /*
   * Full states are owned only by m_visited_states. All graph references are
   * pointers to those states, avoiding copies of full State objects.
   */
  std::map<StatePtr, ExplorationInfo> exploration_info;
  std::priority_queue<QueueEntry, std::vector<QueueEntry>, Compare> queue;
  std::queue<StatePtr> reverse_queue;

  size_t sequence = 0;

  const size_t max_depth = static_cast<size_t>(
      ArgumentParser::get_instance().get_dataset_depth());

  constexpr int score_width = 10;

  std::unique_ptr<HeuristicsManager<StateRepr>> heuristics_manager;

  if (use_heuristic) {
    heuristics_manager =
        std::make_unique<HeuristicsManager<StateRepr>>(initial_state);

    auto &os = ArgumentParser::get_instance().get_output_stream();
    os << "Dataset HFS heuristic: "
       << heuristics_manager->get_used_h_name() << std::endl;
  }

  /*
   * Keep the CSV open for the complete forward and reverse phases. The score
   * field is fixed-width, so it can be overwritten in place later without
   * shifting the remainder of any row.
   */
  std::fstream csv_file(
      m_filepath_csv, std::ios::in | std::ios::out | std::ios::ate);

  if (!csv_file.is_open()) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::NNTrainingFileError,
        "Error opening file: " + m_filepath_csv);
  }

  auto write_row = [&](const std::string &base_filename,
                       const size_t depth,
                       const int score,
                       const std::string &predecessor,
                       const std::string &action) -> std::streamoff {
    const std::string filename = format_name(base_filename);
    const std::string predecessor_filename = format_name(predecessor);

    csv_file << filename << "," << depth << ",";

    const std::streamoff score_position =
        static_cast<std::streamoff>(csv_file.tellp());

    csv_file << std::setw(score_width)
             << std::setfill('0')
             << score
             << std::setfill(' ')
             << ","
             << m_goal_file_path
             << ","
             << predecessor_filename
             << ","
             << m_action_to_id[action]
             << "\n";

    return score_position;
  };

  /*
   * ============================================================
   * INITIAL STATE
   * ============================================================
   */
  int initial_priority = 0;
  if (use_heuristic) {
    initial_priority =
        heuristics_manager->get_heuristic_value(initial_state);
    initial_state.set_heuristic_value(initial_priority);
  } else {
    initial_state.set_heuristic_value(0);
  }

  auto [initial_it, initial_inserted] =
      m_visited_states.insert(initial_state);
  (void)initial_inserted;

  const StatePtr initial_ptr = std::addressof(*initial_it);
  const std::string initial_filename =
      print_state_for_dataset(*initial_ptr);

  ExplorationInfo initial_info;
  initial_info.score = initial_ptr->is_goal() ? 0 : m_failed_state;
  initial_info.score_position =
      write_row(initial_filename, 0, initial_info.score, "init", "no-op");

  exploration_info.emplace(initial_ptr, std::move(initial_info));

  queue.push({initial_ptr, 0, initial_filename,
              initial_priority, sequence++});

  /*
   * ============================================================
   * PHASE 1: FORWARD BFS / HFS EXPLORATION
   * ============================================================
   *
   * BFS has priority zero for every state. sequence is then the sole
   * discriminator, giving FIFO behavior.
   *
   * HFS uses the configured heuristic as priority, with FIFO ordering among
   * states having the same heuristic value.
   */
  while (!queue.empty() &&
         m_current_nodes < m_threshold_node_generation) {

    QueueEntry current = queue.top();
    queue.pop();

    const StatePtr current_ptr = current.state;

    /*
     * std::set exposes elements as const. compute_successor() is non-const,
     * therefore expand a single temporary mutable copy.
     */
    StateType state = *current_ptr;

    ++m_current_nodes;

#ifdef DEBUG
    if (m_threshold_node_generation > 0) {
      const int percent = static_cast<int>(
          (m_current_nodes * 100) / m_threshold_node_generation);

      static int last_percent = -1;

      if (percent != last_percent) {
        last_percent = percent;

        auto &os = ArgumentParser::get_instance().get_output_stream();

        os << std::left
           << std::setw(35)
           << (use_heuristic
                   ? "[DEBUG] Dataset Generation Progress with HFS:"
                   : "[DEBUG] Dataset Generation Progress with BFS:")
           << " " << std::setw(5)
           << (std::to_string(percent) + "%")
           << " " << std::setw(20) << "Explored nodes:"
           << " " << std::setw(10) << m_current_nodes
           << " " << std::setw(15) << "Current Depth:"
           << " " << std::setw(5) << current.depth;

        if (use_heuristic) {
          os << " " << std::setw(15) << "Heuristic:"
             << " " << std::setw(10) << current.priority;
        }

        os << " " << std::setw(15) << "Goals found:"
           << " " << std::setw(10) << m_goal_founds
           << " " << std::setw(15) << "Dataset nodes:"
           << " " << std::setw(10) << m_added_to_dataset
           << std::endl;
      }
    }
#endif

    if (state.is_goal()) {
      ++m_goal_founds;
      exploration_info.find(current_ptr)->second.score = 0;
    }

    if (current.depth >= max_depth) {
      continue;
    }

    /*
     * Do NOT use m_max_threshold_node_creation here. For dataset exploration,
     * the generation threshold limits the number of expanded nodes. Every
     * discovered BFS/HFS state is kept in the dataset.
     */
    for (const auto &action : *actions) {

      if (!state.is_executable(action)) {
        continue;
      }

      auto next_state = state.compute_successor(action);

      if (Configuration::get_instance().get_bisimulation()) {
        next_state.contract_with_bisimulation();
      }

      /*
       * Look up the canonical state before calculating its heuristic. This
       * avoids unnecessary heuristic evaluation for already discovered states.
       */
      auto existing = m_visited_states.find(next_state);

      if (existing != m_visited_states.end()) {
        const StatePtr next_ptr = std::addressof(*existing);

        auto info_it = exploration_info.find(next_ptr);
        if (info_it == exploration_info.end()) {
          ExitHandler::exit_with_message(
              ExitHandler::ExitCode::NNTrainingFileError,
              "Internal BFS/HFS error: discovered state has no exploration "
              "metadata.");
        }

        /*
         * Preserve every graph edge. This is necessary for the exact reverse
         * distance computation.
         */
        info_it->second.predecessors.push_back(current_ptr);
        continue;
      }

      int priority = 0;

      if (use_heuristic) {
        priority =
            heuristics_manager->get_heuristic_value(next_state);
        next_state.set_heuristic_value(priority);
      } else {
        next_state.set_heuristic_value(0);
      }

      /*
       * Insert the full state exactly once. std::set guarantees that the
       * address of the inserted element remains stable while it stays in the
       * set.
       */
      auto [next_it, inserted] =
          m_visited_states.insert(std::move(next_state));

      const StatePtr next_ptr = std::addressof(*next_it);

      if (!inserted) {
        auto info_it = exploration_info.find(next_ptr);
        if (info_it == exploration_info.end()) {
          ExitHandler::exit_with_message(
              ExitHandler::ExitCode::NNTrainingFileError,
              "Internal BFS/HFS error: state insertion mismatch.");
        }

        info_it->second.predecessors.push_back(current_ptr);
        continue;
      }

      const std::string next_filename =
          print_state_for_dataset(*next_ptr);

      ExplorationInfo info;
      info.score = next_ptr->is_goal() ? 0 : m_failed_state;
      info.score_position =
          write_row(next_filename,
                    current.depth + 1,
                    info.score,
                    current.filename,
                    action.get_name());

      /*
       * current -> next, therefore current is a predecessor of next.
       */
      info.predecessors.push_back(current_ptr);

      exploration_info.emplace(next_ptr, std::move(info));

      queue.push({next_ptr,
                  current.depth + 1,
                  next_filename,
                  priority,
                  sequence++});
    }
  }

  csv_file.flush();

  /*
   * ============================================================
   * PHASE 2: REVERSE MULTI-SOURCE BFS
   * ============================================================
   *
   * Start from every discovered goal with score 0. Since all edges have unit
   * cost, the first reverse-BFS visit of a state gives its shortest distance to
   * any discovered goal.
   */
  for (auto &[state_ptr, info] : exploration_info) {
    if (state_ptr->is_goal()) {
      info.score = 0;
      reverse_queue.push(state_ptr);
    } else {
      info.score = m_failed_state;
    }
  }

  while (!reverse_queue.empty()) {
    const StatePtr current_ptr = reverse_queue.front();
    reverse_queue.pop();

    const auto current_it = exploration_info.find(current_ptr);
    if (current_it == exploration_info.end()) {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::NNTrainingFileError,
          "Internal BFS/HFS error: reverse state metadata not found.");
    }

    const int predecessor_score =
        current_it->second.score + 1;

    for (const StatePtr predecessor :
         current_it->second.predecessors) {

      auto predecessor_it = exploration_info.find(predecessor);

      if (predecessor_it == exploration_info.end()) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::NNTrainingFileError,
            "Internal BFS/HFS error: predecessor metadata not found.");
      }

      if (predecessor_it->second.score != m_failed_state) {
        continue;
      }

      predecessor_it->second.score = predecessor_score;
      reverse_queue.push(predecessor);
    }
  }

  /*
   * ============================================================
   * PHASE 3: UPDATE SCORES IN THE CSV
   * ============================================================
   */
  for (const auto &[state_ptr, info] : exploration_info) {
    (void)state_ptr;

    csv_file.seekp(info.score_position);
    csv_file << std::setw(score_width)
             << std::setfill('0')
             << info.score
             << std::setfill(' ');
  }

  csv_file.flush();
  csv_file.close();

#ifdef DEBUG
  size_t reachable_count = 0;
  size_t failed_count = 0;

  for (const auto &[state_ptr, info] : exploration_info) {
    (void)state_ptr;
    if (info.score == m_failed_state) {
      ++failed_count;
    } else {
      ++reachable_count;
    }
  }

  auto &debug_os = ArgumentParser::get_instance().get_output_stream();
  debug_os << "[DEBUG] States with finite distance-to-goal: "
           << reachable_count << std::endl;
  debug_os << "[DEBUG] States with no discovered goal reachable: "
           << failed_count << std::endl;
#endif

  return ((m_goal_founds > 0) &&
          (m_added_to_dataset > m_min_threshold_node_creation));
}

template <StateRepresentation StateRepr>
bool TrainingDataset<StateRepr>::dfs_exploration(
    State<StateRepr> &initial_state, ActionsSet *actions, const bool is_stochastic) {

  dfs_worker(initial_state, 0, actions, "init", "no-op",is_stochastic);
  return (
    (m_goal_founds > 0) &&
    (m_added_to_dataset >
    m_min_threshold_node_creation)); // Return true if dataset is not empty and goals were found and if we added at least a minimum number of nodes
}

template <StateRepresentation StateRepr>
int TrainingDataset<StateRepr>::dfs_worker(State<StateRepr> &state,
                                           const size_t depth,
                                           ActionsSet *actions,
                                           const std::string &predecessor,
                                           const std::string &action,
                                           const bool is_stochastic) {
#ifdef DEBUG
  if (m_current_nodes > 0 && m_threshold_node_generation > 0) {
    int percent = (m_current_nodes * 100) / m_threshold_node_generation;
    static int last_percent = -1;
    if (percent != last_percent) {
      last_percent = percent;
      auto &os = ArgumentParser::get_instance().get_output_stream();

      os << std::left << std::setw(35)
         << "[DEBUG] Dataset Generation Progress with DFS:" << " " << std::setw(5)
         << (std::to_string(percent) + "%") << " " << std::setw(20)
         << "Explored nodes:" << " " << std::setw(10) << m_current_nodes << " "
         << std::setw(15) << "Current Depth:" << " " << std::setw(5) << depth
         << " " << std::setw(15) << "Goals found:" << " " << std::setw(10)
         << m_goal_founds << " " << std::setw(15) << "Valid nodes found:" << " "
         << std::setw(5) << m_added_to_dataset << std::endl;
    }
  }
#endif

  auto this_state_filename = print_state_for_dataset(state);

  if (m_current_nodes >= m_threshold_node_generation) {
    if (state.is_goal()) {
      add_to_dataset(this_state_filename, depth, 0, predecessor, action);
      return 0;
    }
    return m_failed_state;
  }

  // If already visited, return memoized score
  if (m_visited_states.contains(state)) {
    return m_states_scores[state];
  }
  m_current_nodes++;

  // Initial score
  int current_score = m_failed_state;

  if (state.is_goal()) {
    current_score = 0;
    m_goal_founds++;
    m_goal_recently_found = true;
  }

  // Mark state as visited before recursion
  m_visited_states.insert(state);
  m_states_scores[state] = current_score;

  int best_successor_score = m_failed_state;

  const auto max_depth =
      static_cast<size_t>(ArgumentParser::get_instance().get_dataset_depth());

  if (depth < max_depth) {
    // Possibility of discarding the current state
    {
      // Compute discard probability
      double discard_probability = 0.0;
      if (m_total_possible_nodes_log > m_threshold_node_generation_log && is_stochastic) {
        const double depth_ratio = static_cast<double>(depth) / max_depth;
        const double fullness_ratio =
            static_cast<double>(m_current_nodes) /
            static_cast<double>(m_threshold_node_generation);

        discard_probability += 0.2 * std::pow(depth_ratio, 2);
        discard_probability += 0.2 * fullness_ratio;
        discard_probability += std::min(
            0.01 * std::pow(static_cast<double>(m_discard_augmentation_factor) /
                                (3 * max_depth),
                            2),
            0.1);
        if (m_goal_recently_found) {
          discard_probability += 0.2;
        }

        const auto discard_factor =
            ArgumentParser::get_instance().get_dataset_discard_factor();
        if (discard_factor < 0.0 || discard_factor >= 1.0) {
          ExitHandler::exit_with_message(ExitHandler::ExitCode::ParsingError,
                                         "Invalid discard factor: " +
                                             std::to_string(discard_factor));
        }

        discard_probability = std::min(discard_probability, discard_factor);
      }

      if (m_dis(m_gen) < discard_probability) {
        m_goal_recently_found = false;
        m_discard_augmentation_factor = 0.0;
        // Still record the current state (even if skipping)

        // std::cout << "[DEBUG] Discarding state at depth "
        //           << depth << " with score " << current_score
        //           << " and discard probability " << discard_probability
        //           << std::endl;
        add_to_dataset(this_state_filename, depth, current_score, predecessor,
                       action);
        m_states_scores[state] = current_score;
        return current_score;
      }

      m_discard_augmentation_factor++;
    }

    std::vector<Action> local_actions(actions->begin(), actions->end());
    std::ranges::shuffle(local_actions, m_gen);

    for (const auto &action : local_actions) {
      if (state.is_executable(action)) {
        auto next_state = state.compute_successor(action);

        if (Configuration::get_instance().get_bisimulation()) {
          next_state.contract_with_bisimulation();
        }

        const int child_score =
            dfs_worker(next_state, depth + 1, actions, this_state_filename,
                       action.get_name(), is_stochastic);

        if (child_score < best_successor_score) {
          best_successor_score = child_score;
        }
      }
    }
  }

  if (current_score > (best_successor_score + 1)) {
    current_score = best_successor_score + 1;
  }

  add_to_dataset(this_state_filename, depth, current_score, predecessor,
                 action);
  m_states_scores[state] = current_score;

  return current_score;
}

template <StateRepresentation StateRepr>
void TrainingDataset<StateRepr>::add_to_dataset(
    const std::string &base_filename, const size_t depth, const int score,
    const std::string &predecessor, const std::string &action) {
  constexpr bool minimized_dataset = false;

  if (minimized_dataset && score >= m_failed_state) {
    return;
  }

  if (score >= m_failed_state) {
    auto m_total_failures = m_current_nodes - m_added_to_dataset;
    if (m_total_failures % m_threshold_failures_print_modulo != 0) {
      return;
    }
  } else {
    m_added_to_dataset++;
  }

  std::stringstream ss;

  std::string filename = format_name(base_filename);
  std::string predecessor_filename = format_name(predecessor);

  ss << filename << "," << depth << "," << score << "," << m_goal_file_path
     << "," << predecessor_filename << "," << m_action_to_id[action];

  std::ofstream result_file(m_filepath_csv, std::ofstream::app);
  result_file << ss.str() << "\n";
  result_file.close();
}

template <StateRepresentation StateRepr>
std::string TrainingDataset<StateRepr>::print_state_for_dataset(
    const State<StateRepr> &state) {
  ++m_file_counter;
  std::string base_filename =
      std::string(6 - std::to_string(m_file_counter).length(), '0') +
      std::to_string(m_file_counter);

  std::ofstream out(format_name(base_filename));
  state.print_dataset_format(out);
  out.close();

  return base_filename;
}

template <StateRepresentation StateRepr>
std::string TrainingDataset<StateRepr>::format_name(
    const std::string &base_filename) const {
  std::string result_filename = create_complete_path() + base_filename + ".dot";
  return result_filename;
}
