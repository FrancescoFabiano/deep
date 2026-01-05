#include "ArgumentParser.h"
#include "Configuration.h"
#include "Domain.h"
#include "ExitHandler.h"
#include "HelperPrint.h"
#include "TrainingDataset.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip> // Make sure this is included at the top of your file
#include <sstream>
#include <string>

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
    m_folder =
        make_unique_folder(OutputPaths::DATASET_TRAINING_FOLDER, domain_name);
    m_training_raw_files_folder = m_folder + "RawFiles/";

    const std::string filename =
        domain_name + "_depth_" +
        std::to_string(ArgumentParser::get_instance().get_dataset_depth()) +
        ".csv";

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
void TrainingDataset<StateRepr>::create_instance() {
  if (!instance) {
    instance = new TrainingDataset();
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
  result << "File Path,Depth,Distance From Goal,Goal,File Path Predecessor,Action" << std::endl;
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

  auto start_time = std::chrono::system_clock::now();

  const bool result = dfs_exploration(initial_state, &actions);

  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;
  auto &os = ArgumentParser::get_instance().get_output_stream();
  os << "\nDataset Generated in " << elapsed.count() << " seconds."
     << std::endl;
  os << "Dataset stored in " << m_folder << " folder." << std::endl;

  return result;
}

template <StateRepresentation StateRepr>
bool TrainingDataset<StateRepr>::dfs_exploration(
    State<StateRepr> &initial_state, ActionsSet *actions) {
  const auto max_depth = ArgumentParser::get_instance().get_dataset_depth();
  m_visited_states.clear();

  if (const size_t branching_factor = actions->size(); branching_factor <= 1) {
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
  if (m_total_possible_nodes_log > m_threshold_node_generation_log) {
    os << "Decision: using SPARSE DFS." << std::endl;
  } else {
    os << "Decision: using COMPLETE DFS." << std::endl;
  }
  os << "Seed = " << m_seed << std::endl;

  dfs_worker(initial_state, 0, actions, "init", "no-op");

  if (m_goal_founds > 0) {
    os << "Number of goals found: " << m_goal_founds << std::endl;
  } else {
    os << "[WARNING] No goals found, this is not a good training set (recreate "
          "it)."
       << std::endl;
  }

  return (
      (m_goal_founds > 0) &&
      (m_added_to_dataset >
       m_min_threshold_node_creation)); // Return true if dataset is not empty
                                        // and goals were found and if we added
                                        // at least a minimum number of nodes
}

template <StateRepresentation StateRepr>
int TrainingDataset<StateRepr>::dfs_worker(State<StateRepr> &state,
                                           const size_t depth,
                                           ActionsSet *actions,
                                           const std::string & predecessor,
                                           const std::string & action) {
#ifdef DEBUG
  if (m_current_nodes > 0 && m_threshold_node_generation > 0) {
    int percent = (m_current_nodes * 100) / m_threshold_node_generation;
    static int last_percent = -1;
    if (percent != last_percent) {
      last_percent = percent;
      auto &os = ArgumentParser::get_instance().get_output_stream();

      os << std::left << std::setw(35)
         << "[DEBUG] Dataset Generation Progress:" << " " << std::setw(5)
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


  if (m_current_nodes >= m_threshold_node_generation ||
      m_added_to_dataset >= m_max_threshold_node_creation) {
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
      if (m_total_possible_nodes_log > m_threshold_node_generation_log) {
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
        add_to_dataset(this_state_filename, depth, current_score, predecessor, action);
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

        const int child_score = dfs_worker(next_state, depth + 1, actions, this_state_filename, action.get_name());

        if (child_score < best_successor_score) {
          best_successor_score = child_score;
        }
      }
    }
  }

  if (current_score > (best_successor_score + 1)) {
    current_score = best_successor_score + 1;
  }

  add_to_dataset(this_state_filename, depth, current_score, predecessor, action);
  m_states_scores[state] = current_score;

  return current_score;
}

template <StateRepresentation StateRepr>
void TrainingDataset<StateRepr>::add_to_dataset(const std::string &base_filename,
                                                const size_t depth,
                                                const int score,
                                                const std::string & predecessor,
                                                const std::string & action) {
  constexpr bool minimized_dataset = true;

  if (minimized_dataset && score >= m_failed_state) {
    return;

  }

  m_added_to_dataset++;

  std::stringstream ss;

  std::string filename = format_name(base_filename);
  std::string predecessor_filename = format_name(predecessor);

  ss << filename << "," << depth << "," << score << "," << m_goal_file_path << "," << predecessor_filename << "," << action;

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
