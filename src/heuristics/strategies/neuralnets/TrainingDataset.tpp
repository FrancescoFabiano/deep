#include "ArgumentParser.h"
#include "Configuration.h"
#include "Domain.h"
#include "ExitHandler.h"
#include "TrainingDataset.h"
#include <chrono>
#include <filesystem>
#include <fstream>
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
constexpr const std::string &
TrainingDataset<StateRepr>::get_to_goal_edge_id_string() {
  return m_to_goal_edge_id;
}

template <StateRepresentation StateRepr>
constexpr const std::string &
TrainingDataset<StateRepr>::get_to_state_edge_id_string() {
  return m_to_state_edge_id;
}

template <StateRepresentation StateRepr>
constexpr const std::string &
TrainingDataset<StateRepr>::get_epsilon_node_id_string() {
  return m_epsilon_node_id;
}

template <StateRepresentation StateRepr>
constexpr const std::string &
TrainingDataset<StateRepr>::get_goal_parent_id_string() {
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
int TrainingDataset<StateRepr>::get_shift_state_ids() const {
  return m_shift_state_ids;
}

template <StateRepresentation StateRepr>
TrainingDataset<StateRepr>::TrainingDataset() {
  const std::string domain_name = Domain::get_instance().get_name();

  if (ArgumentParser::get_instance().get_dataset_mode()) {
    m_folder = std::string(OutputPaths::DATASET_TRAINING_FOLDER) + "/" +
               domain_name + "/";
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
      if (ArgumentParser::get_instance().get_dataset_mapped() ||
          ArgumentParser::get_instance().get_dataset_both()) {
        if (ArgumentParser::get_instance().get_dataset_merged() ||
            ArgumentParser::get_instance().get_dataset_merged_both()) {
          std::filesystem::create_directories(
              m_training_raw_files_folder +
              std::string(OutputPaths::DATASET_NN_DATASET_MAPPED) + "/" +
              std::string(OutputPaths::DATASET_NN_DATASET_MERGED) + "/");
        }
        if (!ArgumentParser::get_instance().get_dataset_merged() ||
            ArgumentParser::get_instance().get_dataset_merged_both()) {
          std::filesystem::create_directories(
              m_training_raw_files_folder +
              std::string(OutputPaths::DATASET_NN_DATASET_MAPPED) + "/" +
              std::string(OutputPaths::DATASET_NN_DATASET_SEPARATED) + "/");
        }
      }
      if (!ArgumentParser::get_instance().get_dataset_mapped() ||
          ArgumentParser::get_instance().get_dataset_both()) {
        if (ArgumentParser::get_instance().get_dataset_merged() ||
            ArgumentParser::get_instance().get_dataset_merged_both()) {
          std::filesystem::create_directories(
              m_training_raw_files_folder +
              std::string(OutputPaths::DATASET_NN_DATASET_HASHED) + "/" +
              std::string(OutputPaths::DATASET_NN_DATASET_MERGED) + "/");
        }
        if (!ArgumentParser::get_instance().get_dataset_merged() ||
            ArgumentParser::get_instance().get_dataset_merged_both()) {
          std::filesystem::create_directories(
              m_training_raw_files_folder +
              std::string(OutputPaths::DATASET_NN_DATASET_HASHED) + "/" +
              std::string(OutputPaths::DATASET_NN_DATASET_SEPARATED) + "/");
        }
      }
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

  // This stores the goal tree in a string for efficient printing
  generate_goal_tree_subgraph();

  if (ArgumentParser::get_instance().get_dataset_merged() ||
      ArgumentParser::get_instance().get_dataset_merged_both()) {
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
  result << "Path Hash,Path Hash Merged,Path Mapped,Path Mapped "
            "Merged,Depth,Distance From Goal,Goal"
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
void TrainingDataset<StateRepr>::generate_goal_tree_subgraph() {
  std::stringstream string_goals_graph;
  const auto goal_list = Domain::get_instance().get_goal_description();
  size_t goal_counter = m_to_state_edge_id_int + 1;

  size_t next_id = m_shift_state_ids;

  // This is the root node of the root node of goals connected to the parent
  // when exists (which has m_failed_state as label)
  for (const auto &goal : goal_list) {
    generate_goal_subtree(goal, ++goal_counter, next_id, m_goal_parent_id,
                          string_goals_graph);
  }

  m_shift_state_ids += +1;
  // The final value of shits so that state, when mapped starts from the latest
  // node generated for the goals + 1

  m_goal_string = string_goals_graph.str();
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
    const std::string &parent_node, std::ostream &os) {
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
      os << "  " << parent_node << " -> " << node_name << " [label=\""
         << goal_counter << "\"];\n";
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
        os << "  " << m_parent_node << " -> " << node_name << " [label=\""
           << goal_counter << "\"];\n";
        m_m_parent_node = node_name;
      }

      for (const auto &fl : fls_set) {
        // REMOVE LETTERS ofs << "  " << m_m_parent_node << " -> F" <<
        // get_unique_f_id_from_map(fl) << " [label=\"" << goal_counter <<
        // "\"];\n";
        os << "  " << m_m_parent_node << " -> " << get_unique_f_id_from_map(fl)
           << " [label=\"" << goal_counter << "\"];\n";
      }
    }
    break;
  }

  case BeliefFormulaType::BELIEF_FORMULA: {
    // REMOVE LETTERS node_name = "B" + std::to_string(current_node_id);
    node_name = std::to_string(current_node_id);
    // ofs << "  " << node_name << " [label=\"" << current_node_id << "\"];\n";
    os << "  " << parent_node << " -> " << node_name << " [label=\""
       << goal_counter << "\"];\n";
    // REMOVE LETTERS ofs << "  " << node_name << " -> A" <<
    // get_unique_a_id_from_map(to_print.get_agent()) << " [label=\"" <<
    // goal_counter << "\"];\n"; REMOVE LETTERS ofs << "  A" <<
    // get_unique_a_id_from_map(to_print.get_agent()) << " -> " << node_name <<
    // " [label=\"" << goal_counter << "\"];\n";
    os << "  " << node_name << " -> "
       << get_unique_a_id_from_map(to_print.get_agent()) << " [label=\""
       << goal_counter << "\"];\n";
    os << "  " << get_unique_a_id_from_map(to_print.get_agent()) << " -> "
       << node_name << " [label=\"" << goal_counter << "\"];\n";

    generate_goal_subtree(to_print.get_bf1(), goal_counter, next_id, node_name,
                          os);
    break;
  }

  case BeliefFormulaType::C_FORMULA: {
    // REMOVE LETTERS node_name = "C" + std::to_string(current_node_id);
    node_name = std::to_string(current_node_id);
    // ofs << "  " << node_name << " [label=\"" << current_node_id << "\"];\n";
    os << "  " << parent_node << " -> " << node_name << " [label=\""
       << goal_counter << "\"];\n";

    for (const auto &ag : to_print.get_group_agents()) {
      // REMOVE LETTERS ofs << "  " << node_name << " -> A" <<
      // get_unique_a_id_from_map(ag) << " [label=\"" << goal_counter <<
      // "\"];\n"; REMOVE LETTERS ofs << "  A" << get_unique_a_id_from_map(ag)
      // << " -> " << node_name << " [label=\"" << goal_counter << "\"];\n";
      os << "  " << node_name << " -> " << get_unique_a_id_from_map(ag)
         << " [label=\"" << goal_counter << "\"];\n";
      os << "  " << get_unique_a_id_from_map(ag) << " -> " << node_name
         << " [label=\"" << goal_counter << "\"];\n";
    }

    generate_goal_subtree(to_print.get_bf1(), goal_counter, next_id, node_name,
                          os);
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
    os << "  " << parent_node << " -> " << node_name << " [label=\""
       << goal_counter << "\"];\n";
    generate_goal_subtree(to_print.get_bf1(), goal_counter, next_id, node_name,
                          os);

    if (!to_print.is_bf2_null()) {
      generate_goal_subtree(to_print.get_bf2(), goal_counter, next_id,
                            node_name, os);
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

  std::vector<std::string> global_dataset;
  global_dataset.reserve(m_threshold_node_generation);

  const bool result = dfs_exploration(initial_state, &actions, global_dataset);

  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;
  auto &os = ArgumentParser::get_instance().get_output_stream();
  os << "\nDataset Generated in " << elapsed.count() << " seconds" << std::endl;

  std::ofstream result_file(m_filepath_csv, std::ofstream::app);
  for (const auto &row : global_dataset) {
    result_file << row << "\n";
  }
  result_file.close();

  return result;
}

template <StateRepresentation StateRepr>
bool TrainingDataset<StateRepr>::dfs_exploration(
    State<StateRepr> &initial_state, ActionsSet *actions,
    std::vector<std::string> &global_dataset) {
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

  dfs_worker(initial_state, 0, actions, global_dataset);

  if (m_goal_founds > 0) {
    os << "Number of goals found: " << m_goal_founds << std::endl;
  } else {
    os << "[WARNING] No goals found, this is not a good training set (recreate "
          "it)."
       << std::endl;
  }

  return !global_dataset.empty();
}

template <StateRepresentation StateRepr>
int TrainingDataset<StateRepr>::dfs_worker(
    State<StateRepr> &state, const size_t depth, ActionsSet *actions,
    std::vector<std::string> &global_dataset) {
  if (m_current_nodes >= m_threshold_node_generation) {
    if (state.is_goal()) {
      global_dataset.push_back(format_row(state, depth, 0));
      return 0;
    }
    return m_failed_state;
  }
  m_current_nodes++;

  int current_score = m_failed_state;

  if (m_visited_states.count(state)) {
    return m_states_scores[state];
  }

  if (state.is_goal()) {
    current_score = 0;
    m_goal_founds++;
    m_goal_recently_found = true;
  }

  int best_successor_score = m_failed_state;
  const auto max_depth =
      static_cast<size_t>(ArgumentParser::get_instance().get_dataset_depth());
  // Create a local vector from action_set
  std::vector<Action> local_actions(actions->begin(), actions->end());

  // Shuffle local_actions
  std::ranges::shuffle(local_actions, m_gen);

  for (const auto &action : local_actions) {
    if (state.is_executable(action)) {
      auto next_state = state.compute_successor(action);

      if (Configuration::get_instance().get_bisimulation()) {
        next_state.contract_with_bisimulation();
      }

      if (depth >= max_depth) {
        break;
      } else {
        double discard_probability = 0.0;
        // Only start discarding if total search space is big
        if (m_total_possible_nodes_log > m_threshold_node_generation_log) {
          const auto depth_ratio =
              static_cast<double>(depth) / static_cast<double>(max_depth);
          const auto fullness_ratio =
              static_cast<double>(m_current_nodes) /
              static_cast<double>(m_threshold_node_generation);

          // Start discard_probability low, increase as depth grows
          discard_probability = 0.2 * std::pow(depth_ratio, 2);

          // Slightly boost as dataset fills up
          discard_probability += 0.2 * fullness_ratio;

          discard_probability += std::min(
              0.01 *
                  std::pow(static_cast<double>(m_discard_augmentation_factor) /
                               (3 * max_depth),
                           2),
              0.1);

          // Boost if a nearby goal was found
          if (m_goal_recently_found) {
            discard_probability += 0.2;
          }

          // Clamp between [0, 0.8] (still allow deeper exploration)
          discard_probability = std::min(discard_probability, 0.8);
        }

        if (m_dis(m_gen) < discard_probability) {
          m_goal_recently_found = false;
          m_discard_augmentation_factor = 0.0;
          continue; // Randomly skip exploration
        }

        // Increase the augmentation factor for non-discarded series
        m_discard_augmentation_factor++;

        const int child_score =
            dfs_worker(next_state, depth + 1, actions, global_dataset);

        if (child_score < best_successor_score) {
            best_successor_score = child_score;
          }

        }
    }
  }

  if (current_score > (best_successor_score + 1)) {
    current_score = best_successor_score + 1;
  }

  global_dataset.push_back(format_row(state, depth, current_score));
  m_visited_states.insert(state);
  m_states_scores[state] = current_score;

  return current_score;
}

template <StateRepresentation StateRepr>
std::string
TrainingDataset<StateRepr>::format_row(const State<StateRepr> &state,
                                       const size_t depth, const int score) {
  std::stringstream ss;
  auto base_filename = print_state_for_dataset(state);

  std::string filename_hash =
      format_name(base_filename, OutputPaths::DATASET_NN_DATASET_HASHED, false);
  std::string filename_hash_merged =
      format_name(base_filename, OutputPaths::DATASET_NN_DATASET_HASHED, true);
  std::string filename_emap =
      format_name(base_filename, OutputPaths::DATASET_NN_DATASET_MAPPED, false);
  std::string filename_emap_merged =
      format_name(base_filename, OutputPaths::DATASET_NN_DATASET_HASHED, true);

  if (ArgumentParser::get_instance().get_dataset_mapped() &&
      !ArgumentParser::get_instance().get_dataset_both()) {
    filename_hash = "NOT CALCULATED";
    filename_hash_merged = "NOT CALCULATED";
  }
  if (!ArgumentParser::get_instance().get_dataset_mapped() &&
      !ArgumentParser::get_instance().get_dataset_both()) {
    filename_emap = "NOT CALCULATED";
    filename_emap_merged = "NOT CALCULATED";
  }
  if (ArgumentParser::get_instance().get_dataset_merged() &&
      !ArgumentParser::get_instance().get_dataset_merged_both()) {
    filename_hash = "NOT CALCULATED";
    filename_emap = "NOT CALCULATED";
  }
  if (!ArgumentParser::get_instance().get_dataset_merged() &&
      !ArgumentParser::get_instance().get_dataset_merged_both()) {
    filename_hash_merged = "NOT CALCULATED";
    filename_emap_merged = "NOT CALCULATED";
  }

  ss << filename_hash << "," << filename_hash_merged << "," << filename_emap
     << "," << filename_emap_merged << "," << depth << "," << score << ","
     << m_goal_file_path;

  return ss.str();
}

template <StateRepresentation StateRepr>
std::string TrainingDataset<StateRepr>::print_state_for_dataset(
    const State<StateRepr> &state) {
  ++m_file_counter;
  std::string base_filename =
      std::string(6 - std::to_string(m_file_counter).length(), '0') +
      std::to_string(m_file_counter);

  if (ArgumentParser::get_instance().get_dataset_mapped() ||
      ArgumentParser::get_instance().get_dataset_both()) {
    if (ArgumentParser::get_instance().get_dataset_merged() ||
        ArgumentParser::get_instance().get_dataset_merged_both()) {
      print_state_for_dataset_internal(
          state, base_filename, OutputPaths::DATASET_NN_DATASET_MAPPED, true);
    }
    if (!ArgumentParser::get_instance().get_dataset_merged() ||
        ArgumentParser::get_instance().get_dataset_merged_both()) {
      print_state_for_dataset_internal(
          state, base_filename, OutputPaths::DATASET_NN_DATASET_MAPPED, false);
    }
  }
  if (!ArgumentParser::get_instance().get_dataset_mapped() ||
      ArgumentParser::get_instance().get_dataset_both()) {
    if (ArgumentParser::get_instance().get_dataset_merged() ||
        ArgumentParser::get_instance().get_dataset_merged_both()) {
      print_state_for_dataset_internal(
          state, base_filename, OutputPaths::DATASET_NN_DATASET_HASHED, true);
    }
    if (!ArgumentParser::get_instance().get_dataset_merged() ||
        ArgumentParser::get_instance().get_dataset_merged_both()) {
      print_state_for_dataset_internal(
          state, base_filename, OutputPaths::DATASET_NN_DATASET_HASHED, false);
    }
  }

  return base_filename;
}

template <StateRepresentation StateRepr>
void TrainingDataset<StateRepr>::print_state_for_dataset_internal(
    const State<StateRepr> &state, const std::string &base_filename,
    const std::string &type, const bool merged) const {
  std::ofstream out(format_name(base_filename, type, merged));
  state.print_dataset_format(
      out, type == OutputPaths::DATASET_NN_DATASET_HASHED, merged);
  out.close();
}

template <StateRepresentation StateRepr>
std::string
TrainingDataset<StateRepr>::format_name(const std::string &base_filename,
                                        const std::string &type,
                                        const bool merged) const {
  const std::string merge_folder =
      merged ? OutputPaths::DATASET_NN_DATASET_MERGED
             : OutputPaths::DATASET_NN_DATASET_SEPARATED;
  std::string result_filename = m_training_raw_files_folder + type + "/" +
                                merge_folder + "/" + base_filename + "_" +
                                type + ".dot";
  return result_filename;
}
