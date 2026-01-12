#pragma once
#include "Define.h"
#include "State.h"
#include <cmath>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

/// \brief Random device for seeding the random number generator.
inline std::random_device rd;

/// \brief Uniform real distribution in the range [0.0, 1.0).
inline std::uniform_real_distribution<> m_dis(0.0, 1.0);

/**
 * \class TrainingDataset
 * \brief Singleton class for Neural Network-based heuristic dataset generation
 * and management.
 *
 * This class is responsible for generating datasets for ML heuristics, managing
 * state and agent mappings, and providing utility functions for dataset
 * formatting and exploration. It is templated on the StateRepresentation.
 *
 * \tparam StateRepr The state representation type.
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 31, 2025
 */
template <StateRepresentation StateRepr> class TrainingDataset {
public:
  /**
   * \brief Get the singleton instance of TrainingDataset.
   * \return Reference to the singleton instance.
   */
  static TrainingDataset &get_instance();

  /**
   * \brief Create the singleton instance of TrainingDataset.
   */
  static void create_instance();

  /**
   * \brief Generates the dataset for ML heuristics.
   * \return True if dataset generation was successful, false otherwise.
   */
  [[nodiscard]] bool generate_dataset();

  /**
   * \brief Gets the folder path for datasets.
   * \return The path to the folder.
   */
  const std::string &get_folder() const;

  /// \brief Gets the to-goal edge ID as string.
  /// \return The to-goal edge ID.
  const std::string &get_to_goal_edge_id_string();

  /// \brief Gets the to-state edge ID as string.
  /// \return The to-state edge ID.
  const std::string &get_to_state_edge_id_string();

  /// \brief Gets the epsilon node ID as string.
  /// \return The epsilon node ID.
  const std::string &get_epsilon_node_id_string();

  /// \brief Gets the goal parent ID as string.
  /// \return The goal parent ID.
  const std::string &get_goal_parent_id_string();

  /// \brief Gets the to-goal edge ID as int.
  /// \return The to-goal edge ID.
  static constexpr int get_to_goal_edge_id_int();

  /// \brief Gets the to-state edge ID as int.
  /// \return The to-state edge ID.
  static constexpr int get_to_state_edge_id_int();

  /// \brief Gets the epsilon node ID as int.
  /// \return The epsilon node ID.
  static constexpr int get_epsilon_node_id_int();

  /// \brief Gets the goal parent ID as int.
  /// \return The goal parent ID.
  static constexpr int get_goal_parent_id_int();

  /// \brief Gets the shift state IDs.
  /// \return The shift state IDs.
  int get_shift_state_ids() const;

  /// \brief Gets the goal dot string.
  /// \return The dot string.
  constexpr const std::string &get_goal_string() const;

  /// \brief Gets the goal dot string with non-binary ids forced for BITMASK
  /// representation. \return The dot string.
  constexpr const std::string &get_goal_forced_string() const;

  /**
   * \brief Gets the goal file path.
   * \return The goal file path.
   */
  const std::string &get_goal_file_path() const;

  /**
   * \brief Get unique agent ID from map.
   * \param ag The agent.
   * \return Unique ID.
   */
  size_t get_unique_a_id_from_map(const Agent &ag) const;

  /** \brief Deleted copy constructor (singleton pattern). */
  TrainingDataset(const TrainingDataset &) = delete;

  /** \brief Deleted copy assignment operator (singleton pattern). */
  TrainingDataset &operator=(const TrainingDataset &) = delete;

  /** \brief Deleted move constructor (singleton pattern). */
  TrainingDataset(TrainingDataset &&) = delete;

  /** \brief Deleted move assignment operator (singleton pattern). */
  TrainingDataset &operator=(TrainingDataset &&) = delete;

private:
  /**
   * \brief Private constructor for singleton pattern.
   */
  TrainingDataset();

  // --- Singleton instance ---
  static TrainingDataset *instance; ///< Singleton instance pointer

  // --- Dataset and file management ---
  std::string m_folder;
  ///< Folder for datasets (contains raw files of states,
  ///< csv with all the info and goal file)
  std::string m_training_raw_files_folder; ///< Raw data folder
  std::string m_filepath_csv;              ///< Current dataset csv file path
  std::string m_goal_file_path;            ///< Goal file path

  unsigned long m_file_counter = 0; ///< Counter for dataset files

  /// \brief Mersenne Twister random number generator, seeded with rd.
  std::mt19937_64 m_gen;
  int64_t m_seed = -1; ///< Seed for the random number generator

  // --- Mappings ---
  std::unordered_map<Fluent, size_t>
      m_fluent_to_id; ///< Mapping from fluent to unique ID
  std::unordered_map<Agent, size_t>
      m_agent_to_id; ///< Mapping from agent to unique ID
    std::unordered_map<std::string, size_t>
    m_action_to_id; ///< Mapping from action name to unique ID. This is not for the print in the dot but for the print in the CSV file
  std::string m_goal_string;
  ///< String representation of the goal tree for
  ///< efficient printing
  std::string m_goal_forced_string;
  ///< String representation of the goal tree for
  ///< efficient printing without binary ids in case of bitmask for efficiency
  int m_failed_state = 1000000; ///< Value for the failed exploration

  // --- Node and search statistics ---
  size_t m_current_nodes = 0;                   ///< Current number of nodes
  size_t m_threshold_node_generation = 100000;  ///< Node generation threshold
  size_t m_max_threshold_node_creation = 25000; ///< MAX Node creation threshold
  size_t m_min_threshold_node_creation = 10;    ///< min Node creation threshold
  size_t m_added_to_dataset = 0; ///< Number of nodes added to dataset

  double m_threshold_node_generation_log =
      std::log(m_threshold_node_generation * 3);
  ///< Log threshold for node generation
  double m_total_possible_nodes_log = 0; ///< Log of total possible nodes
  bool m_goal_recently_found = false;    ///< Flag for recent goal finding
  double m_discard_augmentation_factor =
      0;                    ///< Augmentation factor for non-discarded paths
  size_t m_goal_founds = 0; ///< Number of goals found

  // --- State tracking ---
  std::set<State<StateRepr>> m_visited_states;
  ///< Set of visited states \warning cannot use unordered set because I am
  ///< missing a clear way of hashing the state
  std::map<State<StateRepr>, int> m_states_scores; ///< State scores

  /// \brief Integer edge ID for goal connection in merged graph.
  static constexpr int m_to_goal_edge_id_int = 2;
  /// \brief String edge ID for goal connection in merged graph.
  std::string m_to_goal_edge_id = "2";
  /// \brief Integer edge ID for state connection in merged graph.
  static constexpr int m_to_state_edge_id_int = 3;
  /// \brief String edge ID for state connection in merged graph.
  std::string m_to_state_edge_id = "3";
  /// \brief Integer node ID that connects the two sub-graphs (goal and state)
  /// in the merged graph.
  static constexpr int m_epsilon_node_id_int = 0;
  /// \brief String node ID that connects the two sub-graphs (goal and state) in
  /// the merged graph.
  std::string m_epsilon_node_id = "0";
  /// \brief Integer node ID that is the initial parent of the goal graph.
  static constexpr int m_goal_parent_id_int = 1;
  /// \brief String node ID that is the initial parent of the goal graph.
  std::string m_goal_parent_id = "1";
  int m_shift_state_ids = 0;
  ///< Used to shift the ids of the state (especially when mapped) so that there
  ///< is no overlap between the goal and the state (only the agents ID are
  ///< preserved ank kept the same)

  // --- Internal utility functions ---
  /**
   * \brief Get ID from a map given a key.
   * \param id_map The map from key to ID.
   * \param key The key to look up.
   * \param type_name The type name for error reporting.
   * \return The ID associated with the key.
   */
  static size_t get_id_from_map(
      const std::unordered_map<boost::dynamic_bitset<>, size_t> &id_map,
      const boost::dynamic_bitset<> &key, const std::string &type_name);

  /**
   * \brief Populate IDs from a set of keys.
   * \param keys_set Set of keys.
   * \param id_map Map to populate.
   * \param start_id Starting ID.
   */
  static void populate_ids_from_bitset(
      const std::set<boost::dynamic_bitset<>> &keys_set,
      std::unordered_map<boost::dynamic_bitset<>, size_t> &id_map,
      size_t start_id);

  /**
   * \brief Get unique fluent ID from map.
   * \param fl The fluent.
   * \return Unique ID.
   */
  size_t get_unique_f_id_from_map(const Fluent &fl) const;

  /**
   * \brief Populate fluent IDs.
   * \param start_id Starting ID.
   */
  void populate_fluent_ids(size_t start_id);

  /**
   * \brief Populate agent IDs.
   * \param start_id Starting ID.
   */
  void populate_agent_ids(size_t start_id);

  /**
   * \brief Generate the goal tree.
   */
  void print_goal_tree() const;

  /**
   * \brief Generate the goal tree subgraph to store in m_goal_string (efficient
   * for printing).
   * \param force_non_binary_ids If true, forces the creation of another goal
   * where the ids are not bitmask for faster management during tensor
   * conversion \return String representation of the goal tree subgraph.
   */
  void generate_goal_tree_subgraph(bool force_non_binary_ids);

  /**
   * \brief Print the goal subtree.
   * \param to_print Formula to print.
   * \param goal_counter Goal counter.
   * \param next_id Next node ID.
   * \param parent_node Parent node name.
   * \param os Output stream.
   * \param force_non_binary_ids If true, forces the creation of another goal
   * where the ids are not bitmask for faster management during tensor
   * conversion
   */
  void generate_goal_subtree(const BeliefFormula &to_print, size_t goal_counter,
                             size_t &next_id, const std::string &parent_node,
                             std::ostream &os, bool force_non_binary_ids);

  /**
   * \brief Explore the search space.
   * \return True if exploration was successful.
   */
  bool search_space_exploration();

  /**
   * \brief Perform DFS exploration for dataset generation.
   * \param initial_state The initial state.
   * \param actions Set of actions.
   * \return True if successful.
   */
  bool dfs_exploration(State<StateRepr> &initial_state, ActionsSet *actions);

  /**
   * \brief DFS worker for dataset generation.
   * \param state Current state.
   * \param depth Current depth.
   * \param actions Set of actions.
   * \param predecessor
   * \param action
   * \return Score.
   */
  int dfs_worker(State<StateRepr> &state, size_t depth, ActionsSet *actions, const std::string &predecessor, const std::string &action);

  /**
   * \brief Format a row for the dataset and insert it to the dataset itself.
   * \param base_filename
   * \param depth The depth.
   * \param score The score.
   * \param predecessor
   * \param action
   */
  void add_to_dataset(const std::string &base_filename, size_t depth, int score, const std::string &predecessor, const std::string &action);

  /**
   * \brief Print state for dataset.
   * \param state The state.
   * \return String representation.
   */
  std::string print_state_for_dataset(const State<StateRepr> &state);

  /**
   * \brief Format a name for dataset files.
   * \param base_filename Base filename.
   * \return Formatted name.
   */
  std::string format_name(const std::string &base_filename) const;

  /**
   * @brief Generates a unique folder path by appending an incrementing number
   * if needed.
   *
   * Constructs a folder path using the base path and domain name. If the
   * resulting path already exists in the filesystem, appends an underscore and
   * a number (e.g., _1, _2, ...) until a unique path is found.
   *
   * @param base_path The root path where the dataset folder should be created.
   * @param domain_name The base name of the folder (e.g., derived from the
   * dataset name).
   * @return A string representing a unique folder path that does not currently
   * exist, with a trailing slash.
   */
  static std::string make_unique_folder(const std::string &base_path,
                                        const std::string &domain_name);

  /**
   * @brief Retrieves the output path corresponding to the current dataset type.
   *
   * This function inspects the dataset type provided by the ArgumentParser
   * singleton and returns the appropriate output path constant from
   * OutputPaths. If the dataset type is invalid, the function will terminate
   * the program with an error message via ExitHandler.
   *
   * @return std::string Path associated with the selected dataset type.
   */
  std::string create_complete_path() const;

  /// \brief Converts an integer to a binary string.
  /// \param value Integer value.
  /// \param force_non_binary_ids If true, forces the creation of another goal
  /// where the ids are not bitmask for faster management during tensor
  /// conversion \return Binary string of length bit_width.
  static std::string to_binary_string(bool force_non_binary_ids, size_t value);

  /// \brief Converts an integer (as string) to a binary string.
  /// \param str_value Integer value expressed as a string.
  /// /// \param force_non_binary_ids If true, forces the creation of another
  /// goal where the ids are not bitmask for faster management during tensor
  /// conversion \return Binary string of length bit_width.
  static std::string to_binary_string(bool force_non_binary_ids,
                                      const std::string &str_value);

  /// \brief Updates all string IDs with GOAL_ENCODING_BITS-bit binary
  /// representations.
  void update_binary_ids();
};

#include "TrainingDataset.tpp"
