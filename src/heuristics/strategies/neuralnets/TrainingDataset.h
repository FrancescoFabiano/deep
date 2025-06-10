#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <set>
#include <cmath>
#include "Define.h"
#include "State.h"

/// \brief Random device for seeding the random number generator.
inline std::random_device rd;

/// \brief Mersenne Twister random number generator, seeded with rd.
inline std::mt19937 m_gen(rd());

/// \brief Uniform real distribution in the range [0.0, 1.0).
inline std::uniform_real_distribution<> m_dis(0.0, 1.0);

/**
 * \class TrainingDataset
 * \brief Singleton class for Neural Network-based heuristic dataset generation and management.
 *
 * This class is responsible for generating datasets for ML heuristics, managing state and agent mappings,
 * and providing utility functions for dataset formatting and exploration. It is templated on the StateRepresentation.
 *
 * \tparam StateRepr The state representation type.
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 31, 2025
 */
template<StateRepresentation StateRepr>
class TrainingDataset {
public:
    /**
     * \brief Get the singleton instance of GraphNN.
     * \return Reference to the singleton instance.
     */
    static TrainingDataset &get_instance();

    /**
     * \brief Create the singleton instance of GraphNN.
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
    std::string m_folder; ///< Folder for datasets (contains raw files of states, csv with all the info and goal file)
    std::string m_training_raw_files_folder; ///< Raw data folder
    std::string m_filepath_csv; ///< Current dataset csv file path
    std::string m_goal_file_path; ///< Goal file path

    unsigned long m_file_counter = 0; ///< Counter for dataset files

    // --- Mappings ---
    std::unordered_map<Fluent, size_t> m_fluent_to_id; ///< Mapping from fluent to unique ID
    std::unordered_map<Agent, size_t> m_agent_to_id; ///< Mapping from agent to unique ID

    // --- Node and search statistics ---
    size_t m_current_nodes = 0; ///< Current number of nodes
    size_t m_threshold_node_generation = 500; ///< Node generation threshold
    double m_threshold_node_generation_log = std::log(50000 * 3); ///< Log threshold for node generation
    double m_total_possible_nodes_log = 0; ///< Log of total possible nodes
    bool m_goal_recently_found = false; ///< Flag for recent goal finding
    double m_discard_augmentation_factor = 0; ///< Augmentation factor for non-discarded paths
    size_t m_goal_founds = 0; ///< Number of goals found

    // --- State tracking ---
    std::set<State<StateRepr> > m_visited_states;
    ///< Set of visited states \warning cannot use unordered set because I am missing a clear way of hashing the state
    std::map<State<StateRepr>, int> m_states_scores; ///< State scores

    // --- Internal utility functions ---
    /**
     * \brief Get ID from a map given a key.
     * \param id_map The map from key to ID.
     * \param key The key to look up.
     * \param type_name The type name for error reporting.
     * \return The ID associated with the key.
     */
    static size_t get_id_from_map(const std::unordered_map<boost::dynamic_bitset<>, size_t> &id_map,
                                  const boost::dynamic_bitset<> &key,
                                  const std::string &type_name);

    /**
     * \brief Populate IDs from a set of keys.
     * \param keys_set Set of keys.
     * \param id_map Map to populate.
     * \param start_id Starting ID.
     */
    static void populate_ids_from_bitset(const std::set<boost::dynamic_bitset<> > &keys_set,
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
    void generate_goal_tree();

    /**
     * \brief Print the goal subtree.
     * \param to_print Formula to print.
     * \param goal_counter Goal counter.
     * \param next_id Next node ID.
     * \param parent_node Parent node name.
     * \param ofs Output file stream.
     */
    void print_goal_subtree(const BeliefFormula &to_print, size_t goal_counter, size_t &next_id,
                            const std::string &parent_node, std::ofstream &ofs);

    /**
     * \brief Explore the search space.
     * \return True if exploration was successful.
     */
    bool search_space_exploration();

    /**
     * \brief Perform DFS exploration for dataset generation.
     * \param initial_state The initial state.
     * \param actions Set of actions.
     * \param global_dataset Dataset vector.
     * \return True if successful.
     */
    bool dfs_exploration(State<StateRepr> &initial_state, ActionsSet *actions,
                         std::vector<std::string> &global_dataset);

    /**
     * \brief DFS worker for dataset generation.
     * \param state Current state.
     * \param depth Current depth.
     * \param actions Set of actions.
     * \param global_dataset Dataset vector.
     * \return Score.
     */
    int dfs_worker(State<StateRepr> &state, size_t depth, ActionsSet *actions,
                   std::vector<std::string> &global_dataset);

    /**
     * \brief Format a row for the dataset.
     * \param state The state.
     * \param depth The depth.
     * \param score The score.
     * \return Formatted string.
     */
    std::string format_row(const State<StateRepr> &state, size_t depth, int score);

    /**
     * \brief Print state for dataset.
     * \param state The state.
     * \return String representation.
     */
    std::string print_state_for_dataset(const State<StateRepr> &state);

    /**
     * \brief Internal print for dataset.
     * \param state The state.
     * \param base_filename Base filename.
     * \param type Type string.
     */
    void print_state_for_dataset_internal(const State<StateRepr> &state, const std::string &base_filename,
                                          const std::string &type) const;

    /**
     * \brief Format a name for dataset files.
     * \param base_filename Base filename.
     * \param type Type string.
     * \return Formatted name.
     */
    std::string format_name(const std::string &base_filename, const std::string &type) const;
};

#include "TrainingDataset.tpp"
