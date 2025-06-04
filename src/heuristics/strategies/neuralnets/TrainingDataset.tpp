#include "TrainingDataset.h"
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <filesystem>
#include "ArgumentParser.h"
#include "Configuration.h"
#include "Domain.h"
#include "ExitHandler.h"

// --- Singleton implementation ---
template <StateRepresentation StateRepr>
TrainingDataset<StateRepr>& TrainingDataset<StateRepr>::get_instance()
{
    if (!instance)
    {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::NNInstanceError,
            "GNN instance not created. Call create_instance() first."
        );
        //Just to please the compiler
        std::exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));
    }
    return *instance;
}

template <StateRepresentation StateRepr>
const std::string& TrainingDataset<StateRepr>::get_folder() const
{
    return m_folder;
}


template <StateRepresentation StateRepr>
TrainingDataset<StateRepr>::TrainingDataset()
{
    const std::string domain_name = Domain::get_instance().get_name();

    if (ArgumentParser::get_instance().get_dataset_mode())
    {
        m_folder = OutputPaths::DATASET_TRAINING_FOLDER + domain_name + "/";
        m_training_raw_files_folder = m_folder + "RawFiles/";


        const std::string filename = domain_name + "_depth_" + std::to_string(
            ArgumentParser::get_instance().get_dataset_depth()) + ".csv";

        m_filepath_csv = m_folder + filename;
        // Use std::filesystem for directory creation (C++17+)
        try
        {
            std::filesystem::create_directories(m_folder);
            std::filesystem::create_directories(m_training_raw_files_folder);
            if (ArgumentParser::get_instance().get_dataset_mapped() && !ArgumentParser::get_instance().
                get_dataset_both())
            {
                std::filesystem::create_directories(
                    m_training_raw_files_folder + OutputPaths::DATASET_NN_DATASET_MAPPED + "/");
            }
            if (!ArgumentParser::get_instance().get_dataset_mapped() && !ArgumentParser::get_instance().
                get_dataset_both())
            {
                std::filesystem::create_directories(
                    m_training_raw_files_folder + OutputPaths::DATASET_NN_DATASET_HASHED + "/");
            }
        }
        catch (const std::filesystem::filesystem_error& e)
        {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::NNDirectoryCreationError,
                std::string("Error creating directories: ") + e.what()
            );
        }
    }
    else
    {
        m_folder = OutputPaths::DATASET_INFERENCE_FOLDER + domain_name + "/";
        try
        {
            std::filesystem::create_directories(m_folder);
        }
        catch (const std::filesystem::filesystem_error& e)
        {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::NNDirectoryCreationError,
                std::string("Error creating directories: ") + e.what()
            );
        }
    }


    m_goal_file_path = m_folder + "goal_tree.dot";

    populate_agent_ids(0);
    populate_fluent_ids(static_cast<int>(m_agent_to_id.size()) + 1);


    generate_goal_tree();
}

template <StateRepresentation StateRepr>
void TrainingDataset<StateRepr>::create_instance()
{
    if (!instance)
    {
        instance = new TrainingDataset();
    }
}

template <StateRepresentation StateRepr>
bool TrainingDataset<StateRepr>::generate_dataset()
{
    std::ofstream result(m_filepath_csv);
    if (!result.is_open())
    {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::NNTrainingFileError,
            "Error opening file: " + m_filepath_csv
        );
        std::exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));
    }
    result << "Path Hash,Path Mapped,Depth,Distance From Goal,Goal" << std::endl;
    result.close();

    return search_space_exploration();
}


template <StateRepresentation StateRepr>
void TrainingDataset<StateRepr>::generate_goal_tree()
{
    std::ofstream dot_file(m_goal_file_path);
    if (!dot_file.is_open())
    {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::NNTrainingFileError,
            "Error opening file: " + m_goal_file_path
        );
    }

    const auto goal_list = Domain::get_instance().get_goal_description();
    size_t goal_counter = 0;
    dot_file << "digraph G {\n";

    size_t next_id = m_fluent_to_id.size() + m_agent_to_id.size() + 2;
    const std::string parent_name = "-1";
    for (const auto& goal : goal_list)
    {
        print_goal_subtree(goal, ++goal_counter, next_id, parent_name, dot_file);
    }

    dot_file << "}\n";
    dot_file.close();
}

template <StateRepresentation StateRepr>
size_t TrainingDataset<StateRepr>::get_id_from_map(const std::unordered_map<boost::dynamic_bitset<>, size_t>& id_map,
                                                   const boost::dynamic_bitset<>& key, const std::string& type_name)
{
    if (const auto it = id_map.find(key); it != id_map.end())
    {
        return it->second;
    }
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::NNMappingError,
        "Error accessing a key in " + type_name + " map. Key not found."
    );
    // Jut to please the compiler
    std::exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));
}

template <StateRepresentation StateRepr>
void TrainingDataset<StateRepr>::populate_ids_from_bitset(const std::set<boost::dynamic_bitset<>>& keys_set,
                                                          std::unordered_map<boost::dynamic_bitset<>, size_t>& id_map,
                                                          size_t start_id)
{
    size_t current_id = start_id;
    for (const auto& key : keys_set)
    {
        id_map[key] = current_id++;
    }
}

template <StateRepresentation StateRepr>
size_t TrainingDataset<StateRepr>::get_unique_f_id_from_map(const Fluent& fl) const
{
    return get_id_from_map(m_fluent_to_id, fl, "Fluent");
}

template <StateRepresentation StateRepr>
size_t TrainingDataset<StateRepr>::get_unique_a_id_from_map(const Agent& ag) const
{
    return get_id_from_map(m_agent_to_id, ag, "Agent");
}

template <StateRepresentation StateRepr>
void TrainingDataset<StateRepr>::populate_fluent_ids(const size_t start_id)
{
    populate_ids_from_bitset(Domain::get_instance().get_fluents(), m_fluent_to_id, start_id);
}

template <StateRepresentation StateRepr>
void TrainingDataset<StateRepr>::populate_agent_ids(const size_t start_id)
{
    populate_ids_from_bitset(Domain::get_instance().get_agents(), m_agent_to_id, start_id);
}

template <StateRepresentation StateRepr>
void TrainingDataset<StateRepr>::print_goal_subtree(const BeliefFormula& to_print, const size_t goal_counter,
                                                    size_t& next_id,
                                                    const std::string& parent_node, std::ofstream& ofs)
{
    size_t current_node_id = ++next_id;
    std::string node_name;

    switch (to_print.get_formula_type())
    {
    case BeliefFormulaType::FLUENT_FORMULA:
        {
            std::string m_parent_node = parent_node;
            if (to_print.get_fluent_formula().size() > 1)
            {
                //REMOVE LETTERS node_name = "F_OR" + std::to_string(current_node_id);
                node_name = std::to_string(current_node_id);
                current_node_id = ++next_id;
                //ofs << "  " << node_name << " [label=\"" << current_node_id << "\"];\n";
                ofs << "  " << parent_node << " -> " << node_name << " [label=\"" << goal_counter << "\"];\n";
                m_parent_node = node_name;
            }

            for (const auto& fls_set : to_print.get_fluent_formula())
            {
                std::string m_m_parent_node = m_parent_node;

                if (fls_set.size() > 1)
                {
                    //REMOVE LETTERS node_name = "F_AND" + std::to_string(current_node_id);
                    node_name = std::to_string(current_node_id);
                    current_node_id = ++next_id;
                    //ofs << "  " << node_name << " [label=\"" << current_node_id << "\"];\n";
                    ofs << "  " << m_parent_node << " -> " << node_name << " [label=\"" << goal_counter <<
                        "\"];\n";
                    m_m_parent_node = node_name;
                }

                for (const auto& fl : fls_set)
                {
                    //REMOVE LETTERS ofs << "  " << m_m_parent_node << " -> F" << get_unique_f_id_from_map(fl) << " [label=\"" << goal_counter << "\"];\n";
                    ofs << "  " << m_m_parent_node << " -> " << get_unique_f_id_from_map(fl) << " [label=\"" <<
                        goal_counter << "\"];\n";
                }
            }
            break;
        }

    case BeliefFormulaType::BELIEF_FORMULA:
        {
            //REMOVE LETTERS node_name = "B" + std::to_string(current_node_id);
            node_name = std::to_string(current_node_id);
            //ofs << "  " << node_name << " [label=\"" << current_node_id << "\"];\n";
            ofs << "  " << parent_node << " -> " << node_name << " [label=\"" << goal_counter << "\"];\n";
            //REMOVE LETTERS ofs << "  " << node_name << " -> A" << get_unique_a_id_from_map(to_print.get_agent()) << " [label=\"" << goal_counter << "\"];\n";
            //REMOVE LETTERS ofs << "  A" << get_unique_a_id_from_map(to_print.get_agent()) << " -> " << node_name << " [label=\"" << goal_counter << "\"];\n";
            ofs << "  " << node_name << " -> " << get_unique_a_id_from_map(to_print.get_agent()) << " [label=\"" <<
                goal_counter << "\"];\n";
            ofs << "  " << get_unique_a_id_from_map(to_print.get_agent()) << " -> " << node_name << " [label=\"" <<
                goal_counter << "\"];\n";

            print_goal_subtree(to_print.get_bf1(), goal_counter, next_id, node_name, ofs);
            break;
        }

    case BeliefFormulaType::C_FORMULA:
        {
            //REMOVE LETTERS node_name = "C" + std::to_string(current_node_id);
            node_name = std::to_string(current_node_id);
            //ofs << "  " << node_name << " [label=\"" << current_node_id << "\"];\n";
            ofs << "  " << parent_node << " -> " << node_name << " [label=\"" << goal_counter << "\"];\n";

            for (const auto& ag : to_print.get_group_agents())
            {
                //REMOVE LETTERS ofs << "  " << node_name << " -> A" << get_unique_a_id_from_map(ag) << " [label=\"" << goal_counter << "\"];\n";
                //REMOVE LETTERS ofs << "  A" << get_unique_a_id_from_map(ag) << " -> " << node_name << " [label=\"" << goal_counter << "\"];\n";
                ofs << "  " << node_name << " -> " << get_unique_a_id_from_map(ag) << " [label=\"" << goal_counter
                    << "\"];\n";
                ofs << "  " << get_unique_a_id_from_map(ag) << " -> " << node_name << " [label=\"" << goal_counter
                    << "\"];\n";
            }

            print_goal_subtree(to_print.get_bf1(), goal_counter, next_id, node_name, ofs);
            break;
        }

    case BeliefFormulaType::PROPOSITIONAL_FORMULA:
        {
            switch (to_print.get_operator())
            {
            case BeliefFormulaOperator::BF_NOT:
            case BeliefFormulaOperator::BF_AND:
            case BeliefFormulaOperator::BF_OR:
                {
                    break;
                }
            case BeliefFormulaOperator::BF_FAIL:
            default:
                {
                    ExitHandler::exit_with_message(
                        ExitHandler::ExitCode::BeliefFormulaOperatorUnset,
                        "Error in reading a Belief Formula during the GOAL dot generation."
                    );
                    break;
                }
            }

            //REMOVE LETTERS node_name = node_name + std::to_string(current_node_id);
            node_name = std::to_string(current_node_id);
            //ofs << "  " << node_name << " [label=\"" << current_node_id << "\"];\n";
            ofs << "  " << parent_node << " -> " << node_name << " [label=\"" << goal_counter << "\"];\n";
            print_goal_subtree(to_print.get_bf1(), goal_counter, next_id, node_name, ofs);

            if (!to_print.is_bf2_null())
            {
                print_goal_subtree(to_print.get_bf2(), goal_counter, next_id, node_name, ofs);
            }

            break;
        }

    case BeliefFormulaType::BF_EMPTY:
    case BeliefFormulaType::BF_TYPE_FAIL:
    default:
        {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::BeliefFormulaTypeUnset,
                "Error in reading a Belief Formula during the GOAL dot generation."
            );
            break;
        }
    }
}

template <StateRepresentation StateRepr>
bool TrainingDataset<StateRepr>::search_space_exploration()
{
    State<StateRepr> initial_state;
    initial_state.build_initial();

    if (Configuration::get_instance().get_bisimulation())
    {
        initial_state.contract_with_bisimulation();
    }

    ActionsSet actions = Domain::get_instance().get_actions();

    auto start_time = std::chrono::system_clock::now();

    std::vector<std::string> global_dataset;
    global_dataset.reserve(m_threshold_node_generation);

    const bool result = dfs_exploration(initial_state, &actions, global_dataset);

    auto end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    auto& os = ArgumentParser::get_instance().get_output_stream();
    os << "\nDataset Generated in " << elapsed.count() << " seconds" << std::endl;

    std::ofstream result_file(m_filepath_csv, std::ofstream::app);
    for (const auto& row : global_dataset)
    {
        result_file << row << "\n";
    }
    result_file.close();

    return result;
}

template <StateRepresentation StateRepr>
bool TrainingDataset<StateRepr>::dfs_exploration(State<StateRepr>& initial_state, ActionsSet* actions,
                                                 std::vector<std::string>& global_dataset)
{
    auto max_depth = ArgumentParser::get_instance().get_dataset_depth();
    m_visited_states.clear();

    size_t branching_factor = actions->size();
    if (branching_factor <= 1)
    {
        m_total_possible_nodes_log = log(max_depth + 1);
    }
    else
    {
        // Calculate expected log of total nodes
        double numerator_log = (max_depth + 1) * std::log(branching_factor);
        double denominator_log = std::log(branching_factor - 1);
        m_total_possible_nodes_log = numerator_log - denominator_log;
    }

    auto& os = ArgumentParser::get_instance().get_output_stream();

    os << "Total possible nodes exceed threshold." << std::endl;
    os << "Approximate number of nodes (exp(log)) = " << std::exp(m_total_possible_nodes_log) << std::endl;
    os << "Threshold number of nodes = " << m_threshold_node_generation << std::endl;
    if (m_total_possible_nodes_log > m_threshold_node_generation_log)
    {
        os << "Decision: using SPARSE DFS." << std::endl;
    }
    else
    {
        os << "Decision: using COMPLETE DFS." << std::endl;
    }

    dfs_worker(initial_state, 0, actions, global_dataset);

    if (m_goal_founds > 0)
    {
        os << "Number of goals found: " << m_goal_founds << std::endl;
    }
    else
    {
        os << "[WARNING] No goals found, this is not a good training set (recreate it)." << std::endl;
    }

    return !global_dataset.empty();
}

template <StateRepresentation StateRepr>
int TrainingDataset<StateRepr>::dfs_worker(State<StateRepr>& state, const size_t depth, ActionsSet* actions,
                                           std::vector<std::string>& global_dataset)
{
    if (m_current_nodes >= m_threshold_node_generation)
    {
        if (state.is_goal())
        {
            global_dataset.push_back(format_row(state, depth, 0));
            return 0;
        }
        return -1;
    }
    m_current_nodes++;


    int current_score = -1;

    if (m_visited_states.count(state))
    {
        return m_states_scores[state];
    }

    if (state.is_goal())
    {
        current_score = 0;
        m_goal_founds++;
        m_goal_recently_found = true;
    }

    int best_successor_score = -1;
    bool has_successor = false;
    size_t max_depth = static_cast<size_t>(ArgumentParser::get_instance().get_dataset_depth());
    // Create a local vector from action_set
    std::vector<Action> local_actions(actions->begin(), actions->end());

    // Shuffle local_actions
    std::ranges::shuffle(local_actions, m_gen);

    for (const auto& action : local_actions)
    {
        if (state.is_executable(action))
        {
            auto next_state = state.compute_successor(action);

            if (Configuration::get_instance().get_bisimulation())
            {
                next_state.contract_with_bisimulation();
            }

            if (depth >= max_depth)
            {
                break;
            }
            else
            {
                double discard_probability = 0.0;
                // Only start discarding if total search space is big
                if (m_total_possible_nodes_log > m_threshold_node_generation_log)
                {
                    const auto depth_ratio = static_cast<double>(depth) / static_cast<double>(max_depth);
                    const auto fullness_ratio = static_cast<double>(m_current_nodes) / static_cast<double>(
                        m_threshold_node_generation);

                    // Start discard_probability low, increase as depth grows
                    discard_probability = 0.2 * std::pow(depth_ratio, 2);

                    // Slightly boost as dataset fills up
                    discard_probability += 0.2 * fullness_ratio;

                    discard_probability += std::min(
                        0.01 * std::pow(static_cast<double>(m_discard_augmentation_factor) / (3 * max_depth), 2), 0.1);

                    // Boost if a nearby goal was found
                    if (m_goal_recently_found)
                    {
                        discard_probability += 0.2;
                    }

                    // Clamp between [0, 0.8] (still allow deeper exploration)
                    discard_probability = std::min(discard_probability, 0.8);
                }

                if (m_dis(m_gen) < discard_probability)
                {
                    m_goal_recently_found = false;
                    m_discard_augmentation_factor = 0.0;
                    continue; // Randomly skip exploration
                }

                // Increase the augmentation factor for non-discarded series
                m_discard_augmentation_factor++;

                int child_score = dfs_worker(next_state, depth + 1, actions, global_dataset);

                if (child_score >= 0)
                {
                    if (!has_successor || child_score < best_successor_score)
                    {
                        best_successor_score = child_score;
                    }
                    has_successor = true;
                }
            }
        }
    }

    if (current_score == -1 && has_successor)
    {
        current_score = best_successor_score + 1;
    }

    global_dataset.push_back(format_row(state, depth, current_score));
    m_visited_states.insert(state);
    m_states_scores[state] = current_score;

    return current_score;
}


template <StateRepresentation StateRepr>
std::string TrainingDataset<StateRepr>::format_row(const State<StateRepr>& state, const size_t depth, const int score)
{
    std::stringstream ss;
    auto base_filename = print_state_for_dataset(state);

    std::string filename_hash = format_name(base_filename, OutputPaths::DATASET_NN_DATASET_HASHED);
    std::string filename_emap = format_name(base_filename, OutputPaths::DATASET_NN_DATASET_MAPPED);

    if (ArgumentParser::get_instance().get_dataset_mapped() && !ArgumentParser::get_instance().get_dataset_both())
    {
        filename_hash = "NOT CALCULATED";
    }
    if (!ArgumentParser::get_instance().get_dataset_mapped() && !ArgumentParser::get_instance().get_dataset_both())
    {
        filename_emap = "NOT CALCULATED";
    }

    ss << filename_hash << "," << filename_emap << "," << depth << "," << score << "," << m_goal_file_path;

    return ss.str();
}

template <StateRepresentation StateRepr>
std::string TrainingDataset<StateRepr>::print_state_for_dataset(const State<StateRepr>& state)
{
    ++m_file_counter;
    std::string base_filename = std::string(6 - std::to_string(m_file_counter).length(), '0') + std::to_string(m_file_counter);

    if (ArgumentParser::get_instance().get_dataset_mapped() || ArgumentParser::get_instance().get_dataset_both())
    {
        print_state_for_dataset_internal(state, base_filename, OutputPaths::DATASET_NN_DATASET_MAPPED);
    }
    if (!ArgumentParser::get_instance().get_dataset_mapped() || ArgumentParser::get_instance().get_dataset_both())
    {
        print_state_for_dataset_internal(state, base_filename, OutputPaths::DATASET_NN_DATASET_HASHED);
    }

    return base_filename;
}

template <StateRepresentation StateRepr>
void TrainingDataset<StateRepr>::print_state_for_dataset_internal(const State<StateRepr>& state,
                                                                  const std::string& base_filename,
                                                                  const std::string& type) const
{
    std::ofstream out(format_name(base_filename, type));
    state.print_dataset_format(out);
    out.close();
}

template <StateRepresentation StateRepr>
std::string TrainingDataset<StateRepr>::format_name(const std::string& base_filename, const std::string& type) const
{
    return m_training_raw_files_folder + type + "/" + base_filename + "_" + type + ".dot";
}