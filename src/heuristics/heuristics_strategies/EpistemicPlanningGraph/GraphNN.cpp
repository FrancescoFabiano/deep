//
// Created by franc on 5/31/2025.
//

#include "GraphNN.h"


template <class T>
bool planner<T>::ML_dataset_creation(ML_Dataset_Params* ML_dataset) {
    std::string folder = "out/ML_HEUR_datasets/";
    folder += (ML_dataset->useDFS) ? "DFS/" : "BFS/";
    system(("mkdir -p " + folder).c_str());

    std::string fname = domain::get_instance().get_name() + "_d_" + std::to_string(ML_dataset->depth) + ".csv";
    std::string fpath = folder + fname;

    std::ofstream result(fpath);
    if (!result.is_open()) {
        std::cerr << "Error opening file: " << fpath << std::endl;
        return false;
    }
    result << "Path Hash,Path Mapped,Depth,Distance From Goal,Goal" << std::endl;
    result.close();

/*     auto goal_list = domain::get_instance().get_goal_description();

    std::stringstream buffer;
    auto original_buf = std::cout.rdbuf(buffer.rdbuf());

    bool first = true;
    for (auto& goal : goal_list) {
        if (!first) std::cout << " AND ";
        first = false;
        goal.print();
    }
    std::cout.rdbuf(original_buf);
    std::string goal_str = buffer.str(); */

	std::string goal_dot_file = domain::get_instance().get_name() + "_goal_tree.dot";
	std::string goal_file_path =  folder + goal_dot_file;
	generate_goal_tree(goal_file_path);

    return dataset_launcher(fpath, ML_dataset->depth, ML_dataset->useDFS, goal_file_path);
}

template <class T>
int planner<T>::get_id_from_map(const std::map<boost::dynamic_bitset<>, int>& id_map, const boost::dynamic_bitset<>& key, const std::string& type_name) {
    auto it = id_map.find(key);
    if (it != id_map.end()) {
        return it->second;
    } else {
        throw std::runtime_error(type_name + " not found in map");
    }
}

template <class T>
void planner<T>::populate_ids_from_bitset(const std::set<boost::dynamic_bitset<>>& keys_set, std::map<boost::dynamic_bitset<>, int>& id_map, int start_id) {
	int current_id = start_id;
	for (const auto& key : keys_set) {
        id_map[key] = current_id++;
    }
}

template <class T>
int planner<T>::get_unique_f_id_from_map(fluent fl) {
    return get_id_from_map(m_fluent_to_id, fl, "Fluent");
}

template <class T>
int planner<T>::get_unique_a_id_from_map(agent ag) {
    return get_id_from_map(m_agent_to_id, ag, "Agent");
}

template <class T>
void planner<T>::populate_fluent_ids(int start_id) {
    populate_ids_from_bitset(domain::get_instance().get_fluents(), m_fluent_to_id, start_id);
}

template <class T>
void planner<T>::populate_agent_ids(int start_id) {
    populate_ids_from_bitset(domain::get_instance().get_agents(), m_agent_to_id, start_id);
}

template <class T>
const std::string & planner<T>::generate_goal_tree(const std::string & goal_file_name) {
	auto goal_list = domain::get_instance().get_goal_description();
	int goal_counter = 0;

    std::ofstream dot_file(goal_file_name);
    if (!dot_file.is_open()) {
        std::cerr << "Unable to open output file.\n";
        exit(1);
    }

    dot_file << "digraph G {\n";

	// Populate fluent and agent ID maps

	populate_fluent_ids(0);
	populate_agent_ids(static_cast<int>(m_fluent_to_id.size())+1);

	int next_id = m_fluent_to_id.size() + m_agent_to_id.size() + 2;

	// Print fluent nodes (only by ID)
	/* for (const auto& [fluent_index, fluent_id] : m_fluent_to_id) {
		dot_file << "  F" << fluent_id << " [label=\"" << fluent_id << "\"];\n";
	}

	// Print agent nodes (only by ID)
	for (const auto& [agent_index, agent_id] : m_agent_to_id) {
		dot_file << "  A" << agent_id << " [label=\"" << agent_id << "\", shape=box];\n";
	}*/

	std::string parent_name = "-1";
	for (auto& goal : goal_list) {
		print_goal_subtree(goal, ++goal_counter, next_id, parent_name,dot_file);
	}


    dot_file << "}\n";
    dot_file.close();

    std::cout << "DOT file 'graph.dot' created.\n";
	return goal_file_name;
}

template <class T>
void planner<T>::print_goal_subtree(const belief_formula & to_print, int goal_counter, int & next_id, const std::string & parent_node, std::ofstream& dot_file) {


	int current_node_id = ++next_id;
	std::string node_name;

	switch (to_print.get_formula_type()) {
		case FLUENT_FORMULA: {
			std::string m_parent_node = parent_node;
			if  (to_print.get_fluent_formula().size() > 1) {
				//REMOVE LETTERS node_name = "F_OR" + std::to_string(current_node_id);
				node_name = std::to_string(current_node_id);
				current_node_id = ++next_id;
				//dot_file << "  " << node_name << " [label=\"" << current_node_id << "\"];\n";
				dot_file << "  " << parent_node << " -> " << node_name << " [label=\"" << goal_counter << "\"];\n";
				m_parent_node = node_name;
			}

			for (const auto& fls_set : to_print.get_fluent_formula()) {

				std::string m_m_parent_node = m_parent_node;

				if  (fls_set.size() > 1) {
					//REMOVE LETTERS node_name = "F_AND" + std::to_string(current_node_id);
					node_name = std::to_string(current_node_id);
					current_node_id = ++next_id;
					//dot_file << "  " << node_name << " [label=\"" << current_node_id << "\"];\n";
					dot_file << "  " << m_parent_node << " -> " << node_name << " [label=\"" << goal_counter << "\"];\n";
					m_m_parent_node = node_name;
				}

				for (const auto& fl : fls_set) {
					//REMOVE LETTERS dot_file << "  " << m_m_parent_node << " -> F" << get_unique_f_id_from_map(fl) << " [label=\"" << goal_counter << "\"];\n";
					dot_file << "  " << m_m_parent_node << " -> " << get_unique_f_id_from_map(fl) << " [label=\"" << goal_counter << "\"];\n";
				}
			}
			break;
		}

		case BELIEF_FORMULA: {
			//REMOVE LETTERS node_name = "B" + std::to_string(current_node_id);
			node_name = std::to_string(current_node_id);
			//dot_file << "  " << node_name << " [label=\"" << current_node_id << "\"];\n";
			dot_file << "  " << parent_node << " -> " << node_name << " [label=\"" << goal_counter << "\"];\n";
			//REMOVE LETTERS dot_file << "  " << node_name << " -> A" << get_unique_a_id_from_map(to_print.get_agent()) << " [label=\"" << goal_counter << "\"];\n";
			//REMOVE LETTERS dot_file << "  A" << get_unique_a_id_from_map(to_print.get_agent()) << " -> " << node_name << " [label=\"" << goal_counter << "\"];\n";
			dot_file << "  " << node_name << " -> " << get_unique_a_id_from_map(to_print.get_agent()) << " [label=\"" << goal_counter << "\"];\n";
			dot_file << "  " << get_unique_a_id_from_map(to_print.get_agent()) << " -> " << node_name << " [label=\"" << goal_counter << "\"];\n";

			print_goal_subtree(to_print.get_bf1(), goal_counter, next_id, node_name,dot_file);
			break;
		}

		case C_FORMULA: {
			//REMOVE LETTERS node_name = "C" + std::to_string(current_node_id);
			node_name = std::to_string(current_node_id);
			//dot_file << "  " << node_name << " [label=\"" << current_node_id << "\"];\n";
			dot_file << "  " << parent_node << " -> " << node_name << " [label=\"" << goal_counter << "\"];\n";

			for (const auto& ag : to_print.get_group_agents()) {
				//REMOVE LETTERS dot_file << "  " << node_name << " -> A" << get_unique_a_id_from_map(ag) << " [label=\"" << goal_counter << "\"];\n";
				//REMOVE LETTERS dot_file << "  A" << get_unique_a_id_from_map(ag) << " -> " << node_name << " [label=\"" << goal_counter << "\"];\n";
				dot_file << "  " << node_name << " -> " << get_unique_a_id_from_map(ag) << " [label=\"" << goal_counter << "\"];\n";
				dot_file << "  " << get_unique_a_id_from_map(ag) << " -> " << node_name << " [label=\"" << goal_counter << "\"];\n";
			}

			print_goal_subtree(to_print.get_bf1(), goal_counter, next_id, node_name,dot_file);
			break;
		}

		case PROPOSITIONAL_FORMULA: {
			switch (to_print.get_operator()) {
				case BF_NOT: {
					node_name = "NOT";
					break;
				}

				case BF_AND: {
					node_name = "AND";
					break;
				}

				case BF_OR: {
					node_name = "OR";
					break;
				}

				case BF_FAIL:
				default: {
					std::cerr << "\n ERROR IN DECLARATION\n.";
					exit(1);
					break;
				}
			}

			/* if (m_node_printed.find(node_name) == m_node_printed.end() || m_node_printed[node_name] == false){
				dot_file << "  " << node_name << " [label=\"" << m_special_nodes[node_name] << "\"];\n";
				m_node_printed[node_name] = true;
			} */

			//REMOVE LETTERS node_name = node_name + std::to_string(current_node_id);
			node_name = std::to_string(current_node_id);
			//dot_file << "  " << node_name << " [label=\"" << current_node_id << "\"];\n";
			dot_file << "  " << parent_node << " -> " << node_name << " [label=\"" << goal_counter << "\"];\n";
			print_goal_subtree(to_print.get_bf1(), goal_counter, next_id, node_name,dot_file);

			if (!to_print.is_bf2_null()) {
				print_goal_subtree(to_print.get_bf2(), goal_counter, next_id, node_name,dot_file);
			}

			break;
		}

		case BF_EMPTY:
		case BF_TYPE_FAIL:
		default: {
			std::cerr << "\n Unknown belief_formula type.";
			exit(1);
			break;
		}
	}
}

template <class T>
bool planner<T>::dataset_launcher(const std::string& fpath, int max_depth, bool useDFS, const std::string& goal_str) {
    T initial;
    initial.build_initial();

    bool bisimulation = (domain::get_instance().get_bisimulation() != BIS_NONE);

    if (bisimulation) {
        initial.calc_min_bisimilar();
    }

    action_set actions = domain::get_instance().get_actions();

    auto start_time = std::chrono::system_clock::now();

    std::vector<std::string> global_dataset;
    global_dataset.reserve(m_threshold_node_generation_ML);

    bool result;
    if (useDFS) {
        result = dataset_DFS_serial(initial, max_depth, &actions, goal_str, global_dataset, bisimulation);
    } else {
        std::cerr << "Recursion through BFS is not implemented yet for ML dataset generation.\n";
        exit(1);
    }

    auto end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "\nDataset Generated in " << elapsed.count() << " seconds" << std::endl;

    std::ofstream result_file(fpath, std::ofstream::app);
    for (const auto& row : global_dataset) {
        result_file << row << "\n";
    }
    result_file.close();

    return result;
}


template <class T>
bool planner<T>::dataset_DFS_serial(T& initial_state, int max_depth, action_set* actions, const std::string& goal_str, std::vector<std::string>& global_dataset, bool bisimulation) {

    m_visited_states_ML.clear();

    int branching_factor = actions->size();
	if (branching_factor <= 1) {
        m_total_possible_nodes_log_ML = log(max_depth+1);
    }
	else{
		// Calculate expected log of total nodes
		double numerator_log = (max_depth + 1) * std::log(branching_factor);
		double denominator_log = std::log(branching_factor - 1);
		m_total_possible_nodes_log_ML = numerator_log - denominator_log;
	}


	std::cout << "Total possible nodes exceed threshold." << std::endl;
	std::cout << "Approximate number of nodes (exp(log)) = " << std::exp(m_total_possible_nodes_log_ML) << std::endl;
	std::cout << "Threshold number of nodes = " << m_threshold_node_generation_ML << std::endl;
	if (m_total_possible_nodes_log_ML > m_threshold_node_generation_log_ML) {
		std::cout << "Decision: using SPARSE DFS." << std::endl;
	} else {
		std::cout << "Decision: using COMPLETE DFS." << std::endl;
	}

    dataset_DFS_worker(initial_state, 0, max_depth, actions, goal_str, global_dataset, bisimulation);

	if (m_goal_founds_ML > 0) {
		std::cout << "Number of goals found: "<< m_goal_founds_ML << std::endl;
	} else {
		std::cout << "No goals found, this is not a good training set (recreate it)." << std::endl;
	}

    return !global_dataset.empty();
}

template <class T>
int planner<T>::dataset_DFS_worker(T& state, int depth, int max_depth, action_set* actions, const std::string& goal_str, std::vector<std::string>& global_dataset, bool bisimulation) {

    if (m_current_nodes_ML >= m_threshold_node_generation_ML) {
		if (state.is_goal()) {
			global_dataset.push_back(format_row(state, depth, 0, goal_str));
        	return 0;
		}
        return -1;
    }
	m_current_nodes_ML++;


    int current_score = -1;

    if (m_visited_states_ML.count(state)) {
        return m_states_scores[state];
    }

    if (state.is_goal()) {
        current_score = 0;
		m_goal_founds_ML++;
		m_goal_recently_found_ML = true;
    }

    int best_successor_score = -1;
    bool has_successor = false;

	// Create a local vector from action_set
	std::vector<action> local_actions(actions->begin(), actions->end());

	// Shuffle local_actions
	std::shuffle(local_actions.begin(), local_actions.end(), gen);

	for (const auto& action : local_actions) {
		if (state.is_executable(action)) {
			T next_state = state.compute_succ(action);

			if (bisimulation) {
				next_state.calc_min_bisimilar();
			}

			if (depth >= max_depth) {
				break;
			} else {
				double discard_probability = 0.0;
				// Only start discarding if total search space is big
				if (m_total_possible_nodes_log_ML > m_threshold_node_generation_log_ML) {

					double depth_ratio = (double)depth / (double)max_depth;
					double fullness_ratio = (double)m_current_nodes_ML / (double)m_threshold_node_generation_ML;

					// Start discard_probability low, increase as depth grows
					discard_probability = 0.2 * std::pow(depth_ratio, 2);

					// Slightly boost as dataset fills up
					discard_probability += 0.2 * fullness_ratio;

					discard_probability += std::min(0.01 * std::pow(static_cast<double>(m_discard_augmentation_factor_ML) / (3*max_depth), 2), 0.1);

					// Boost if a nearby goal was found
					if (m_goal_recently_found_ML) {
						discard_probability += 0.2;
					}

					// Clamp between [0, 0.8] (still allow deeper exploration)
					discard_probability = std::min(discard_probability, 0.8);
				}

				if (dis(gen) < discard_probability) {
					m_goal_recently_found_ML = false;
					m_discard_augmentation_factor_ML = 0.0;
					continue; // Randomly skip exploration
				}

				// Increase the augmentation factor for non-discarded series
				m_discard_augmentation_factor_ML++;

				int child_score = dataset_DFS_worker(next_state, depth + 1, max_depth, actions, goal_str, global_dataset, bisimulation);

				if (child_score >= 0) {
					if (!has_successor || child_score < best_successor_score) {
						best_successor_score = child_score;
					}
					has_successor = true;
				}
			}
		}
	}

    if (current_score == -1 && has_successor) {
        current_score = best_successor_score + 1;
    }

    global_dataset.push_back(format_row(state, depth, current_score, goal_str));
    m_visited_states_ML.insert(state);
    m_states_scores[state] = current_score;

    return current_score;
}


template <class T>
std::string planner<T>::format_row(T& state, int depth, int score, const std::string& goal_str) {
    std::stringstream ss;
    //ss << "\"";
    //printer::get_instance().print_list(state.get_executed_actions());
    //ss << "\",";
	auto [folder, base_filename] = state.print_graphviz_ML_dataset("");

	std::string folder_hash = folder + "/hash/";
	std::string folder_emap = folder + "/emap/";

	std::string filename_hash = folder_hash + base_filename + "_hash.dot";
	std::string filename_emap = folder_emap + base_filename + "_emap.dot";

	if (!domain::get_instance().is_gnn_mapped_enabled() && !domain::get_instance().is_gnn_both_enabled()){
		filename_emap = "NOT CALCULATED";
	}
	if (domain::get_instance().is_gnn_mapped_enabled() && !domain::get_instance().is_gnn_both_enabled()){
		filename_hash = "NOT CALCULATED";
	}

	ss << filename_hash << "," << filename_emap << "," << depth << "," << score << "," << goal_str;

    return ss.str();
}


struct WorldHash {
    std::size_t operator()(const std::pair<std::set<fluent>, int>& p) const {
        std::size_t seed = 0;
        for (const auto& f : p.first) {
            boost::hash_combine(seed, boost::hash_value(f));
        }
        boost::hash_combine(seed, std::hash<int>{}(p.second));
        return seed;
    }
};


std::string to_base36(int num) {
    /*const char* digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    if (num == 0) return "0";
    std::string result;
    while (num > 0) {
        result = digits[num % 36] + result;
        num /= 36;
    } For now we keep the information in integer for better compatibility with GNNs*/
    return std::to_string(num + domain::get_instance().get_agents().size());
}

int agent_to_int(const agent& ag) {
    try {
        // If agent is boost::dynamic_bitset<>
        unsigned long val = ag.to_ulong();
        if (val > static_cast<unsigned long>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("Agent value exceeds int max");
        }
        return static_cast<int>(val);
    } catch (const std::overflow_error& e) {
        std::cerr << "Error: agent bitset too large to fit in int (" << e.what() << ")\n";
        exit(1);
    }
}

void pstate::write_graphviz_dataset(std::ostream& out, bool use_hash) const
{
    std::unordered_map<std::size_t, int> world_map;
    int world_counter = 1;

    // Assign compact IDs
    for (const auto& pw : get_worlds()) {
        auto key = std::make_pair(pw.get_fluent_set(), pw.get_repetition());
        std::size_t hash = WorldHash{}(key);
        if (world_map.find(hash) == world_map.end()) {
            world_map[hash] = world_counter++;
        }
    }

    out << "digraph G {" << std::endl;

    // Print nodes
    for (const auto& [hash, id] : world_map) {
        out << (use_hash ? std::to_string(hash) : to_base36(id)) << ";" << std::endl;
    }

    // Pointed world
    auto pointed_key = std::make_pair(get_pointed().get_fluent_set(), get_pointed().get_repetition());
    std::size_t pointed_hash = WorldHash{}(pointed_key);
    out << (use_hash ? std::to_string(pointed_hash) : to_base36(world_map[pointed_hash]))
        << " [shape=doublecircle];" << std::endl;

    // Edges
    std::map<std::pair<std::size_t, std::size_t>, std::set<agent>> edge_map;
    for (const auto& [from_pw, from_map] : get_beliefs()) {
        for (const auto& [ag, to_set] : from_map) {
            for (const auto& to_pw : to_set) {
                auto from_key = std::make_pair(from_pw.get_fluent_set(), from_pw.get_repetition());
                auto to_key = std::make_pair(to_pw.get_fluent_set(), to_pw.get_repetition());

                std::size_t from_hash = WorldHash{}(from_key);
                std::size_t to_hash = WorldHash{}(to_key);

                edge_map[{from_hash, to_hash}].insert(ag);
            }
        }
    }

/*     for (const auto& [edge, agents] : edge_map) {
        auto from_label = use_hash ? std::to_string(edge.first) : to_base36(world_map[edge.first]);
        auto to_label = use_hash ? std::to_string(edge.second) : to_base36(world_map[edge.second]);

        out << from_label << " -> " << to_label << " [label=\"";
        bool first = true;
        for (const auto& ag : agents) {
            if (!first) out << ",";
            out << domain::get_instance().get_grounder().deground_agent(ag);
            first = false;
        }
        out << "\"];" << std::endl;
    } */

	//One edge per agent
	for (const auto& [edge, agents] : edge_map) {
    auto from_label = use_hash ? std::to_string(edge.first) : to_base36(world_map[edge.first]);
    auto to_label = use_hash ? std::to_string(edge.second) : to_base36(world_map[edge.second]);

    for (const auto& ag : agents) {
        out << from_label << " -> " << to_label << " [label=\""
            << agent_to_int(ag)
            << "\"];" << std::endl;
		}
	}

    out << "}" << std::endl;
}


void pstate::print_ML_dataset_emap(const std::string& folder, const std::string& base_filename) const
{
    std::string folder_emap = folder + "/emap/";
    system(("mkdir -p " + folder_emap).c_str());
    std::string filename_emap = folder_emap + base_filename + "_emap.dot";

    std::ofstream out(filename_emap);
    write_graphviz_dataset(out, false); // false = use emap IDs
    out.close();
}

void pstate::print_ML_dataset_hash(const std::string& folder, const std::string& base_filename) const
{
    std::string folder_hash = folder + "/hash/";
    system(("mkdir -p " + folder_hash).c_str());
    std::string filename_hash = folder_hash + base_filename + "_hash.dot";

    std::ofstream out(filename_hash);
    write_graphviz_dataset(out, true); // true = use hash values
    out.close();
}

void pstate::print_ML_dataset_dual(const std::string& folder, const std::string& base_filename) const
{

	if (domain::get_instance().is_gnn_mapped_enabled() || domain::get_instance().is_gnn_both_enabled()){
		print_ML_dataset_emap(folder, base_filename);
	}
	if (!domain::get_instance().is_gnn_mapped_enabled() || domain::get_instance().is_gnn_both_enabled()){
		print_ML_dataset_hash(folder, base_filename);
	}
}



void pstate::print_graphviz_explicit(std::ostream & graphviz) const
{
	string_set::const_iterator it_st_set;
	fluent_set::const_iterator it_fs;


	graphviz << "//WORLDS List:" << std::endl;
	std::map<fluent_set, std::string> map_world_to_index;
	std::map<unsigned short, int> map_rep_to_name;
	int found_rep = 0;
	int found_fs = 0;
	fluent_set tmp_fs;
	char tmp_unsh;
	string_set tmp_stset;
	bool print_first;
	pworld_ptr_set::const_iterator it_pwset;
	for (it_pwset = get_worlds().begin(); it_pwset != get_worlds().end(); it_pwset++) {
		if (*it_pwset == get_pointed())
			graphviz << "	node [shape = doublecircle] ";
		else
			graphviz << "	node [shape = circle] ";

		print_first = false;
		tmp_fs = it_pwset->get_fluent_set();
		std::string world_value = "";
		if (map_world_to_index.count(tmp_fs) == 0) {

			tmp_stset = domain::get_instance().get_grounder().deground_fluent(tmp_fs);
			for (it_st_set = tmp_stset.begin(); it_st_set != tmp_stset.end(); it_st_set++) {
				if (print_first) {
					world_value += ",";
				}
				print_first = true;
				world_value += *it_st_set;
			}

			map_world_to_index[tmp_fs] = world_value;
			found_fs++;
		}
		tmp_unsh = it_pwset->get_repetition();
		if (map_rep_to_name.count(tmp_unsh) == 0) {
			map_rep_to_name[tmp_unsh] = found_rep;
			found_rep++;
		}
		graphviz << "\"" << map_rep_to_name[tmp_unsh] << "_" << map_world_to_index[tmp_fs] << "\";";
		graphviz << "// (";
		tmp_stset = domain::get_instance().get_grounder().deground_fluent(tmp_fs);
		for (it_st_set = tmp_stset.begin(); it_st_set != tmp_stset.end(); it_st_set++) {
			if (print_first) {
				graphviz << ",";
			}
			print_first = true;
			graphviz << *it_st_set;
		}
		graphviz << ")\n";
	}

	graphviz << "\n\n";
	graphviz << "//RANKS List:" << std::endl;

	std::map<int, pworld_ptr_set> for_rank_print;
	for (it_pwset = get_worlds().begin(); it_pwset != get_worlds().end(); it_pwset++) {
		for_rank_print[it_pwset->get_repetition()].insert(*it_pwset);
	}

	std::map<int, pworld_ptr_set>::const_iterator it_map_rank;
	for (it_map_rank = for_rank_print.begin(); it_map_rank != for_rank_print.end(); it_map_rank++) {
		graphviz << "	{rank = same; ";
		for (it_pwset = it_map_rank->second.begin(); it_pwset != it_map_rank->second.end(); it_pwset++) {
			graphviz << "\"" << map_rep_to_name[it_pwset->get_repetition()] << "_" << map_world_to_index[it_pwset->get_fluent_set()] << "\"; ";
		}
		graphviz << "}\n";
	}


	graphviz << "\n\n";
	graphviz << "//EDGES List:" << std::endl;

	std::map < std::tuple<std::string, std::string>, std::set<std::string> > edges;

	pworld_transitive_map::const_iterator it_pwtm;
	pworld_map::const_iterator it_pwm;
	std::tuple<std::string, std::string> tmp_tuple;
	std::string tmp_string = "";

	for (it_pwtm = get_beliefs().begin(); it_pwtm != get_beliefs().end(); it_pwtm++) {
		pworld_ptr from = it_pwtm->first;
		pworld_map from_map = it_pwtm->second;

		for (it_pwm = from_map.begin(); it_pwm != from_map.end(); it_pwm++) {
			agent ag = it_pwm->first;
			pworld_ptr_set to_set = it_pwm->second;

			for (it_pwset = to_set.begin(); it_pwset != to_set.end(); it_pwset++) {
				pworld_ptr to = *it_pwset;

				tmp_string = std::to_string(map_rep_to_name[from.get_repetition()]);
				tmp_string += "_" + map_world_to_index[from.get_fluent_set()];
				//tmp_string.insert(0, 1, std::to_string(map_rep_to_name[from.get_repetition()]));
				std::get<0>(tmp_tuple) = tmp_string;

				tmp_string = std::to_string(map_rep_to_name[to.get_repetition()]);
				tmp_string += "_" + map_world_to_index[to.get_fluent_set()];
				//tmp_string.insert(0, 1, map_rep_to_name[to.get_repetition()]);
				std::get<1>(tmp_tuple) = tmp_string;

				edges[tmp_tuple].insert(domain::get_instance().get_grounder().deground_agent(ag));

			}
		}
	}

	std::map < std::tuple<std::string, std::string>, std::set < std::string>>::iterator it_map;
	std::map < std::tuple<std::string, std::string>, std::set < std::string>>::const_iterator it_map_2;

	std::map < std::tuple<std::string, std::string>, std::set < std::string>> to_print_double;
	for (it_map = edges.begin(); it_map != edges.end(); it_map++) {
		for (it_map_2 = it_map; it_map_2 != edges.end(); it_map_2++) {
			if (std::get<0>(it_map->first).compare(std::get<1>(it_map_2->first)) == 0) {
				if (std::get<1>(it_map->first).compare(std::get<0>(it_map_2->first)) == 0) {
					if (it_map->second == it_map_2->second) {
						if (std::get<0>(it_map->first).compare(std::get<1>(it_map->first)) != 0) {
							to_print_double[it_map->first] = it_map->second;
							//std::cerr << std::get<0>(it_map->first) << " " << std::get<0>(it_map_2->first) << "\n";
							it_map_2 = edges.erase(it_map_2);
							it_map = edges.erase(it_map);
						}
					}
				}
			}
		}
	}

	std::set<std::string>::const_iterator it_stset;
	for (it_map = edges.begin(); it_map != edges.end(); it_map++) {
		graphviz << "	\"";
		graphviz << std::get<0>(it_map->first);
		graphviz << "\" -> \"";
		graphviz << std::get<1>(it_map->first);
		graphviz << "\" ";
		graphviz << "[ label = \"";
		tmp_string = "";
		for (it_stset = it_map->second.begin(); it_stset != it_map->second.end(); it_stset++) {
			tmp_string += *it_stset;
			tmp_string += ",";
		}
		tmp_string.pop_back();
		graphviz << tmp_string;
		graphviz << "\" ];\n";
	}

	for (it_map = to_print_double.begin(); it_map != to_print_double.end(); it_map++) {
		graphviz << "	\"";
		graphviz << std::get<0>(it_map->first);
		graphviz << "\" -> \"";
		graphviz << std::get<1>(it_map->first);
		graphviz << "\" ";
		graphviz << "[ dir=both label = \"";
		tmp_string = "";
		for (it_stset = it_map->second.begin(); it_stset != it_map->second.end(); it_stset++) {

			tmp_string += *it_stset;
			tmp_string += ",";
		}
		tmp_string.pop_back();
		graphviz << tmp_string;
		graphviz << "\" ];\n";
	}

}
