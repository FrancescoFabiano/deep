/*
 * \brief Implementation of \ref kstate.h
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date April 9, 2019
 */

#include <iostream>
#include <tuple>
#include <stack>
#include <stdio.h>


#include "kstate.h"
#include "../../domain/domain.h"
#include "../../utilities/helper.h"

void kstate::set_worlds(const kworld_ptr_set & to_set)
{
	m_worlds.clear();
	m_worlds = to_set;
}

void kstate::set_edges(const kedge_ptr_set & to_set)
{
	m_edges.clear();
	m_edges = to_set;
}

void kstate::set_pointed(const kworld_ptr & to_set)
{
	m_pointed = to_set;
}

void kstate::set_max_depth(unsigned int to_set)
{
	if (m_max_depth < to_set) m_max_depth = to_set;
}

const kworld_ptr_set & kstate::get_worlds() const
{
	return m_worlds;
}

const kedge_ptr_set & kstate::get_edges() const
{
	return m_edges;
}

const kworld_ptr & kstate::get_pointed() const
{
	return m_pointed;
}

unsigned int kstate::get_max_depth() const
{
	return m_max_depth;
}

bool kstate::entails(fluent f) const
{
	return entails(f, m_pointed);
}

bool kstate::entails(fluent f, kworld_ptr world) const
{
	return world.get_ptr()->entails(f);
}

bool kstate::entails(const fluent_set & fl) const
{
	return entails(fl, m_pointed);
}

bool kstate::entails(const fluent_set & fl, kworld_ptr world) const
{
	return world.get_ptr()->entails(fl);
}

bool kstate::entails(const fluent_formula & ff) const
{
	return entails(ff, m_pointed);
}

bool kstate::entails(const fluent_formula & ff, kworld_ptr world) const
{
	return world.get_ptr()->entails(ff);
}

bool kstate::entails(const belief_formula & bf) const
{
	return entails(bf, m_pointed);
}

bool kstate::operator=(const kstate & to_copy)
{
	set_edges(to_copy.get_edges());
	set_worlds(to_copy.get_worlds());
	m_max_depth = get_max_depth();
	set_pointed(to_copy.get_pointed());
	return true;
}

bool kstate::operator==(const kstate & to_compare) const
{
	if (m_pointed.get_numerical_id() == to_compare.get_pointed().get_numerical_id()) {
		return true;
	}

	bisimulation b;

	return b.compare_automata_eq(*this, to_compare);
}

bool kstate::operator<(const kstate & to_compare) const
{
//	if (m_max_depth < to_compare.get_max_depth()) {
//		return true;
//	} else if (m_max_depth > to_compare.get_max_depth()) {
//		return false;
//	}

	 if (m_pointed.get_numerical_id() < to_compare.get_pointed().get_numerical_id()) {
	 	return true;
	 } else if (m_pointed.get_numerical_id() > to_compare.get_pointed().get_numerical_id()) {
	 	return false;
	 }

//	bisimulation b;
//
//	if (m_edges < to_compare.get_edges()) {
//		return b.compare_automata(*this, to_compare);
//	} else {
//		return b.compare_automata(to_compare, *this);
//	}

	 if (m_worlds < to_compare.get_worlds()) {
	 	return true;
	 } else if (m_worlds > to_compare.get_worlds()) {
	 	return false;
	 }

	 if (m_edges < to_compare.get_edges()) {
	 	return true;
	 } else if (m_edges > to_compare.get_edges()) {
	 	return false;
	 }

	//These are implemented in std

	//	if (m_worlds.size() < to_compare.get_worlds().size()) {
	//		return true;
	//	} else if (m_worlds.size() > to_compare.get_worlds().size()) {
	//		return false;
	//	}
	//
	//	if (m_edges.size() < to_compare.get_edges().size()) {
	//		return true;
	//	} else if (m_edges.size() > to_compare.get_edges().size()) {
	//		return false;
	//	}

	//	kworld_ptr_set::const_iterator it_kwpts1, it_kwpts2;
	//	it_kwpts2 = to_compare.get_worlds().begin();
	//	for (it_kwpts1 = m_worlds.begin(); it_kwpts1 != m_worlds.end(); it_kwpts1++) {
	//		if ((*it_kwpts1) < (*it_kwpts2)) {
	//			return true;
	//		} else if ((*it_kwpts1) > (*it_kwpts2)) {
	//			return false;
	//		}
	//		//The case when to compare has less world than this
	//		it_kwpts2++;
	//	}

	//	kedge_ptr_set::const_iterator it_kepts1, it_kepts2;
	//	it_kepts2 = to_compare.get_edges().begin();
	//	for (it_kepts1 = m_edges.begin(); it_kepts1 != m_edges.end(); it_kepts1++) {
	//		if ((*it_kepts1) < (*it_kepts2)) {
	//			return true;
	//		} else if ((*it_kepts1) > (*it_kepts2)) {
	//			return false;
	//		}
	//		it_kepts2++;
	//	}

	return false;
}

bool kstate::entails(const belief_formula & to_check, const kworld_ptr_set & reachable) const
{
	kworld_ptr_set::const_iterator it_kwl;
	for (it_kwl = reachable.begin(); it_kwl != reachable.end(); it_kwl++) {
		/**\todo why setted contary?*/
		if (!entails(to_check, (*it_kwl)))
			return false;
	}
	return true;
}

bool kstate::entails(const belief_formula & bf, kworld_ptr world) const
{
	/*
	 The entailment of a \ref belief_formula just call recursively the entailment on all the reachable world with that formula.
	 */
	kworld_ptr_set D_reachable;
	switch ( bf.get_formula_type() ) {

	case FLUENT_FORMULA:
	{
		/** \todo Make sure its grounded. Maybe add to \ref belief_formula a bool that store if grounded or not or maybe ground
		 * when \ref domain created.
		 * @see belief_formula::ground(const grounder &).*/
		return entails(bf.get_fluent_formula(), world);
		break;
	}

	case BELIEF_FORMULA:

		/** \todo what was the at_lest_one of the previous version?*/
		return entails(bf.get_bf1(), get_B_reachable_worlds(bf.get_agent(), world));
		break;

	case PROPOSITIONAL_FORMULA:
		switch ( bf.get_operator() ) {
		case BF_NOT:
			return !entails(bf.get_bf1(), world);
			break;
		case BF_OR:
			return entails(bf.get_bf1(), world) || entails(bf.get_bf2(), world);
			break;
		case BF_AND:
			return entails(bf.get_bf1(), world) && entails(bf.get_bf2(), world);
			break;
		case BF_FAIL:
		default:
			std::cerr << "Something went wrong in checking entailment for Propositional formula";
			exit(1);
		}
		break;

	case E_FORMULA:
		//Check the entails on the E-reachable worlds
		return entails(bf.get_bf1(), get_E_reachable_worlds(bf.get_group_agents(), world));
		break;
		//Check the entails on the D-reachable worlds
	case D_FORMULA:
		D_reachable = get_D_reachable_worlds(bf.get_group_agents(), world);
		if (D_reachable.size() > 0) {
			return entails(bf.get_bf1(), D_reachable);
		} else {
			return false;
		}
		break;
		//Check the entails on the C-reachable worlds
	case C_FORMULA:
		return entails(bf.get_bf1(), get_C_reachable_worlds(bf.get_group_agents(), world));
		break;
	case BF_EMPTY:
	{

		return true;
		break;
	}
	case BF_TYPE_FAIL:
	default:
		std::cerr << "Something went wrong in checking entailment for Belief formula";
		exit(1);
	}
	return false;

}

bool kstate::entails(const formula_list & to_check, kworld_ptr world) const
{
	//formula_list expresses CNF formula
	formula_list::const_iterator it_fl;
	for (it_fl = to_check.begin(); it_fl != to_check.end(); it_fl++) {
		if (!entails((*it_fl), world)) {
			return false;
		}
	}
	return true;
}

const kworld_ptr_set kstate::get_B_reachable_worlds(agent ag, kworld_ptr world) const
{
	kworld_ptr_set ret;
	kedge_ptr_set::const_iterator it_kedge;
	for (it_kedge = m_edges.begin(); it_kedge != m_edges.end(); it_kedge++) {
		if (((*it_kedge).get_from() == world) && ((*it_kedge).get_label() == ag)) {
			ret.insert((*it_kedge).get_to());
		}
	}
	return ret;
}

bool kstate::get_B_reachable_worlds_recoursive(agent ag, kworld_ptr world, kworld_ptr_set & ret) const
{
	/** \todo check: If a--i-->b, b--i-->c then a--i-->c must be there*/
	bool is_fixed_point = true;
	kedge_ptr_set::const_iterator it_kedge;
	for (it_kedge = m_edges.begin(); it_kedge != m_edges.end(); it_kedge++) {
		if (((*it_kedge).get_from() == world) && ((*it_kedge).get_label() == ag)) {
			//We use the pair of insert, if we add a new world (true in the set::insert) then is not a fixed point
			//if (std::get<1>(ret.insert((*it_kedge).get_to()))) {
			if ((ret.insert((*it_kedge).get_to())).second) {
				is_fixed_point = false;
			}
		}
	}
	return is_fixed_point;
}

const kworld_ptr_set kstate::get_E_reachable_worlds(const agent_set & ags, kworld_ptr world) const
{
	/*Optimized, the K^0 call of this function
	 *
	 * Not calling B_reachability iteratively for optimization
	 */
	kworld_ptr_set ret;
	kedge_ptr_set::const_iterator it_kedge;
	for (it_kedge = m_edges.begin(); it_kedge != m_edges.end(); it_kedge++) {
		if (((*it_kedge).get_from() == world) && (ags.find((*it_kedge).get_label()) != ags.end())) {
			ret.insert((*it_kedge).get_to());
		}
	}
	return ret;
}

bool kstate::get_E_reachable_worlds_recoursive(const agent_set & ags, const kworld_ptr_set &worlds, kworld_ptr_set & ret) const
{
	/*Optimized, the K^i (recoursive) call of this function*/

	bool is_fixed_point = true;
	kedge_ptr_set::const_iterator it_kedge;
	/*\bug What if the pointed is not reachable, correct this*/
	for (it_kedge = m_edges.begin(); it_kedge != m_edges.end(); it_kedge++) {
		if ((worlds.find((*it_kedge).get_from()) != worlds.end()) && (ags.find((*it_kedge).get_label()) != ags.end())) {
			if (ret.insert((*it_kedge).get_to()).second) {
				is_fixed_point = false;
			}
		}
	}
	return is_fixed_point;
}

const kworld_ptr_set kstate::get_C_reachable_worlds(const agent_set & ags, kworld_ptr world) const
{
	//Use of fixed point to stop.
	bool is_fixed_point = false;
	//THIS IS K^0
	kworld_ptr_set newly_reached = get_E_reachable_worlds(ags, world);
	kworld_ptr_set already_reached;
	kworld_ptr_set ret;
	//FROM HERE K^i UNTIL FIXED_POINT
	while (!is_fixed_point) {
		sum_set(newly_reached, ret);
		minus_set(newly_reached, already_reached);
		is_fixed_point = get_E_reachable_worlds_recoursive(ags, newly_reached, ret);
		already_reached = newly_reached;
	}
	return ret;
}

const kworld_ptr_set kstate::get_D_reachable_worlds(const agent_set & ags, kworld_ptr world) const
{
	/**@bug: Notion of D-Reachable is correct (page 24 of Reasoning about Knowledge)*/
	agent_set::const_iterator it_agset = ags.begin();
	kworld_ptr_set ret = get_B_reachable_worlds((*it_agset), world);
	it_agset++;

	for (; it_agset != ags.end(); it_agset++) {

		kworld_ptr_set::iterator it_pwset1 = ret.begin();
		kworld_ptr_set to_intersect = get_B_reachable_worlds((*it_agset), world);
		kworld_ptr_set::const_iterator it_pwset2 = to_intersect.begin();
		while ((it_pwset1 != ret.end()) && (it_pwset2 != to_intersect.end())) {

			if ((*it_pwset1 < *it_pwset2) && ((*it_pwset1).get_fluent_based_id().compare((*it_pwset2).get_fluent_based_id()) != 0)) {
				ret.erase(it_pwset1++);
			} else if ((*it_pwset2 < *it_pwset1) && ((*it_pwset1).get_fluent_based_id().compare((*it_pwset2).get_fluent_based_id()) != 0)) {
				++it_pwset2;
			} else { // *it_pwset1 == *it_pwset2
				++it_pwset1;
				++it_pwset2;
			}
		}

		// Anything left in ret from here on did not appear in to_intersect,
		// so we remove it.
		ret.erase(it_pwset1, ret.end());

	}
	std::cerr << "\nERROR: D_REACHABLILITY not yet Implemented correctly\n";
	exit(1);
	return ret;
}

void kstate::get_all_reachable_worlds(const kworld_ptr & kw, kworld_ptr_set & reached_worlds, const std::map<kworld_ptr, kworld_ptr_set> & adj_list) const
{
	kworld_ptr_set::const_iterator it_kwps;

	//reached_worlds.insert(kw);
	kworld_ptr_set kw_list = adj_list.at(kw);

	for (it_kwps = kw_list.begin(); it_kwps != kw_list.end(); it_kwps++) {
		if (reached_worlds.insert(*it_kwps).second) {
			get_all_reachable_worlds(*it_kwps, reached_worlds, adj_list);
		}
	}
}

void kstate::clean_unreachable_kworlds(std::map<kworld_ptr, kworld_ptr_set> & adj_list)
{

	kworld_ptr_set reached_worlds;
	kedge_ptr_set reached_edges;

	kedge_ptr_set::const_iterator it_keps;

	reached_worlds.insert(get_pointed());
	get_all_reachable_worlds(get_pointed(), reached_worlds, adj_list);

	for (it_keps = m_edges.begin(); it_keps != m_edges.end(); it_keps++) {
		if (reached_worlds.find(it_keps->get_from()) != reached_worlds.end()) {
			reached_edges.insert(*it_keps);
		}/* else {
			adj_list[it_keps->get_from()].erase(it_keps->get_to());

			if (adj_list.at(it_keps->get_from()).size() == 0) {
				adj_list.erase(it_keps->get_from());
				//reached_worlds.erase(it_keps->get_from());
			}
		}*/
	}

	set_worlds(reached_worlds);
	set_edges(reached_edges);

}

const automa kstate::kstate_to_automaton(/*const std::map<kworld_ptr, kworld_ptr_set> & adj_list,*/ std::vector<kworld_ptr> & kworld_vec, const std::map<agent, bis_label> & agent_to_label) const
{

	std::map<int, int> compact_indices;
	std::map<kworld_ptr, int> index_map;
	kbislabel_map label_map; // Map: from -> (to -> ag_set)

	automa *a;
	int Nvertex = get_worlds().size();
	int ag_set_size = domain::get_instance().get_agents().size();
	//BIS_ADAPTATION For the loop that identifies the id (We add one edge for each node)
	v_elem *Vertex;

	Vertex = (v_elem *) malloc(sizeof(v_elem) * Nvertex);

	// Initializating vertices
	kworld_ptr_set::const_iterator it_kwps;
	kedge_ptr_set::const_iterator it_keps;
	kbislabel_map::const_iterator it_klm;
	bis_label_set::const_iterator it_bislab;
	std::map<kworld_ptr, bis_label_set>::const_iterator it_kw_bislab;

	//std::cerr << "\nDEBUG: Inizializzazione Edges\n";

	// The pointed world is set to the index 0. This ensures that, when deleting the bisimilar nodes, the pointed kworld
	// is always chosen as the first of its block. Therefore, we do not need to update it when converting back to a kstate
	index_map[get_pointed()] = 0;
	kworld_vec.push_back(get_pointed());
	compact_indices[get_pointed().get_numerical_id()] = 0;

	//For the loop that identifies the id
	//BIS_ADAPTATION For the loop that identifies the id (+1)
	///@bug: If the pointed has no self-loop to add
	//kworld_ptr_set pointed_adj = adj_list.at(get_pointed());

	Vertex[0].ne = 0; // pointed_adj.size(); // edge_counter[get_pointed()];
	//	if (pointed_adj.find(get_pointed()) == pointed_adj.end()) {
	//		Vertex[0].ne++;
	//	}
	//	Vertex[0].e = (e_elem *) malloc(sizeof(e_elem) * Vertex[0].ne);

	int i = 1, c = 1;

	//std::cerr << "\nDEBUG: Inizializzazione Vertex\n";

	for (it_kwps = m_worlds.begin(); it_kwps != m_worlds.end(); it_kwps++) {
		if (!(*it_kwps == get_pointed())) {
			index_map[*it_kwps] = i;
			kworld_vec.push_back(*it_kwps);

			// if (compact_indices.find(it_kwps->get_numerical_id()) == compact_indices.end()) {
			if (compact_indices.insert({it_kwps->get_numerical_id(), c}).second) {
				// compact_indices[it_kwps->get_numerical_id()] = c;
				c++;
				//std::cerr << "\nDEBUG: Added:" << it_kwps->get_id() << "\n";
			}

			//BIS_ADAPTATION For the loop that identifies the id (+1)
			Vertex[i].ne = 0; //adj_list.at(*it_kwps).size(); // edge_counter[*it_kwps];
			//			Vertex[i].e = (e_elem *) malloc(sizeof(e_elem) * Vertex[i].ne);
			i++;
		}
		//BIS_ADAPTATION (Added self-loop)
		label_map[*it_kwps][*it_kwps].insert(compact_indices[it_kwps->get_numerical_id()] + ag_set_size);
		//std::cerr << "\nDEBUG: Added to " << it_kwps->get_numerical_id() << " the label " << compact_indices[it_kwps->get_numerical_id()] + ag_set_size << std::endl;
	}


	//BIS_ADAPTATION For the loop that identifies the id (We add one potential label for each node)
	int bhtabSize = ag_set_size + c;

	//std::cerr << "\nDEBUG: Inizializzazione Behavs\n";

	//BIS_ADAPTATION (Moved down here)
	for (it_keps = m_edges.begin(); it_keps != m_edges.end(); it_keps++) {

		//DEBUG:Change this
		// if (adj_list[it_keps->get_from()][it_keps->get_to()].empty())
		// {
		// 	edge_counter[it_keps->get_from()]++;
		// }

		label_map[it_keps->get_from()][it_keps->get_to()].insert(agent_to_label.at(it_keps->get_label()));
		Vertex[index_map[it_keps->get_from()]].ne++;
	}

	i = 0;
	for (it_kwps = m_worlds.begin(); it_kwps != m_worlds.end(); it_kwps++) {
		Vertex[i].ne++; //Self loop bisimulation
		Vertex[i].e = (e_elem *) malloc(sizeof(e_elem) * Vertex[i].ne);
		i++;
	}

	//std::cerr << "\nDEBUG: Fine Inizializzazione Vertex\n";


	int from, to, j = 0; //, k = 0, nbh;

	//std::cerr << "\nDEBUG: Inizializzazione Mappa\n";
	for (it_klm = label_map.begin(); it_klm != label_map.end(); it_klm++) {
		from = index_map[it_klm->first]; // For each kworld 'from'

		//std::cerr << "\nDEBUG: Inizializzazione K\n";

		for (it_kw_bislab = it_klm->second.begin(); it_kw_bislab != it_klm->second.end(); it_kw_bislab++) { // For each edge that reaches the kworld 'to'
			to = index_map[it_kw_bislab->first];
			//nbh = it_kw_bislab->second.size();

			for (it_bislab = it_kw_bislab->second.begin(); it_bislab != it_kw_bislab->second.end(); it_bislab++) { // For each agent 'ag' in the label of the kedge
				//std::cerr << "\nDEBUG: j is: " << j << " and k is: " << k << "\n";
				//nbh = 1;
				Vertex[from].e[j].nbh = 1; // Let j be the index of the adjacency list of from that stores the kedge (from, to)
				Vertex[from].e[j].bh = (int *) malloc(sizeof(int)); // Let nbh be the number of agents in such kedge
				Vertex[from].e[j].tv = to; // Update the value of the reache kworld
				Vertex[from].e[j].bh[0] = *it_bislab; // Update the value of the label at index k to 'ag'
				//std::cerr << "\nDEBUG: j is: " << j << " and k is: " << k << "\n";

				j++; // Update the value of the index j
			}

			//std::cerr << "\nDEBUG J: " << j-1 << "\n";

			//k = 0; // Reset k
		}
		//std::cerr << "\nDEBUG: Fine Inizializzazione K\n";

		j = 0; // Reset j
	}

	//std::cerr << "\nDEBUG: Fine Inizializzazione Mappa\n";

	//		std::vector<kworld_ptr>::const_iterator it_kwp;
	//	int temp_counter = 0;
	//	for (it_kwp = kworld_vec.begin(); it_kwp != kworld_vec.end(); it_kwp++) {
	//		//std::cerr << "DEBUG: World " << temp_counter << " has ID: " << it_kwp->get_numerical_id() << std::endl;
	//		temp_counter++;
	//	}
	//
	// Building the automaton
	int Nbehavs = bhtabSize;
	a = (automa *) malloc(sizeof(automa));
	a->Nvertex = Nvertex;
	a->Nbehavs = Nbehavs;
	a->Vertex = Vertex;

	//	std::cerr << "\nDEBUG: \n\tNvertex = " << Nvertex << std::endl;
	//	std::cerr << "\tNbehavs = " << Nbehavs << std::endl;

	return *a;
}

void kstate::automaton_to_kstate(const automa & a, const std::vector<kworld_ptr> & kworld_vec, const std::map<bis_label, agent> & label_to_agent)
{
	kworld_ptr_set worlds;
	kedge_ptr_set edges;
	// The pointed world does not change when we calculate the minimum bisimilar state
	// Hence we do not need to update it

	int i, j, k, label, agents_size = domain::get_instance().get_agents().size();

	for (i = 0; i < a.Nvertex; i++) {
		if (a.Vertex[i].ne > 0) {
			worlds.insert(kworld_vec[i]);

			for (j = 0; j < a.Vertex[i].ne; j++) {
				for (k = 0; k < a.Vertex[i].e[j].nbh; k++) {
					label = a.Vertex[i].e[j].bh[k];
					if (label < agents_size) {
						edges.insert(kstore::get_instance().add_edge(kedge(kworld_vec[i], kworld_vec[a.Vertex[i].e[j].tv], label_to_agent.at(label))));
					}
				}
			}
		}

	}


	set_worlds(worlds);
	set_edges(edges);
}

void kstate::calc_min_bisimilar()
{

	// ************* Cleaning unreachable kworlds *************
	//DEBUG_add_extra_world();

	
	std::map<bis_label, agent> label_to_agent;
	std::map<agent, bis_label> agent_to_label;


	auto agents = domain::get_instance().get_agents();
	auto it_ag = agents.begin();
	bis_label ag_label = 0;
	agent lab_agent;
	for (; it_ag != agents.end(); it_ag++) {
		lab_agent = *it_ag;
		label_to_agent.insert(std::make_pair(ag_label, lab_agent));
		agent_to_label.insert(std::make_pair(lab_agent, ag_label));
		ag_label++;

	}


	std::map<kworld_ptr, kworld_ptr_set> adj_list;
	kedge_ptr_set::const_iterator it_keps;

	for (it_keps = m_edges.begin(); it_keps != m_edges.end(); it_keps++) {
		adj_list[it_keps->get_from()].insert(it_keps->get_to());
	}

	clean_unreachable_kworlds(adj_list);

	//std::cerr << "\nDEBUG: INIZIO BISIMULATION IN KSTATE\n" << std::flush;
	std::vector<kworld_ptr> kworld_vec; // Vector of all kworld_ptr
	//std::cerr << "\nDEBUG: PRE-ALLOCAZIONE AUTOMA\n" << std::flush;


	//	std::cerr << "\nDEBUG: \n\tNvertex_before = " << m_worlds.size() << std::endl;
	//	std::cerr << "\tNbehavs_before = " << m_edges.size() << std::endl;

	automa a;
	kworld_vec.reserve(get_worlds().size());

	a = kstate_to_automaton(/*adj_list,*/ kworld_vec, agent_to_label);

	bisimulation b;
	//std::cout << "\nDEBUG: Printing automa pre-Bisimulation\n";
	//b.VisAutoma(&a);



	if (domain::get_instance().get_bisimulation() == PaigeTarjan) {
		if (b.MinimizeAutomaPT(&a)) {
			//VisAutoma(a);

			//std::cout << "\nDEBUG: Printing automa post-Bisimulation\n";
			//b.VisAutoma(&a);
			//std::cout << "Done\n";
			automaton_to_kstate(a, kworld_vec,label_to_agent);

			//b.DisposeAutoma(&a);
		}
	} else {
		if (b.MinimizeAutomaFB(&a)) {

			//std::cerr << "\nDEBUG: Printing automa post-Bisimulation\n";
			//b.VisAutoma(&a);
			//std::cerr << "Done\n";
			automaton_to_kstate(a, kworld_vec,label_to_agent);
			//b.DisposeAutoma(&a);
		}
	}

	//std::cerr << "\nDEBUG: PRe Clean" << std::endl;

}

void kstate::add_world(const kworld & world)
{
	m_worlds.insert(kstore::get_instance().add_world(world));
}

kworld_ptr kstate::add_rep_world(const kworld & world, unsigned short repetition, bool& is_new)
{
	kworld_ptr tmp = kstore::get_instance().add_world(world);
	tmp.set_repetition(repetition);
	is_new = std::get<1>(m_worlds.insert(tmp));
	return tmp;
}

kworld_ptr kstate::add_rep_world(const kworld & world, unsigned short old_repetition)
{
	bool tmp = false;
	return add_rep_world(world, get_max_depth() + old_repetition, tmp);
}

kworld_ptr kstate::add_rep_world(const kworld & world)
{
	bool tmp = false;
	return add_rep_world(world, get_max_depth(), tmp);
}

void kstate::add_edge(const kedge & edge)
{
	m_edges.insert(kstore::get_instance().add_edge(edge));
}

void kstate::build_initial()
{
	/** \todo for now prune building.*/
	std::cout << "\nBuilding initial Kripke structure...\n";
	build_initial_prune();
}

void kstate::build_initial_structural()
{

}

void kstate::build_initial_prune()
{

	/*Building of all the possible consistent \ref kworld and setting the pointed world.
	 * Creation of all the \ref fluent combinations. All the consistent ones are added to \ref kstore.*/
	fluent_set permutation;
		initially ini_conditions = domain::get_instance().get_initial_description();

	generate_initial_kworlds(permutation, 0, ini_conditions.get_initially_known_fluents());


	/*Building of all the consistent \ref kedge.*/
	generate_initial_kedges();
}

/*From https://www.geeksforgeeks.org/generate-all-the-binary-strings-of-n-bits/ since is like generating all the binary numbers.*/
void kstate::generate_initial_kworlds(fluent_set& permutation, int index, const fluent_set & initially_known)
{
		int fluent_number = domain::get_instance().get_fluent_number();

	int bit_size = (domain::get_instance().get_size_fluent());

	if (index == fluent_number) {
		kworld to_add(permutation);
		add_initial_kworld(to_add);

		return;
	}

	fluent_set permutation_2 = permutation;
	//Add the \ref fluent in positive version
	boost::dynamic_bitset<> bitSetToFindPositve(bit_size, index);
	boost::dynamic_bitset<> bitSetToFindNegative(bit_size, index);
	bitSetToFindNegative.set(bitSetToFindPositve.size() - 1, 1);
	bitSetToFindPositve.set(bitSetToFindPositve.size() - 1, 0);


	if (initially_known.find(bitSetToFindNegative) == initially_known.end()) {
		permutation.insert(bitSetToFindPositve);
		generate_initial_kworlds(permutation, index + 1, initially_known);
	}
	if (initially_known.find(bitSetToFindPositve) == initially_known.end()) {
		permutation_2.insert(bitSetToFindNegative);
		generate_initial_kworlds(permutation_2, index + 1, initially_known);
	}

}

void kstate::add_initial_kworld(const kworld & possible_add)
{
	initially ini_conditions = domain::get_instance().get_initial_description();

	switch ( ini_conditions.get_ini_restriction() ) {
	case S5:
	{
		/* Since the common knowledge is on all the agent it means that every possible \ref kworld
		 * in the initial state must entail *phi* where C(*phi*) is an initial condition.*/

		//Already setted in \ref domain::build_initially(bool).
		if (possible_add.entails(ini_conditions.get_ff_forS5())) {
			add_world(possible_add);
			if (possible_add.entails(ini_conditions.get_pointed_world_conditions())) {
				m_pointed = kworld_ptr(possible_add);
				/*std::cout << "pointed world: ";
				printer::get_instance().print_list(possible_add.get_fluent_set());
				std::cout << std::endl;*/
			}
		} else {
			//Already generated so we save it on kstore
			kstore::get_instance().add_world(possible_add);
		}
		break;
	}
	case K45:
	{
		std::cerr << "\nNot implemented yet - 1\n";
		exit(1);
		break;
	}
	case NONE:
	{
		std::cerr << "\nNot implemented yet - 2\n";
		exit(1);
		break;
	}
	default:
	{
		std::cerr << "\nNot implemented yet - 3\n";
		exit(1);

		break;
	}
	}
}

void kstate::generate_initial_kedges()
{
	kworld_ptr_set::const_iterator it_kwps_1, it_kwps_2;
	auto agents = domain::get_instance().get_agents();
	kworld_ptr kwptr_tmp1, kwptr_tmp2;

	/*This for add to *this* all the possible edges.*/
	for (it_kwps_1 = m_worlds.begin(); it_kwps_1 != m_worlds.end(); it_kwps_1++) {
		for (it_kwps_2 = it_kwps_1; it_kwps_2 != m_worlds.end(); it_kwps_2++) {
			for (auto i = agents.begin(); i != agents.end(); i++) {
				kwptr_tmp1 = *it_kwps_1;
				kwptr_tmp2 = *it_kwps_2;

				add_edge(kedge(kwptr_tmp1, kwptr_tmp2, *i));
				add_edge(kedge(kwptr_tmp2, kwptr_tmp1, *i));

			}
		}
	}

	//std::cout << "Tot edges: " << m_edges.size() << std::endl;

	initially ini_conditions = domain::get_instance().get_initial_description();

	formula_list::const_iterator it_fl;
	for (it_fl = ini_conditions.get_initial_conditions().begin(); it_fl != ini_conditions.get_initial_conditions().end(); it_fl++) {
		remove_initial_kedge_bf(*it_fl);
	}
	//std::cout << "Removed edges: " << count << std::endl;

	//std::cout << "Final edges: " << m_edges.size() << std::endl;

}

void kstate::remove_kedge(const kedge & to_remove)
{

	m_edges.erase(kstore::get_instance().add_edge(to_remove));
	//if(m_edges.erase(kstore::get_instance().add_edge(to_remove)) > 0)count++;
}

void kstate::remove_initial_kedge(const fluent_formula & known_ff, agent ag)
{

	kworld_ptr_set::const_iterator it_kwps_1, it_kwps_2;

	kworld_ptr kwptr_tmp1, kwptr_tmp2;

	/** \todo maybe don't loop twice on the world but exploit using it_kwps_2 = it_kwps_1:
	 * - remove (_1, _2).
	 * - remove (_2, _1).*/
	for (it_kwps_1 = m_worlds.begin(); it_kwps_1 != m_worlds.end(); it_kwps_1++) {
		for (it_kwps_2 = it_kwps_1; it_kwps_2 != m_worlds.end(); it_kwps_2++) {
			/** \todo or entails(-known_ff)?*/
			kwptr_tmp1 = *it_kwps_1;
			kwptr_tmp2 = *it_kwps_2;
			if (kwptr_tmp1.get_ptr()->entails(known_ff) && !kwptr_tmp2.get_ptr()->entails(known_ff)) {
				remove_kedge(kedge(kwptr_tmp1, kwptr_tmp2, ag));
				remove_kedge(kedge(kwptr_tmp2, kwptr_tmp1, ag));
			} else if (kwptr_tmp2.get_ptr()->entails(known_ff) && !kwptr_tmp1.get_ptr()->entails(known_ff)) {

				remove_kedge(kedge(kwptr_tmp2, kwptr_tmp1, ag));
				remove_kedge(kedge(kwptr_tmp1, kwptr_tmp2, ag));
			}
		}
	}

}

void kstate::remove_initial_kedge_bf(const belief_formula & to_check)
{
	switch ( domain::get_instance().get_initial_description().get_ini_restriction() ) {
	case S5:
	{
		/* Just check whenever is B(--) \/ B(--) and remove that edge*/
		if (to_check.get_formula_type() == C_FORMULA) {

			belief_formula tmp = to_check.get_bf1();

			switch ( tmp.get_formula_type() ) {

				//Only one for edges -- expresses that someone is ignorant.
			case PROPOSITIONAL_FORMULA:
			{
				//We remove all the check on the formula since they have already been controlled when ini_conditions has been created
				if (tmp.get_operator() == BF_OR) {

					//fluent_formula known_ff;
					auto known_ff_ptr = std::make_shared<fluent_formula>();
					helper::check_Bff_notBff(tmp.get_bf1(), tmp.get_bf2(), known_ff_ptr);
					if (known_ff_ptr != nullptr) {
						//printer::get_instance().print_list(*known_ff_ptr);
						remove_initial_kedge(*known_ff_ptr, tmp.get_bf2().get_agent());
					}
					return;

				} else if (tmp.get_operator() == BF_AND) {
					//This case doesn't add knowledge.
					return;

				} else {
					std::cerr << "\nError in the type of initial formulae (FIFTH).\n";
					exit(1);
				}
				return;
				break;
			}
			case FLUENT_FORMULA:
			case BELIEF_FORMULA:
			{
				return;
				break;
			}
			case BF_EMPTY:
			{
				return;
				break;
			}

			default:
			{
				std::cerr << "\nError in the type of initial formulae (SIXTH).\n";
				exit(1);
				break;
			}
			}
		} else {
			std::cerr << "\nError in the type of initial formulae (SEVENTH).\n";
			exit(1);
		}

		return;
		break;
	}

	case K45:
	{
		std::cerr << "\nNot implemented yet - 4\n";
		exit(1);
		break;
	}
	case NONE:
	{
		std::cerr << "\nNot implemented yet - 5\n";
		exit(1);
		break;
	}
	default:
	{
		std::cerr << "\nNot implemented yet - 6\n";
		exit(1);

		break;
	}
	}
}

/** \warning executability should be check in \ref state (or \ref planner).*/
kstate kstate::compute_succ(const action & act) const
{
	switch ( act.get_type() ) {
	case ONTIC:
	{
		/*kstate tmp = execute_ontic(act);
		std::cout << "\n*********************Printing this*********************\n";
		print();
		std::cout << "\n*********************Printing tmp*********************\n";
		tmp.print();*/
		return domain::get_instance().get_k_optimized() ? execute_ontic(act) : execute_ontic_um(act);
		break;
	}
	case SENSING:
	{
		return domain::get_instance().get_k_optimized() ? execute_sensing(act) : execute_sensing_um(act);
		break;
	}
	case ANNOUNCEMENT:
	{
		return domain::get_instance().get_k_optimized() ? execute_announcement(act) : execute_announcement_um(act);
		break;
	}
	default:
	{
		std::cerr << "Error in executing an action: ";
		std::cerr << "the type of the action is not defined correctly";
		std::cerr << std::endl;
		exit(1);

		break;
	}
	}
}

/**We keep the agent_set to not calculate every time the fully obsv if the obsv is global.*/
void kstate::add_ret_ontic_worlds_internal(const kworld_ptr & start, kworld_ptr_set &reached, kstate& ret, const fluent_formula & effects, agent_set & fully_obs_agents, const action & act, action_check act_check, std::map<kworld_ptr, kworld_ptr>& map_for_edges) const
{
	/**\todo Maybe pointed useless as parameter?*/
	agent_set::const_iterator it_agset;
	kworld_ptr_set reachable_by_ag;
	agent tmp_ag;
	kworld_ptr_set::const_iterator it_kwset;
	kworld_ptr tmp_kworld_ptr;
	fluent_formula::const_iterator it_eff;

	fluent_set world_description = start.get_fluent_set();
	//Execute the all the effects
	//std::cout << "\n\n*************************" << act.get_name() << "*************************\n";
	//Formula in helper that deals with the fluent_formula
	for (it_eff = effects.begin(); it_eff != effects.end(); it_eff++) {
		helper::apply_effect(*it_eff, world_description);
	}

	//Insert into a map the pair <old_world, new_world>.
	/**\todo to optimize*/
	map_for_edges.insert(std::pair<kworld_ptr, kworld_ptr>(start, ret.add_rep_world(kworld(world_description), start.get_repetition())));
	for (it_agset = fully_obs_agents.begin(); it_agset != fully_obs_agents.end(); it_agset++) {

		tmp_ag = *it_agset;

		reachable_by_ag = get_B_reachable_worlds(tmp_ag, start);
		minus_set(reachable_by_ag, reached);

		if (reachable_by_ag.size() != 0) {
			sum_set(reached, reachable_by_ag);

			for (it_kwset = reachable_by_ag.begin(); it_kwset != reachable_by_ag.end(); it_kwset++) {
				tmp_kworld_ptr = *it_kwset;
				//Find the new fully observant if visibility is relative
				if (!domain::get_instance().get_is_global_obsv()) {
					fully_obs_agents = get_agents_if_entailed(act.get_fully_observants(), tmp_kworld_ptr);
					std::cerr << "\nRELATIVE_OBSERVABILITY - Not fully implemented yet.\n";
					exit(1);
				}
				if (act_check == EXE_POINTED__COND_POINTED) {
					add_ret_ontic_worlds_internal(tmp_kworld_ptr, reached, ret, effects, fully_obs_agents, act, act_check, map_for_edges);
				} else {
					add_ret_ontic_worlds(tmp_kworld_ptr, reached, ret, fully_obs_agents, act, act_check, map_for_edges);
				}
			}

			ret.set_max_depth(ret.get_max_depth() + 1 + start.get_repetition());
		}
	}
}

void kstate::add_ret_ontic_worlds(const kworld_ptr & start, kworld_ptr_set &reached, kstate& ret, agent_set & fully_obs_agents, const action & act, action_check act_check, std::map<kworld_ptr, kworld_ptr>& map_for_edges) const
{
	fluent_formula effects;
	switch ( act_check ) {
	case (EXE_POINTED__COND_POINTED):
	{
		effects = get_effects_if_entailed(act.get_effects(), get_pointed());
		break;
	}
	case (EXE_POINTED__COND_WORLD):
	{
		effects = get_effects_if_entailed(act.get_effects(), start);
		break;
	}
	case (EXE_WORLD__COND_WORLD):
	{
		//The action executability is already be checked in state_T
		if (entails(act.get_executability(), start)) {
			effects = get_effects_if_entailed(act.get_effects(), start);
		} else {
			return;
		}
		break;
	}
	default:
	{
		std::cerr << "\nType of action checking not correctly decalred.\n";
		exit(1);
	}
	}
	add_ret_ontic_worlds_internal(start, reached, ret, effects, fully_obs_agents, act, act_check, map_for_edges);
}

void kstate::add_ste_worlds(kstate &ret, const kworld_ptr &kw, kstate_map &kmap, const event_type e, const action & act) const
{
	///Inside this function is also updated the fluent_set of the worlds linked to ontic and fully observant
	///(Updating the interpretations of the worlds)
	bool tmp = false;
	int offset = kw.get_repetition() + e;
	fluent_set world_description = kw.get_fluent_set();

	if (e == SIGMA) {
		if (act.get_type() == ONTIC) {
			fluent_formula::const_iterator it_eff;
			fluent_formula effects = get_effects_if_entailed(act.get_effects(), kw);
			if (effects.size() > 0) {
				for (it_eff = effects.begin(); it_eff != effects.end(); it_eff++) {
					helper::apply_effect(*it_eff, world_description);
				}

				if (world_description != kw.get_fluent_set()) {
					ret.set_max_depth(ret.get_max_depth() + 2);
					offset += ret.get_max_depth();
				}
			}
		}
	}

	kmap.insert(kstate_map::value_type({kw, e},
	ret.add_rep_world(kworld(world_description), offset, tmp)));

	if (e == SIGMA && kw == get_pointed()) {
		ret.set_pointed(kmap[{kw, SIGMA}
		]);
	}
}

kstate kstate::execute_action_um(const action& act, const event_type_set& events, const event_type_relation& fully_obs_r, const event_type_relation& partially_obs_r, const event_type_relation & oblivious_obs_r) const
{
	//The execution are all the same if we consider that false beliefs don't count.
	kstate ret;

	//This finds all the worlds that are reachable from the initial state following
	//the edges labeled with fully observant agents.
	agent_set agents = domain::get_instance().get_agents();
	agent_set fully_obs_agents = get_agents_if_entailed(act.get_fully_observants(), get_pointed());
	agent_set partially_obs_agents = get_agents_if_entailed(act.get_partially_observants(), get_pointed());

	agent_set oblivious_obs_agents = agents;
	minus_set(oblivious_obs_agents, fully_obs_agents);
	minus_set(oblivious_obs_agents, partially_obs_agents);

	kstate_map kmap;

	if (!oblivious_obs_agents.empty()) {
		ret.set_max_depth(get_max_depth() + 1);
	} else {
		ret.set_max_depth(get_max_depth());
	}
	kworld_ptr_set::const_iterator it_kwset1;
	// Creating the new worlds for the kstate ret
	for (it_kwset1 = get_worlds().begin(); it_kwset1 != get_worlds().end(); it_kwset1++) {
		for (event_type e : events) {
			switch ( act.get_type() ) {
			case ONTIC:
			{
				if ((e == EPSILON && oblivious_obs_agents.size() > 0) || (e == SIGMA)) {
					add_ste_worlds(ret, *it_kwset1, kmap, e, act);
				}
				break;
			}
			case SENSING:
			case ANNOUNCEMENT:
			{
				fluent_formula effects = get_effects_if_entailed(act.get_effects(), *it_kwset1);
				/** todo: controllare che l'effetto condizionale vada rispettato */
				if (e == EPSILON && (oblivious_obs_agents.size() > 0)) {
					add_ste_worlds(ret, *it_kwset1, kmap, e, act);
				} else { // if (entails(act.get_executability(), *it_kwset1)) {
					bool ent = (entails(effects, *it_kwset1) == entails(effects));
					//Is sigma when both the pointed and the world we are updating have the same interpretation of the fluent
					if ((e == SIGMA && ent) || (e == TAU && !ent && partially_obs_agents.size() > 0)) {
						add_ste_worlds(ret, *it_kwset1, kmap, e, act);
					}
				}
				break;
			}
			default:
			{
				std::cerr << "Error in executing an action: ";
				std::cerr << "the type of the action is not defined correctly";
				std::cerr << std::endl;
				exit(1);
			}
			}
		}
	}

	// Updating the edges of ret
	kedge_ptr_set::const_iterator it_kedptr;
	event_type_relation::const_iterator it_etr;

	for (it_kedptr = get_edges().begin(); it_kedptr != get_edges().end(); it_kedptr++) {
		agent ag = it_kedptr->get_label();
		event_type_relation e_type_set =
			fully_obs_agents.find(ag) != fully_obs_agents.end() ? fully_obs_r :
			partially_obs_agents.find(ag) != partially_obs_agents.end() ? partially_obs_r :
			oblivious_obs_r;

		for (it_etr = e_type_set.begin(); it_etr != e_type_set.end(); it_etr++) {
			event_type e1 = it_etr->first, e2 = it_etr->second;
			auto kw1 = kmap.find({it_kedptr->get_from(), e1}), kw2 = kmap.find({it_kedptr->get_to(), e2});

			if (kw1 != kmap.end() && kw2 != kmap.end()) {
				ret.add_edge(kedge(kw1->second, kw2->second, ag));
			}
		}
	}

	return ret;
}

kstate kstate::execute_ontic_um(const action & act) const
{
	event_type_set events = {SIGMA, EPSILON};
	event_type_relation fully_obs_r = {
		{SIGMA, SIGMA},
		{EPSILON, EPSILON}
	};
	event_type_relation partially_obs_r = {};
	event_type_relation oblivious_obs_r = {
		{SIGMA, EPSILON},
		{EPSILON, EPSILON}
	};

	return execute_action_um(act, events, fully_obs_r, partially_obs_r, oblivious_obs_r);
}

kstate kstate::execute_sensing_um(const action & act) const
{
	event_type_set events = {SIGMA, TAU, EPSILON};
	event_type_relation fully_obs_r = {
		{SIGMA, SIGMA},
		{TAU, TAU},
		{EPSILON, EPSILON}
	};
	event_type_relation partially_obs_r = {
		{SIGMA, SIGMA},
		{TAU, TAU},
		{EPSILON, EPSILON},
		{SIGMA, TAU},
		{TAU, SIGMA}
	};
	event_type_relation oblivious_obs_r = {
		{SIGMA, EPSILON},
		{TAU, EPSILON},
		{EPSILON, EPSILON}
	};

	return execute_action_um(act, events, fully_obs_r, partially_obs_r, oblivious_obs_r);
}

kstate kstate::execute_announcement_um(const action & act) const
{
	event_type_set events = {SIGMA, TAU, EPSILON};
	event_type_relation fully_obs_r = {
		{SIGMA, SIGMA},
		{TAU, TAU},
		{EPSILON, EPSILON}
	};
	event_type_relation partially_obs_r = {
		{SIGMA, SIGMA},
		{TAU, TAU},
		{EPSILON, EPSILON},
		{SIGMA, TAU},
		{TAU, SIGMA}
	};
	event_type_relation oblivious_obs_r = {
		{SIGMA, EPSILON},
		{TAU, EPSILON},
		{EPSILON, EPSILON}
	};

	return execute_action_um(act, events, fully_obs_r, partially_obs_r, oblivious_obs_r);
}

kstate kstate::execute_ontic(const action & act) const
{
	/** \bug What happen if ontic removes ignorance?
	 * for example:
	 * - act-> i,g then (i,g,h),(-i,g,h),(i,-g,h),(-i,-g,-h) are all equal to (i,g,h) in this case is \ref add_world
	 * - act-> i,g then (-i,g,h) in a "different" k (i,-g,h) then two different states -- ontic shouldn't duplicate worlds
	 * because there are not partial but can inherit the duplicates so
	 *
	 * What if the action does nothing?*/

	//The execution are all the same if we consider that false beliefs don't count.
	kstate ret;

	//This finds all the worlds that are reachable from the initial state following
	//the edges labeled with fully observant agents.
	agent_set agents = domain::get_instance().get_agents();
	agent_set fully_obs_agents = get_agents_if_entailed(act.get_fully_observants(), get_pointed());

	agent_set oblivious_obs_agents = agents;
	minus_set(oblivious_obs_agents, fully_obs_agents);

	//DEBUG PRINT
	/*if (act.get_name().compare("b_check_1") == 0) {
		std::cout << "\nFully observant: ";
		printer::get_instance().print_list_ag(fully_obs_agents);
		std::cout << "\nPartially observant: ";
		printer::get_instance().print_list_ag(partially_obs_agents);
		std::cout << "\nOblivious observant: ";
		printer::get_instance().print_list_ag(oblivious_obs_agents);
	}*/
	agent_set::const_iterator it_agset;

	kworld_ptr_set world_oblivious;
	kworld_ptr_set tmp_world_set;

	kworld_ptr_set::const_iterator it_wo_ob;

	if (oblivious_obs_agents.size() > 0) {
		//ret = (*this);
		tmp_world_set = get_E_reachable_worlds(oblivious_obs_agents, get_pointed());
		for (it_agset = agents.begin(); it_agset != agents.end(); it_agset++) {
			for (it_wo_ob = tmp_world_set.begin(); it_wo_ob != tmp_world_set.end(); it_wo_ob++) {
				sum_set(world_oblivious, get_B_reachable_worlds(*it_agset, *it_wo_ob));
			}
		}
		sum_set(world_oblivious, tmp_world_set);
		ret.set_max_depth(get_max_depth() + 1);
		ret.set_worlds(world_oblivious);
	}



	//pas tmp for entailment
	kworld_ptr_set reached;
	reached.insert(get_pointed());
	std::map<kworld_ptr, kworld_ptr> map_for_edges;

	add_ret_ontic_worlds(get_pointed(), reached, ret, fully_obs_agents, act, domain::get_instance().get_act_check(), map_for_edges);
	kedge_ptr_set::const_iterator it_kedptr;
	std::map<kworld_ptr, kworld_ptr>::const_iterator it_kwmap;

	//The updated edges
	kworld_ptr from_old, to_old;
	kworld_ptr from_new, to_new;
	agent label;
	for (it_kedptr = get_edges().begin(); it_kedptr != get_edges().end(); it_kedptr++) {

		from_old = it_kedptr->get_from();

		if (world_oblivious.find(from_old) != world_oblivious.end()) {
			/**\todo maybe add the check on to_old? if it doesn't exist return error*/
			ret.add_edge(*(it_kedptr->get_ptr()));
		}

		it_kwmap = map_for_edges.find(from_old);
		if (it_kwmap != map_for_edges.end()) {
			from_new = it_kwmap->second;
			to_old = it_kedptr->get_to();
			label = it_kedptr->get_label();
			if (fully_obs_agents.find(label) != fully_obs_agents.end()) {
				it_kwmap = map_for_edges.find(to_old);
				if (it_kwmap != map_for_edges.end()) {
					to_new = it_kwmap->second;
					ret.add_edge(kedge(from_new, to_new, label));
				}
			} else {
				ret.add_edge(kedge(from_new, to_old, label));
			}
		}
	}

	it_kwmap = map_for_edges.find(get_pointed());
	if (it_kwmap != map_for_edges.end()) {
		ret.set_pointed(it_kwmap->second);
	} else {
		std::cerr << "\nInvestigate the case when pointed is not update\n";
		exit(1);
	}

	return ret;
}

kstate kstate::execute_sensing_or_announcement(const action &act, bool sensing) const
{
	/**@bug: We interpret the announced value has in sensing, it has to respect the pointed world????*/
	//The execution are all the same if we consider that false beliefs don't count.
	kstate ret;

	//This finds all the worlds that are reachable from the initial state following
	//the edges labeled with fully observant agents.
	agent_set agents = domain::get_instance().get_agents();

	agent_set fully_obs_agents = get_agents_if_entailed(act.get_fully_observants(), get_pointed());
	agent_set partially_obs_agents = get_agents_if_entailed(act.get_partially_observants(), get_pointed());

	agent_set oblivious_obs_agents = agents;
	minus_set(oblivious_obs_agents, fully_obs_agents);
	minus_set(oblivious_obs_agents, partially_obs_agents);

	agent_set fully_and_partial_ag = fully_obs_agents;
	sum_set(fully_and_partial_ag, partially_obs_agents);

	agent_set::const_iterator it_agset;

	kworld_ptr_set world_oblivious;
	kworld_ptr_set tmp_world_set;

	kworld_ptr_set::const_iterator it_kwset;

	if (oblivious_obs_agents.size() > 0) {
		//ret = (*this);
		tmp_world_set = get_E_reachable_worlds(oblivious_obs_agents, get_pointed());
		for (it_agset = agents.begin(); it_agset != agents.end(); it_agset++) {
			for (it_kwset = tmp_world_set.begin(); it_kwset != tmp_world_set.end(); it_kwset++) {
				sum_set(world_oblivious, get_B_reachable_worlds(*it_agset, *it_kwset));
			}
		}
		sum_set(world_oblivious, tmp_world_set);
		ret.set_max_depth(get_max_depth() + 1);
		ret.set_worlds(world_oblivious);
	}


	///\todo Correct if more than one effects
	fluent_formula effects = get_effects_if_entailed(act.get_effects(), get_pointed());

	/** \todo add multiple fluents sensing
	 * fluent_set tmp_fs;
	 * fluent_set sensed_effects;
	 * fluent tmp_fluent;
	 *
	 * for (it_ff = effects.begin(); it_ff != effects.end(); it_ff++) {
	 *	tmp_fs = *it_ff;
	 *	for (it_fs = tmp_ff.begin(); it_fs != tmp_ff.end(); it_fs++) {
	 *		tmp_fluent = *it_fs;
	 *		if (get_pointed().entails(tmp_fluent)) {
	 *			sensed_effects.insert(tmp_fluent);
	 *		} else {
	 *			sensed_effects.insert(helper::negate_fluent(tmp_fluent));
	 *		}
	 *	}
	 *}*/

	kworld_ptr_set entailing_set = get_C_reachable_worlds(fully_and_partial_ag, get_pointed());
	entailing_set.insert(get_pointed());

	kworld_ptr_set to_duplicate = entailing_set;
	kworld_ptr_set not_entailing_set;
	bool true_sensed = sensing ? get_pointed().entails(effects) : true;

	//	if (act.get_name().compare("tell_a_b2_3") == 0) {
	//		std::cout << "DEBUG: Effects: ";
	//		printer::get_instance().print_list(effects);
	//		std::cout << " Is ";
	//		if (true_sensed) {
	//			std::cout << "True";
	//		} else {
	//			std::cout << "False";
	//		}
	//		std::cout << std::endl;
	//	}

	for (it_kwset = entailing_set.begin(); it_kwset != entailing_set.end();) {
		if ((!(it_kwset->entails(effects)) && true_sensed) || (it_kwset->entails(effects) && !true_sensed)) {
			not_entailing_set.insert(*it_kwset);
			it_kwset = entailing_set.erase(it_kwset);
		} else {
			it_kwset++;
		}
	}

	//	for (it_kwset = entailing_set.begin(); it_kwset != entailing_set.end(); it_kwset++) {
	//		if (act.get_name().compare("tell_a_b2_3") == 0) {
	//
	//			std::cout << "\nKWORLD - ENT (";
	//			printer::get_instance().print_list(it_kwset->get_fluent_set());
	//			std::cout << ")";
	//		}
	//	}
	//
	//	for (it_kwset = not_entailing_set.begin(); it_kwset != not_entailing_set.end(); it_kwset++) {
	//		if (act.get_name().compare("tell_a_b2_3") == 0) {
	//
	//			std::cout << "\nKWORLD - NOT ENT (";
	//			printer::get_instance().print_list(it_kwset->get_fluent_set());
	//			std::cout << ")";
	//		}
	//	}

	//The updated edges
	kworld_ptr from_old, to_old;
	kworld_ptr from_new, to_new;
	agent label;
	kedge_ptr_set::const_iterator it_kedptr;
	for (it_kedptr = get_edges().begin(); it_kedptr != get_edges().end(); it_kedptr++) {


		from_old = it_kedptr->get_from();
		to_old = it_kedptr->get_to();
		label = it_kedptr->get_label();

		//		if (act.get_name().compare("tell_a_b2_3") == 0) {
		//
		//			if (entailing_set.find(from_old) != entailing_set.end() && entailing_set.find(to_old) != entailing_set.end()) {
		//				std::cout << "\n(";
		//				printer::get_instance().print_list(it_kedptr->get_from().get_fluent_set());
		//				std::cout << "," << it_kedptr->get_from().get_repetition();
		//				std::cout << ") - (";
		//				printer::get_instance().print_list(it_kedptr->get_to().get_fluent_set());
		//				std::cout << "," << it_kedptr->get_to().get_repetition();
		//				std::cout << ") ag:" << domain::get_instance().get_grounder().deground_agent(it_kedptr->get_label());
		//				//std::cout << std::endl;
		//			}
		//		}


		///\todo problem with false beliefs
		if (world_oblivious.find(from_old) != world_oblivious.end()) {
			/**\todo maybe add the check on to_old? if it doesn't exist return error*/
			ret.add_edge(*(it_kedptr->get_ptr()));
		}
		if (fully_obs_agents.find(label) != fully_obs_agents.end()) {

			//			if (act.get_name().compare("tell_a_b2_3") == 0) {
			//				if (entailing_set.find(from_old) != entailing_set.end()) {
			//					if (entailing_set.find(to_old) != entailing_set.end()) {
			//						std::cout << " F-";
			//					}
			//				}
			//			}

			/**\todo maybe faster to check the entailment directly*/
			if ((entailing_set.find(from_old) != entailing_set.end() && not_entailing_set.find(to_old) != not_entailing_set.end())
				|| (entailing_set.find(to_old) != entailing_set.end() && not_entailing_set.find(from_old) != not_entailing_set.end())) {
				///\todo ottimiza
				if (partially_obs_agents.size() > 0) {
					//from_new = ret.add_rep_world(*(from_old.get_ptr()), from_old.get_repetition());
					to_new = ret.add_rep_world(*(to_old.get_ptr()), to_old.get_repetition());
					//ret.add_edge(kedge(from_new, from_new, label));
					ret.add_edge(kedge(to_new, to_new, label));
				}
			} else if (entailing_set.find(from_old) != entailing_set.end() && entailing_set.find(to_old) != entailing_set.end()) {
				///\todo ottimiza
				if ((to_duplicate.find(from_old) != to_duplicate.end())) {
					from_new = ret.add_rep_world(*(from_old.get_ptr()), from_old.get_repetition());
					to_new = ret.add_rep_world(*(to_old.get_ptr()), to_old.get_repetition());
					/**\todo add also (from,from)?(to,to)?*/
					ret.add_edge(kedge(from_new, to_new, label));
				}
			} else if (not_entailing_set.find(from_old) != not_entailing_set.end() && not_entailing_set.find(to_old) != not_entailing_set.end()) {
				if (partially_obs_agents.size() > 0) {
					if ((to_duplicate.find(from_old) != to_duplicate.end())) {
						from_new = ret.add_rep_world(*(from_old.get_ptr()), from_old.get_repetition());
						to_new = ret.add_rep_world(*(to_old.get_ptr()), to_old.get_repetition());
						/**\todo add also (from,from)?(to,to)?*/
						ret.add_edge(kedge(from_new, to_new, label));
					}
				}
			}
		} else if (partially_obs_agents.find(label) != partially_obs_agents.end()) {
			//if (oblivious_obs_agents.size() > 1) {
			if ((to_duplicate.find(from_old) != to_duplicate.end()) || (to_duplicate.find(to_old) != to_duplicate.end())) {
				from_new = ret.add_rep_world(*(from_old.get_ptr()), from_old.get_repetition());
				to_new = ret.add_rep_world(*(to_old.get_ptr()), to_old.get_repetition());
				ret.add_edge(kedge(from_new, to_new, label));
			}

		} else {
			//if (oblivious_obs_agents.size() > 1) {
			//if (world_oblivious.find(from_old) == world_oblivious.end()) {
			//\todo to check
			if ((to_duplicate.find(from_old) != to_duplicate.end()) || (to_duplicate.find(to_old) != to_duplicate.end())) {
				from_new = ret.add_rep_world(*(from_old.get_ptr()), from_old.get_repetition());
				ret.add_edge(kedge(from_new, to_old, label));
			}
			//ret.add_edge(kedge(from_new, to_old, label));
			/* Remove edge if not reachable by partial and not entails
			 * if (partial_size > 1 || entailing_set.find(from_old) != entailing_set.end()) {
			}
			if (partial_size > 1 || entailing_set.find(to_old) != entailing_set.end()) {
			}*/
		}
	}

	kworld_ptr pointed_tmp = get_pointed();
	//if (oblivious_obs_agents.size() > 1) {
	//if (world_oblivious.find(pointed_tmp) == world_oblivious.end()) {
	ret.set_pointed(ret.add_rep_world(*(pointed_tmp.get_ptr()), pointed_tmp.get_repetition()));
	//} else {
	//Not really sure, what if not directly reachable?
	//	ret.set_pointed(pointed_tmp);
	//}


	return ret;
}

kstate kstate::execute_sensing(const action & act) const
{
	/** We assume that if you sense something that is in contrast with your belief the sensing wins.
	 * Executing a sensing then is just:
	 * - execute an ontic for the fully observants.
	 * - link the new worlds (derived from the ontic) with copies of old (replica of the old) state for the partially.
	 * - Link this copies with themselves for fully and partially.
	 * - Link the old state for oblivious.*/
	/**\bug Wrong because the fluent changes accordingly to the pointed world*/

	return execute_sensing_or_announcement(act, true);
}

kstate kstate::execute_announcement(const action & act) const
{
	/** We assume that if you announce something everyone believes at it, so:
	 * - execute an ontic for the fully observants.
	 * - link the new worlds (derived from the ontic) with copies of old (replica of the old) state for the partially.
	 * - Link this copies with themselves for fully and partially.
	 * - Link the old state for oblivious.*/
	/**\bug Wrong because the fluent changes accordingly to the pointed world*/

	///Check who execute the action what believes bot what is true in pointed

	return execute_sensing_or_announcement(act, false);
}

void kstate::print() const
{
	int counter = 1;
	std::cout << std::endl;
	std::cout << "The Pointed World has id ";
	printer::get_instance().print_list(get_pointed().get_fluent_set());
	std::cout << std::endl;
	std::cout << "*******************************************************************" << std::endl;
	;
	kworld_ptr_set::const_iterator it_kwset;
	std::cout << "World List:" << std::endl;

	for (it_kwset = get_worlds().begin(); it_kwset != get_worlds().end(); it_kwset++) {
		std::cout << "W-" << counter << ": ";
		printer::get_instance().print_list(it_kwset->get_fluent_set());
		std::cout << " rep:" << it_kwset->get_repetition();
		std::cout << std::endl;
		counter++;
	}
	counter = 1;
	std::cout << std::endl;
	std::cout << "*******************************************************************" << std::endl;
	kedge_ptr_set::const_iterator it_keset;
	std::cout << "Edge List:" << std::endl;
	for (it_keset = get_edges().begin(); it_keset != get_edges().end(); it_keset++) {

		std::cout << "E-" << counter << ": (";
		printer::get_instance().print_list(it_keset->get_from().get_fluent_set());
		std::cout << "," << it_keset->get_from().get_repetition();
		std::cout << ") - (";
		printer::get_instance().print_list(it_keset->get_to().get_fluent_set());
		std::cout << "," << it_keset->get_to().get_repetition();
		std::cout << ") ag:" << domain::get_instance().get_grounder().deground_agent(it_keset->get_label());
		std::cout << std::endl;
		counter++;

	}
	std::cout << "*******************************************************************" << std::endl;
}



void kstate::print_ML_dataset(std::ostream& graphviz) const
{
	std::cerr << "Function print_ML_dataset not yet implmented for K structures.";
	exit(1);
}


void kstate::print_graphviz_explicit(std::ostream & graphviz) const
{
	std::cerr << "Function print_graphviz_explicit not yet implmented for K structures.";
	exit(1);
}


void kstate::print_graphviz(std::ostream & graphviz) const
{
	string_set::const_iterator it_st_set;
	fluent_set::const_iterator it_fs;


	graphviz << "//WORLDS List:" << std::endl;

	kworld_ptr_set::const_iterator it_kpset;
	std::map<fluent_set, int> map_world_to_index;
	std::map<unsigned short, char> map_rep_to_name;
	char found_rep = (char) ((char) domain::get_instance().get_agents().size() + 'A');
	int found_fs = 0;
	fluent_set tmp_fs;
	char tmp_unsh;
	string_set tmp_stset;
	bool print_first;
	kworld_ptr_set::const_iterator it_kwset;
	for (it_kwset = get_worlds().begin(); it_kwset != get_worlds().end(); it_kwset++) {
		if (*it_kwset == get_pointed())
			graphviz << "	node [shape = doublecircle] ";
		else
			graphviz << "	node [shape = circle] ";

		print_first = false;
		tmp_fs = it_kwset->get_fluent_set();
		if (map_world_to_index.count(tmp_fs) == 0) {
			map_world_to_index[tmp_fs] = found_fs;
			found_fs++;
		}
		tmp_unsh = it_kwset->get_repetition();
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

	std::map<int, kworld_ptr_set> for_rank_print;
	for (it_kwset = get_worlds().begin(); it_kwset != get_worlds().end(); it_kwset++) {
		for_rank_print[it_kwset->get_repetition()].insert(*it_kwset);
	}

	std::map<int, kworld_ptr_set>::const_iterator it_map_rank;
	for (it_map_rank = for_rank_print.begin(); it_map_rank != for_rank_print.end(); it_map_rank++) {
		graphviz << "	{rank = same; ";
		for (it_kwset = it_map_rank->second.begin(); it_kwset != it_map_rank->second.end(); it_kwset++) {
			graphviz << "\"" << map_rep_to_name[it_kwset->get_repetition()] << "_" << map_world_to_index[it_kwset->get_fluent_set()] << "\"; ";
		}
		graphviz << "}\n";
	}


	graphviz << "\n\n";
	graphviz << "//EDGES List:" << std::endl;

	std::map < std::tuple<std::string, std::string>, std::set<std::string> > edges;

	kedge_ptr_set::const_iterator it_keset;
	std::tuple<std::string, std::string> tmp_tuple;
	std::string tmp_string = "";
	for (it_keset = get_edges().begin(); it_keset != get_edges().end(); it_keset++) {
		tmp_string = "_" + std::to_string(map_world_to_index[it_keset->get_from().get_fluent_set()]);
		tmp_string.insert(0, 1, map_rep_to_name[it_keset->get_from().get_repetition()]);
		std::get<0>(tmp_tuple) = tmp_string;

		tmp_string = "_" + std::to_string(map_world_to_index[it_keset->get_to().get_fluent_set()]);
		tmp_string.insert(0, 1, map_rep_to_name[it_keset->get_to().get_repetition()]);
		std::get<1>(tmp_tuple) = tmp_string;

		edges[tmp_tuple].insert(domain::get_instance().get_grounder().deground_agent(it_keset->get_label()));
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

	
	std::string color = "<font color=\"#ffffff\">";
	graphviz << "\n\n//WORLDS description Table:" << std::endl;
	graphviz << "	node [shape = plain]\n\n";
	graphviz << "	description[label=<\n";
	graphviz << "	<table border = \"0\" cellborder = \"1\" cellspacing = \"0\" >\n";
	for (it_kwset = get_worlds().begin(); it_kwset != get_worlds().end(); it_kwset++) {
		tmp_fs = it_kwset->get_fluent_set();
		print_first = false;
		graphviz << "		<tr><td>" << map_rep_to_name[it_kwset->get_repetition()] << "_" << map_world_to_index[tmp_fs] << "</td> <td>";
		for (it_fs = tmp_fs.begin(); it_fs != tmp_fs.end(); it_fs++) {
			if (print_first) {
				graphviz << ", ";
			}
			print_first = true;
			if(!helper::is_negate(*it_fs)) color = "<font color=\"#0000ff\"> ";
			else color = "<font color=\"#ff1020\">";
			graphviz << color << domain::get_instance().get_grounder().deground_fluent(*it_fs) << "</font>";
		}
		graphviz << "</td></tr>\n";
	}
	graphviz << "	</table>>]\n";
	graphviz << "	{rank = max; description};\n";

}

/******************************MOVE TO HELPER*********************************/
template <class T>
void kstate::sum_set(std::set<T> & to_modify, const std::set<T> & factor2) const
{
	/**\todo move to helper*/
	typename std::set<T>::const_iterator it_kwset;
	for (it_kwset = factor2.begin(); it_kwset != factor2.end(); it_kwset++) {

		to_modify.insert(*it_kwset);
	}
}

template <class T>
void kstate::minus_set(std::set<T> & to_modify, const std::set<T> & factor2) const
{ /**\todo move to helper*/

	typename std::set<T>::const_iterator it_kwset;
	for (it_kwset = factor2.begin(); it_kwset != factor2.end(); it_kwset++) {

		to_modify.erase(*it_kwset);
	}
}

agent_set kstate::get_agents_if_entailed(const observability_map& map, const kworld_ptr & start) const
{ /**\todo move to helper*/

	agent_set ret;
	observability_map::const_iterator it_map;
	for (it_map = map.begin(); it_map != map.end(); it_map++) {
		if (entails(it_map->second, start)) {

			ret.insert(it_map->first);
		}
	}
	return ret;
}

fluent_formula kstate::get_effects_if_entailed(const effects_map & map, const kworld_ptr & start) const
{ /**\todo move to helper*/

	fluent_formula ret;
	effects_map::const_iterator it_map;
	for (it_map = map.begin(); it_map != map.end(); it_map++) {
		if (entails(it_map->second, start)) {
			ret = helper::and_ff(ret, it_map->first);
		}
	}
	if (ret.size() > 1) {

		std::cerr << "\nNon determinism in action effect is not supported-1.\n";
		exit(1);
	}
	return ret;
}

/*fluent_formula kstate::get_sensing_effects_if_entailed(const effects_map & map, const kworld_ptr & start) const
{ 	fluent_formula ret;
	fluent_formula retret;
	fluent_set ret_tmp;

	effects_map::const_iterator it_map;
	for (it_map = map.begin(); it_map != map.end(); it_map++) {
		if (entails(it_map->second, start)) {
			if (start.get_ptr()->entails(it_map->first))
				ret = helper::and_ff(ret, it_map->first);
			else {
				fluent_formula tmpp = it_map->first;
				if (tmpp.size() == 1) {
					fluent_set tmp = *(tmpp.begin());
					if (tmp.size() == 1) {
						fluent temp = *(tmp.begin());
						if (temp % 2 == 0) {
							temp = temp + 1;
						} else {
							temp = temp - 1;
						}
						ret_tmp.insert(temp);
						retret.insert(ret_tmp);

					} else {
						std::cerr << "DEBUG--ERROR in sensing;";
						exit(1);
					}
				} else {
					std::cerr << "DEBUG--ERROR in sensing;";
					exit(1);
				}
				ret = helper::and_ff(ret, retret);
				retret.erase(ret_tmp);
			}

		}

	}
	if (ret.size() > 1) {

		std::cerr << "\nNon determinism in action effect is not supported-2.\n";
		exit(1);
	}
	return ret;
}*/


//void kstate::DEBUG_add_extra_world()
//{
//
//	kworld_ptr tmp_ptr = *m_worlds.begin();
//	kworld extra(tmp_ptr.get_fluent_set());
//	unsigned short depth = tmp_ptr.get_repetition() + 4;
//	//std::cout << "\n\nDEBUG: Depth: " << depth << "\n\n";
//	kworld_ptr extra_ptr = add_rep_world(extra, depth);
//	set_max_depth(get_max_depth() + depth);
//
//
//
//
//	kedge_ptr_set::const_iterator it_kep;
//	for (it_kep = m_edges.begin(); it_kep != m_edges.end(); it_kep++) {
//		if (it_kep->get_from() == tmp_ptr && !(it_kep->get_to() == tmp_ptr)) {
//			kedge extra_ed(extra_ptr, it_kep->get_to(), it_kep->get_label());
//			add_edge(extra_ed);
//		} else if (it_kep->get_to() == tmp_ptr && !(it_kep->get_from() == tmp_ptr)) {
//			kedge extra_ed(it_kep->get_from(), extra_ptr, it_kep->get_label());
//			add_edge(extra_ed);
//		} else if (it_kep->get_to() == tmp_ptr && it_kep->get_from() == tmp_ptr) {
//			kedge extra_ed(extra_ptr, extra_ptr, it_kep->get_label());
//			add_edge(extra_ed);
//		}
//	}
//
//}








//DEBUG

//void kstate::debug_print(const kstate & to_compare)
//{
//
//	bisimulation b;
//	std::vector<kworld_ptr> kworld_vec;
//	automa a;
//
//	//std::cerr << "\nDEBUG:HERE\n";
//
//
//	if (m_edges < to_compare.get_edges()) {
//		//std::cerr << "\nDEBUG:HERE 1\n";
//
//		a = b.compare_automata_debug(*this, to_compare, kworld_vec);
//	} else {
//		//std::cerr << "\nDEBUG:HERE 1\n";
//
//		a = b.compare_automata_debug(to_compare, *this, kworld_vec);
//	}
//	automaton_to_kstate_debug(a, kworld_vec);
//}
//
//void kstate::automaton_to_kstate_debug(const automa & a, std::vector<kworld_ptr> & kworld_vec)
//{
//	kworld_ptr_set worlds;
//	kedge_ptr_set edges;
//	// The pointed world does not change when we calculate the minimum bisimilar state
//	// Hence we do not need to update it
//
//	int i, j, k, label, agents_size = domain::get_instance().get_agents().size();
//
//	for (i = 0; i < a.Nvertex; i++) {
//		if (a.Vertex[i].ne > 0) {
//			kworld_vec[i].set_repetition(i);
//			std::cerr << "\nDEBUG: Analyzing the " << i << "^th world with num_id = " << kworld_vec[i].get_numerical_id() << " and repetition = " << kworld_vec[i].get_repetition() << std::endl;
//			worlds.insert(kworld_vec[i]);
//
//			for (j = 0; j < a.Vertex[i].ne; j++) {
//				for (k = 0; k < a.Vertex[i].e[j].nbh; k++) {
//					label = a.Vertex[i].e[j].bh[k];
//					if (label < agents_size) {
//						
//							kedge_ptr tmp_edge = kstore::get_instance().add_edge(kedge(kworld_vec[i], kworld_vec[a.Vertex[i].e[j].tv], label));
//							edges.insert(tmp_edge);
//							//tmp_edge.print();
//							//std::cerr << "\n";
//						
//					}
//				}
//			}
//		}
//
//	}
//
//	set_pointed(*worlds.begin());
//	
//	set_worlds(worlds);
//	set_edges(edges);
//}