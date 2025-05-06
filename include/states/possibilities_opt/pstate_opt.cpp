/**
 * \brief Implementation of \ref pstate_opt.h and \ref pstate_opt_ptr.h.
 *
 * \copyright GNU Public License.
 * 
 * @todo: Redefine the insert so that we have the bulk check (Bisimulation with all the graphs). 
 *
 * \author Francesco Fabiano.
 * \date September 14, 2019
 */

#include <iostream>
#include <tuple>
#include <stdexcept>

#include "pstate_opt.h"
#include "pstore_opt.h"

#include "../../domain/domain.h"
#include "../../utilities/helper.h"

pstate_opt::pstate_opt()
{
}

pstate_opt::pstate_opt(const fluent_set & description)
{
	/*
	 * No need because consistent for construction
	 * \throw std::invalid_argument whenever \p description is not consistent.
	 *
	 *try {*/
	set_fluent_set(description);
	set_id();
	/*} catch (const std::invalid_argument& ia) {
	 *	throw ia;
	 *} */
}

pstate_opt::pstate_opt(const fluent_set & description, const pedges_opt & edges)
{
	/*
	 * No need because consistent for construction
	 * \throw std::invalid_argument whenever \p description is not consistent.
	 *
	 *try {*/
	set_fluent_set(description);
	set_edges(edges);
	set_id();
	/*} catch (const std::invalid_argument& ia) {
	 *	throw ia;
	 *} */
}

//pstate_opt::pstate_opt(const fluent_set & description, const pedges_opt & edges, unsigned short repetition)
//{
//	/*
//	 * No need because consistent for construction
//	 * \throw std::invalid_argument whenever \p description is not consistent.
//	 *
//	 *try {*/
//	set_fluent_set(description);
//	set_edges(edges);
//	set_repetition(repetition);
//	set_id();
//	/*} catch (const std::invalid_argument& ia) {
//	 *	throw ia;
//	 *} */
//}

pstate_opt::pstate_opt(const pstate_opt & state)
{
	set_fluent_set(state.get_fluent_set());
	set_edges(state.get_edges());
	//set_repetition(state.get_repetition());
	set_id();
}

pstate_opt_id pstate_opt::hash_fluents_into_id(const fluent_set& fl)
{
	pstate_opt_id ret;
	fluent_set::const_iterator it_fl;
	for (it_fl = fl.begin(); it_fl != fl.end(); it_fl++) {
		ret.append(std::to_string(*it_fl) + "-");
	}
	//ret.append(std::to_string(get_repetition()));
	//Remove last "-" that is useless
	//ret.pop_back();
	return ret;
}

pstate_opt_id pstate_opt::hash_fluents_into_id()
{
	return hash_fluents_into_id(m_fluent_set);
}

void pstate_opt::set_fluent_set(const fluent_set & description)
{
	/*
	 * \throw std::invalid_argument whenever \p description is not consistent.
	 *
	if (consistent(description))*/
	m_fluent_set = description;
	/*else
		throw std::invalid_argument("Non consistent set of fluent");*/
}

void pstate_opt::set_id()
{
	m_id = hash_fluents_into_id();
}

void pstate_opt::set_edges(const pedges_opt & edges)
{
	m_edges = edges;
}

//void pstate_opt::set_repetition(unsigned short repetition)
//{
//	m_repetition = repetition;
//}

const fluent_set& pstate_opt::get_fluent_set() const
{
	return m_fluent_set;
}

const pedges_opt& pstate_opt::get_edges() const
{
	return m_edges;
}

pstate_opt_id pstate_opt::get_id() const
{
	return m_id;
}
//
//void pstate_opt::get_repetition() const
//{
//	return m_repetition;
//}

bool pstate_opt::operator=(const pstate_opt & to_copy)
{

	set_fluent_set(to_copy.get_fluent_set());
	set_edges(to_copy.get_edges());
	//	set_repetition(to_copy.get_repetition());
	set_id();
	return true;
}

bool pstate_opt::entails(fluent to_check) const
{
	return(m_fluent_set.find(to_check) != m_fluent_set.end());
}

bool pstate_opt::entails(const fluent_set & to_check) const
{
	if (to_check.size() == 0) {
		return true;
	}
	fluent_set::const_iterator it_fl;
	for (it_fl = to_check.begin(); it_fl != to_check.end(); it_fl++) {
		if (!entails(*it_fl)) {
			return false;
		}
	}
	return true;
}

bool pstate_opt::entails(const fluent_formula & to_check) const
{
	if (to_check.size() == 0) {
		return true;
	}
	fluent_formula::const_iterator it_fl;
	for (it_fl = to_check.begin(); it_fl != to_check.end(); it_fl++) {
		if (entails(*it_fl)) {
			return true;
		}
	}
	return false;
}

bool pstate_opt::entails(const belief_formula & to_check, const pstate_opt_ptr_set & reachable) const
{
	pstate_opt_ptr_set::const_iterator it_kwl;
	for (it_kwl = reachable.begin(); it_kwl != reachable.end(); it_kwl++) {
		/**\todo why setted contary?*/
		if (!((*it_kwl).entails(to_check)))
			return false;
	}
	return true;
}

bool pstate_opt::entails(const belief_formula & bf) const
{
	/*
	 The entailment of a \ref belief_formula just call recursively the entailment on all the reachable state with that formula.
	 */
	switch ( bf.get_formula_type() ) {

	case FLUENT_FORMULA:
	{
		/** \todo Make sure its grounded. Maybe add to \ref belief_formula a bool that store if grounded or not or maybe ground
		 * when \ref domain created.
		 * @see belief_formula::ground(const grounder &).*/
		return entails(bf.get_fluent_formula());
		break;
	}

	case BELIEF_FORMULA:

		/** \todo what was the at_lest_one of the previous version?*/
		return entails(bf.get_bf1(), get_B_reachable_states(bf.get_agent()));
		break;

	case PROPOSITIONAL_FORMULA:
		switch ( bf.get_operator() ) {
		case BF_NOT:
			return !entails(bf.get_bf1());
			break;
		case BF_OR:
			return entails(bf.get_bf1()) || entails(bf.get_bf2());
			break;
		case BF_AND:
			return entails(bf.get_bf1()) && entails(bf.get_bf2());
			break;
		case BF_FAIL:
		default:
			std::cerr << "Something went wrong in checking entailment for Propositional formula";
			exit(1);
		}
		break;

	case E_FORMULA:
		//Check the entails on the E-reachable states
		return entails(bf.get_bf1(), get_E_reachable_states(bf.get_group_agents()));
		break;

		//Check the entails on the C-reachable states
	case C_FORMULA:
		return entails(bf.get_bf1(), get_C_reachable_states(bf.get_group_agents()));
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
}

bool pstate_opt::entails(const formula_list & to_check) const
{
	//formula_list expresses CNF formula
	formula_list::const_iterator it_fl;
	for (it_fl = to_check.begin(); it_fl != to_check.end(); it_fl++) {
		if (!entails((*it_fl))) {
			return false;
		}
	}
	return true;
}

const pstate_opt_ptr_set pstate_opt::get_B_reachable_states(agent ag) const
{
	pstate_opt_ptr_set ret;
	get_B_reachable_states(ag, ret);

	return ret;
}

bool pstate_opt::get_B_reachable_states(agent ag, pstate_opt_ptr_set& ret) const
{
	return false;
}

const pstate_opt_ptr_set pstate_opt::get_E_reachable_states(const agent_set & ags) const
{
	pstate_opt_ptr_set ret;
	get_E_reachable_states(ags, ret);

	return ret;
}

bool pstate_opt::get_E_reachable_states(const agent_set & ags, pstate_opt_ptr_set& ret) const
{
	bool is_fixed_point = true;
	agent_set::const_iterator it_agset;
	for (it_agset = ags.begin(); it_agset != ags.end(); it_agset++) {
		if (!get_B_reachable_states(*it_agset, ret)) {

			is_fixed_point = false;
		}
	}
	return is_fixed_point;
}

const pstate_opt_ptr_set pstate_opt::get_C_reachable_states(const agent_set & ags) const
{
	//Use of fixed point to stop.
	bool is_fixed_point = false;
	pstate_opt_ptr_set ret;
	while (!is_fixed_point) {
		is_fixed_point = get_E_reachable_states(ags, ret);
	}
	return ret;
}

void pstate_opt::build_initial()
{
	/** \todo for now prune building.*/
	std::cout << "\nBuilding initial possibility...\n";
	build_initial_prune();
}

void pstate_opt::build_initial_structural()
{

}

void pstate_opt::build_initial_prune()
{
	/*Building of all the possible consistent \ref pstate_opt and setting the pointed state.
	 * Creation of all the \ref fluent combinations. All the consistent
	 * ones are added to \ref pstore_opt.*/
	fluent_set permutation;
	pstate_opt_ptr_set initial_pstate_opts;
	generate_initial_pstate_opts(permutation, 0, initial_pstate_opts);


	/*Building of all the consistent edges.*/
	generate_initial_pedges(initial_pstate_opts);

	/*Setting up the pointed state*/
	pstate_opt_ptr_set::const_iterator it_pss_1;
	initially ini_conditions = domain::get_instance().get_initial_description();

	for (it_pss_1 = initial_pstate_opts.begin(); it_pss_1 != initial_pstate_opts.end(); it_pss_1++) {
		if (it_pss_1->entails(ini_conditions.get_pointed_world_conditions())) {
			pstate_opt_ptr tmp_ptr = *it_pss_1;
			set_fluent_set(tmp_ptr.get_fluent_set());
			set_edges(tmp_ptr.get_edges());
			set_id();
		}
	}
}

/*From https://www.geeksforgeeks.org/generate-all-the-binary-strings-of-n-bits/ since is like generating all the binary numbers.*/
void pstate_opt::generate_initial_pstate_opts(fluent_set& permutation, int index, pstate_opt_ptr_set& ret)
{
	initially ini_conditions = domain::get_instance().get_initial_description();
	int fluent_number = domain::get_instance().get_fluent_number();
	if (index / 2 == fluent_number) {
		pstate_opt_ptr to_add = pstate_opt_ptr(pstate_opt(permutation));
		switch ( ini_conditions.get_ini_restriction() ) {
		case S5:
		{
			/* Since the common knowledge is on all the agent it means that every possible \ref pstate_opt
			 * in the initial state must entail *phi* where C(*phi*) is an initial condition.*/

			//Already setted in \ref domain::build_initially(bool).
			if (to_add.entails(ini_conditions.get_ff_forS5())) {
				ret.insert(to_add);
			}
			break;
		}
		case K45:
		case NONE:
		default:
		{
			std::cerr << "\nNot implemented yet\n";
			exit(1);

			break;
		}
		}
		return;
	}
	fluent_set permutation_2 = permutation;
	//Add the \ref fluent in positive version
	permutation.insert(index);
	generate_initial_pstate_opts(permutation, index + 2, ret);

	//Add the \ref fluent in negative version
	permutation_2.insert(index + 1);
	generate_initial_pstate_opts(permutation_2, index + 2, ret);
}

void pstate_opt::generate_initial_pedges(pstate_opt_ptr_set& initial_pstate_opts)
{
	pstate_opt_ptr_set::iterator it_pss_1, it_pss_2;

	/*This for add to *this* all the possible edges.*/
	for (it_pss_1 = initial_pstate_opts.begin(); it_pss_1 != initial_pstate_opts.end(); it_pss_1++) {
		for (it_pss_2 = it_pss_1; it_pss_2 != initial_pstate_opts.end(); it_pss_2++) {
			for (unsigned int i = 0; i < domain::get_instance().get_agents().size(); i++) {
				(*it_pss_1).get_ptr()->add_edge(*it_pss_2, i);
				(*it_pss_2).get_ptr()->add_edge(*it_pss_1, i);
			}
		}
	}

	//std::cout << "Tot edges: " << m_edges.size() << std::endl;

	initially ini_conditions = domain::get_instance().get_initial_description();

	formula_list::const_iterator it_fl;
	for (it_fl = ini_conditions.get_initial_conditions().begin(); it_fl != ini_conditions.get_initial_conditions().end(); it_fl++) {

		remove_initial_pedge_bf(*it_fl, initial_pstate_opts);
	}
	//std::cout << "Removed edges: " << count << std::endl;

	//std::cout << "Final edges: " << m_edges.size() << std::endl;

}

void pstate_opt::add_edge(const pstate_opt_ptr &to, agent ag)
{
	m_edges[ag].insert(to);
}

void pstate_opt::remove_edge(const pstate_opt_ptr &to, const agent ag)
{

	m_edges[ag].erase(to);

}

void pstate_opt::remove_initial_pedge(const fluent_formula &known_ff, agent ag, pstate_opt_ptr_set& initial_pstate_opts)
{
	pstate_opt_ptr_set::iterator it_pss_1, it_pss_2;

	for (it_pss_1 = initial_pstate_opts.begin(); it_pss_1 != initial_pstate_opts.end(); it_pss_1++) {
		for (it_pss_2 = it_pss_1; it_pss_2 != initial_pstate_opts.end(); it_pss_2++) {
			/** \todo or entails(-known_ff)?*/
			if ((*it_pss_1).entails(known_ff) && !((*it_pss_2).entails(known_ff))) {
				(*it_pss_1).get_ptr()->remove_edge(*it_pss_2, ag);
				(*it_pss_2).get_ptr()->remove_edge(*it_pss_1, ag);
			} else if ((*it_pss_2).entails(known_ff) && !((*it_pss_1).entails(known_ff))) {
				(*it_pss_2).get_ptr()->remove_edge(*it_pss_1, ag);
				(*it_pss_1).get_ptr()->remove_edge(*it_pss_2, ag);
			}
		}
	}
}

void pstate_opt::remove_initial_pedge_bf(const belief_formula &to_check, pstate_opt_ptr_set& initial_pstate_opts)
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
						remove_initial_pedge(*known_ff_ptr, tmp.get_bf2().get_agent(), initial_pstate_opts);
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
	case NONE:
	default:
	{
		std::cerr << "\nNot implemented yet\n";
		exit(1);

		break;
	}
	}
}

/** \warning executability should be check in \ref state (or \ref planner).*/
pstate_opt pstate_opt::compute_succ(const action & act) const
{
	switch ( act.get_type() ) {
	case ONTIC:
	{
		return execute_ontic(act);
		break;
	}
	case SENSING:
	{
		return execute_sensing(act);
		break;
	}
	case ANNOUNCEMENT:
	{
		return execute_announcement(act);
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

bool pstate_opt::operator<(const pstate_opt & to_compare) const
{
	///BISIMULATION HERE
	//The leaves are the states with the smallest id
	return false;
}



void pstate_opt::execute_action(const action &act, pstate_opt &ret, agent_set &fully_obs_agents, agent_set &oblivious_obs_agents) const
{

}

pstate_opt_ptr pstate_opt::execute_action_helper(const action &act, pstate_opt &ret, const pstate_opt_ptr &current_pw, agent_set &fully_obs_agents, agent_set &oblivious_obs_agents) const
{

	pstate_opt_ptr new_pw;
	return new_pw;
}

pstate_opt pstate_opt::execute_ontic(const action &act) const
{
	pstate_opt ret;
	return ret;
}

pstate_opt pstate_opt::execute_sensing(const action &act) const
{
	pstate_opt ret;
	return ret;
}

pstate_opt pstate_opt::execute_announcement(const action &act) const
{
	pstate_opt ret;
	return ret;
}

void pstate_opt::print() const
{

}

void pstate_opt::print_graphviz(std::ostream & graphviz) const
{

}

/******************************MOVE TO HELPER*********************************/

template <class T>
void pstate_opt::sum_set(std::set<T> & to_modify, const std::set<T> & factor2) const
{
	/**\todo move to helper*/
	typename std::set<T>::const_iterator it_kwset;
	for (it_kwset = factor2.begin(); it_kwset != factor2.end(); it_kwset++) {

		to_modify.insert(*it_kwset);
	}
}

template <class T>
void pstate_opt::minus_set(std::set<T> & to_modify, const std::set<T> & factor2) const
{ /**\todo move to helper*/

	typename std::set<T>::const_iterator it_kwset;
	for (it_kwset = factor2.begin(); it_kwset != factor2.end(); it_kwset++) {

		to_modify.erase(*it_kwset);
	}
}

agent_set pstate_opt::get_agents_if_entailed(const observability_map& map) const
{ /**\todo move to helper*/

	agent_set ret;
	observability_map::const_iterator it_map;
	for (it_map = map.begin(); it_map != map.end(); it_map++) {
		if (entails(it_map->second)) {

			ret.insert(it_map->first);
		}
	}
	return ret;
}

fluent_formula pstate_opt::get_effects_if_entailed(const effects_map & map) const
{ /**\todo move to helper*/

	fluent_formula ret;
	effects_map::const_iterator it_map;
	for (it_map = map.begin(); it_map != map.end(); it_map++) {
		if (entails(it_map->second)) {
			ret = helper::and_ff(ret, it_map->first);
		}
	}
	if (ret.size() > 1) {

		std::cerr << "\nNon determinism in action effect is not supported-1.\n";
		exit(1);
	}
	return ret;
}

/*\*********************************************************************************************/

pstate_opt_ptr::pstate_opt_ptr()
{
}

pstate_opt_ptr::pstate_opt_ptr(const std::shared_ptr<pstate_opt> & ptr)
{
	set_ptr(ptr);
}

pstate_opt_ptr::pstate_opt_ptr(std::shared_ptr<pstate_opt>&& ptr)
{
	set_ptr(ptr);
}

pstate_opt_ptr::pstate_opt_ptr(const pstate_opt & state)
{
	m_ptr = std::make_shared<pstate_opt>(state);
}

void pstate_opt_ptr::set_ptr(const std::shared_ptr<pstate_opt> & ptr)
{
	m_ptr = ptr;
}

void pstate_opt_ptr::set_ptr(std::shared_ptr<pstate_opt>&& ptr)
{
	m_ptr = ptr;
}

std::shared_ptr<pstate_opt> pstate_opt_ptr::get_ptr() const
{
	return m_ptr;
}

const fluent_set & pstate_opt_ptr::get_fluent_set() const
{
	if (m_ptr != nullptr) {
		return get_ptr()->get_fluent_set();
	}
	std::cerr << "Error in creating a pstate_opt_ptr\n";
	exit(1);
}

pstate_opt_id pstate_opt_ptr::get_id() const
{
	if (m_ptr != nullptr) {
		return(get_ptr()->get_id());
	}
	std::cerr << "\nError in creating a pstate_opt_ptr\n";
	exit(1);
}

const pedges_opt & pstate_opt_ptr::get_edges() const
{
	if (m_ptr != nullptr) {
		return(get_ptr()->get_edges());
	}
	std::cerr << "\nError in creating a pstate_opt_ptr\n";
	exit(1);

}

bool pstate_opt_ptr::entails(fluent to_check) const
{
	return m_ptr->entails(to_check);
}

bool pstate_opt_ptr::entails(const fluent_set& to_check) const
{
	return m_ptr->entails(to_check);
}

bool pstate_opt_ptr::entails(const fluent_formula & to_check) const
{
	return m_ptr->entails(to_check);
}

bool pstate_opt_ptr::entails(const belief_formula & to_check) const
{
	return m_ptr->entails(to_check);
}

bool pstate_opt_ptr::entails(const formula_list & to_check) const
{
	return m_ptr->entails(to_check);
}

bool pstate_opt_ptr::operator<(const pstate_opt_ptr & to_compare) const
{
	if (*(get_ptr()) < *(to_compare.get_ptr())) {
		return true;
	}
	return false;
}

bool pstate_opt_ptr::operator==(const pstate_opt_ptr & to_compare) const
{
	/**std way*/
	if (!((*this) < to_compare) && !(to_compare < (*this))) {
		return true;
	}
	return false;
}

bool pstate_opt_ptr::operator=(const pstate_opt_ptr & to_copy)
{
	set_ptr(to_copy.get_ptr());
	return true;
}