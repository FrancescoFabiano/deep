#include "planning_graph.h"
//Templatic functions implementations


//Planning Graph Code

template <class T>
void pg_state_level::initialize(const formula_list & goals, T & eState)
{
	build_init_f_map(eState);
	build_init_bf_map(goals, eState);
}

template <class T>
void pg_state_level::build_init_f_map(T & eState)
{
	fluent_set fluents = domain::get_instance().get_fluents();
	for (auto it_fls = fluents.begin(); it_fls != fluents.end(); it_fls++) {
		if (eState.entails(*it_fls)) {
			m_pg_f_map.insert(std::pair<fluent, short>(*it_fls, 0));
		} else {
			m_pg_f_map.insert(std::pair<fluent, short>(*it_fls, -1));
		}
	}
}

template <class T>
void pg_state_level::build_init_bf_map(const formula_list & goals, T & eState)
{
	//The one to set to TRUE
	insert_subformula_bf(domain::get_instance().get_initial_description().get_initial_conditions(), eState);


	insert_subformula_bf(goals, eState);
	
	action_set actions = domain::get_instance().get_actions();
	for (auto it_acs = actions.begin(); it_acs != actions.end(); it_acs++) {

		for (auto it_mapeff = it_acs->get_effects().begin(); it_mapeff != it_acs->get_effects().end(); it_mapeff++) {
			//if (it_mapeff->second.size() > 0) {
			insert_subformula_bf(it_mapeff->second, eState);
			//}
		}

		if (it_acs->get_executability().size() > 0) {
			insert_subformula_bf(it_acs->get_executability(), eState);
		}

		for (auto it_full_obs = it_acs->get_fully_observants().begin(); it_full_obs != it_acs->get_fully_observants().end(); it_full_obs++) {
			//if (it_full_obs->second.size() > 0) {
			insert_subformula_bf(it_full_obs->second, eState);
			//}
		}
		for (auto it_part_obs = it_acs->get_partially_observants().begin(); it_part_obs != it_acs->get_partially_observants().end(); it_part_obs++) {
			//if (it_part_obs->second.size() > 0) {
			insert_subformula_bf(it_part_obs->second, eState);
			//}
		}
	}

}

template <class T>
void pg_state_level::insert_subformula_bf(const formula_list & fl, T & eState)
{
	for (auto it_fl = fl.begin(); it_fl != fl.end(); it_fl++) {
		insert_subformula_bf(*it_fl, eState);
	}
}

template <class T>
void pg_state_level::insert_subformula_bf(const belief_formula & bf, T & eState)
{
	//We set all the subformulas to be TRUE for initially. Maybe it is wrong
	//Maybe we don't need the subformulas at all

	short value;
	if (eState.entails(bf)) {
		value = 0;
	} else {
		value = -1;
	}

	switch ( bf.get_formula_type() ) {

	case BELIEF_FORMULA:
		if (m_pg_bf_map.insert(std::pair < belief_formula, short>(bf, value)).second) {
			insert_subformula_bf(bf.get_bf1(), eState);
		}
		break;

	case PROPOSITIONAL_FORMULA:
		switch ( bf.get_operator() ) {
		case BF_NOT:
			//HERE ALWAYS TRUE BECAUSE IS MONOTONIC NOW
			if (m_pg_bf_map.insert(std::pair < belief_formula, short>(bf, value)).second) {
				insert_subformula_bf(bf.get_bf1(), eState);
			}
			break;
		case BF_OR:
		case BF_AND:
			if (m_pg_bf_map.insert(std::pair < belief_formula, short>(bf, value)).second) {
				insert_subformula_bf(bf.get_bf1(), eState);
				insert_subformula_bf(bf.get_bf2(), eState);
			}
			break;
		case BF_FAIL:
		default:
			std::cerr << "Something went wrong in the generation of the subformulas for the Plannign Graph";
			exit(1);
		}
		break;
	case C_FORMULA:
		if (m_pg_bf_map.insert(std::pair < belief_formula, short>(bf, value)).second) {
			insert_subformula_bf(bf.get_bf1(), eState);
		}
		break;
	case FLUENT_FORMULA:
	case BF_EMPTY:
	{
		break;
	}
	case BF_TYPE_FAIL:
	default:
		std::cerr << "Something went wrong in the generation of the subformulas for the Planning Graph";
		exit(1);
	}
}

template <class T>
planning_graph::planning_graph(T & eState)
{auto goals = domain::get_instance().get_goal_description();
	pg_state_level pg_init;
	pg_init.initialize(goals, eState);
	init(goals, pg_init);
}

template <class T>
planning_graph::planning_graph(const formula_list & goal, T & eState)
{
	pg_state_level pg_init;
	pg_init.initialize(goal, eState);
	init(goal, pg_init);
}