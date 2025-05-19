/*
 * \brief Class that implements planning_graph.h.
 *
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date November 1, 2019
 */

#include "planning_graph.h"
#include "../utilities/printer.h"

pg_action_level::pg_action_level()
{

	set_depth(0);
}

pg_action_level::pg_action_level(const action_set & actions)
{

	set_actions(actions);
	set_depth(0);
}

pg_action_level::pg_action_level(const action_set & actions, unsigned short depth)
{

	set_actions(actions);
	set_depth(depth);
}

void pg_action_level::set_actions(const action_set & actions)
{
	m_actions = actions;
}

void pg_action_level::add_action(const action & act)
{
	m_actions.insert(act);
}

void pg_action_level::set_depth(unsigned short depth)
{
	m_depth = depth;
}

unsigned short pg_action_level::get_depth() const
{
	return m_depth;
}

const action_set & pg_action_level::get_actions() const
{
	return m_actions;
}

bool pg_action_level::operator=(const pg_action_level& to_assign)
{
	set_actions(to_assign.get_actions());
	set_depth(to_assign.get_depth());
	return true;
}


//printing

void pg_action_level::print() const
{
	std::cout << "\nPrinting an Action Level:";
	for (action_set::iterator it_acts = m_actions.begin(); it_acts != m_actions.end(); it_acts++) {

		std::cout << "\nAction " << it_acts->get_name() << std::endl;
	}
}

/*\*********************************************************************************************/

pg_state_level::pg_state_level()
{
	set_depth(0);
}

void pg_state_level::initialize(const formula_list & goals)
{
	build_init_f_map();
	build_init_bf_map(goals);
}

pg_state_level::pg_state_level(const pg_state_level & to_assign)
{
	set_pg_state_level(to_assign);
}

pg_state_level::pg_state_level(const pg_f_map & f_map, const pg_bf_map & bf_map, unsigned short depth)
{
	set_f_map(f_map);
	set_bf_map(bf_map);
	set_depth(depth);
}

void pg_state_level::build_init_f_map()
{
	fluent_set initialy_fluent = domain::get_instance().get_initial_description().get_initially_known_fluents();
	for (auto it_fls_ini = initialy_fluent.begin(); it_fls_ini != initialy_fluent.end(); it_fls_ini++) {
		m_pg_f_map.insert(std::pair<fluent, short>(*it_fls_ini, 0));
	}

	fluent_set fluents = domain::get_instance().get_fluents();
	for (auto it_fls = fluents.begin(); it_fls != fluents.end(); it_fls++) {
		//It only inserts if the pair was not there before
		m_pg_f_map.insert(std::pair<fluent, short>(*it_fls, -1));
	}
}

void pg_state_level::insert_subformula_bf(const formula_list & fl, short value)
{
	for (auto it_fl = fl.begin(); it_fl != fl.end(); it_fl++) {
		insert_subformula_bf(*it_fl, value);
	}
}

void pg_state_level::insert_subformula_bf(const BeliefFormula & bf, short value)
{
	//We set all the subformulas to be TRUE for initially. Maybe it is wrong
	//Maybe we don't need the subformulas at all
	switch ( bf.get_formula_type() ) {

	case BELIEF_FORMULA:
		if (m_pg_bf_map.insert(std::pair < BeliefFormula, short>(bf, value)).second) {
			insert_subformula_bf(bf.get_bf1(), value);
		}
		break;

	case PROPOSITIONAL_FORMULA:
		switch ( bf.get_operator() ) {
		case BF_NOT:
			//HERE ALWAYS TRUE BECAUSE IS MONOTONIC NOW
			if (m_pg_bf_map.insert(std::pair < BeliefFormula, short>(bf, 0)).second) {
				insert_subformula_bf(bf.get_bf1(), value);
			}
			break;
		case BF_OR:
			if (m_pg_bf_map.insert(std::pair < BeliefFormula, short>(bf, value)).second) {
				insert_subformula_bf(bf.get_bf1(), value);
				insert_subformula_bf(bf.get_bf2(), value);
			}
			break;
		case BF_AND:
			if (m_pg_bf_map.insert(std::pair < BeliefFormula, short>(bf, value)).second) {
				insert_subformula_bf(bf.get_bf1(), value);
				insert_subformula_bf(bf.get_bf2(), value);
			}
			break;
		case BF_FAIL:
		default:
			std::cerr << "Something went wrong in the generation of the subformulas for the Plannign Graph";
			exit(1);
		}
		break;
	case C_FORMULA:
		if (m_pg_bf_map.insert(std::pair < BeliefFormula, short>(bf, value)).second) {
			insert_subformula_bf(bf.get_bf1(), value);
		}
		break;
	case FLUENT_FORMULA:
	case BF_EMPTY:
	{
		break;
	}
	case BF_TYPE_FAIL:
	default:
		std::cerr << "Something went wrong in the generation of the subformulas for the Plannign Graph";
		exit(1);
	}
}

void pg_state_level::build_init_bf_map(const formula_list & goals)
{
	//The one to set to TRUE
	insert_subformula_bf(domain::get_instance().get_initial_description().get_initial_conditions(), 0);

	insert_subformula_bf(goals, -1);

	action_set actions = domain::get_instance().get_actions();
	for (auto it_acs = actions.begin(); it_acs != actions.end(); it_acs++) {

		for (auto it_mapeff = it_acs->get_effects().begin(); it_mapeff != it_acs->get_effects().end(); it_mapeff++) {
			//if (it_mapeff->second.size() > 0) {
			insert_subformula_bf(it_mapeff->second, -1);
			//}
		}

		if (it_acs->get_executability().size() > 0) {
			insert_subformula_bf(it_acs->get_executability(), -1);
		}

		for (auto it_full_obs = it_acs->get_fully_observants().begin(); it_full_obs != it_acs->get_fully_observants().end(); it_full_obs++) {
			//if (it_full_obs->second.size() > 0) {
			insert_subformula_bf(it_full_obs->second, -1);
			//}
		}
		for (auto it_part_obs = it_acs->get_partially_observants().begin(); it_part_obs != it_acs->get_partially_observants().end(); it_part_obs++) {
			//if (it_part_obs->second.size() > 0) {
			insert_subformula_bf(it_part_obs->second, -1);
			//}
		}
	}

}

void pg_state_level::set_f_map(const pg_f_map & to_set)
{
	m_pg_f_map = to_set;
}

void pg_state_level::set_bf_map(const pg_bf_map & to_set)
{
	m_pg_bf_map = to_set;
}

void pg_state_level::set_depth(unsigned short to_set)
{
	m_depth = to_set;
}

short pg_state_level::get_fluent_value(const fluent & key) const
{
	if (m_pg_f_map.find(key) != m_pg_f_map.end()) {
		return m_pg_f_map.at(key);
	} else {
		return -1;
	}
}

short pg_state_level::get_bf_value(const BeliefFormula & key) const
{
	if (m_pg_bf_map.find(key) != m_pg_bf_map.end()) {
		return m_pg_bf_map.at(key);
	} else if (key.get_formula_type() == FLUENT_FORMULA) {
		fluent_formula ff_temp = key.get_fluent_formula();
		fluent_set fs_temp;
		short ret_val = -1;
		short tmp_val;
		if (ff_temp.size() == 1) {
			fs_temp = *ff_temp.begin();
			for (auto it_f = fs_temp.begin(); it_f != fs_temp.end(); ++it_f) {
				if (!pg_entailment(*it_f)) {
					return -1;
				} else {
					tmp_val = m_pg_f_map.at(*it_f);
					if (tmp_val > ret_val) {
						ret_val = tmp_val;
					}
				}
			}
		} else {
			std::cerr << "\nThe planning BisGraph does not support non-deterministic action yet.";
			exit(1);
		}
		return ret_val;
	} else {
		std::cerr << "\n\nFound bf formula never declared in the Planning Graph.\n";
		exit(1);
	}
}

const pg_f_map & pg_state_level::get_f_map() const
{
	return m_pg_f_map;
}

const pg_bf_map & pg_state_level::get_bf_map() const
{
	return m_pg_bf_map;
}

unsigned short pg_state_level::get_depth() const
{
	return m_depth;
}

short pg_state_level::get_score_from_depth() const
{
	return m_depth;
}

void pg_state_level::modify_fluent_value(const fluent & key, short value)
{
	if (m_pg_f_map.find(key) != m_pg_f_map.end()) {
		if (m_pg_f_map.at(key) < 0) {
			m_pg_f_map[key] = value;
		}
	} else {
		std::cerr << "\n\nFound fluent never declared in the Planning Graph.\n";
		exit(1);
	}
}

void pg_state_level::modify_bf_value(const BeliefFormula & key, short value)
{
	if (m_pg_bf_map.find(key) != m_pg_bf_map.end()) {
		if (m_pg_bf_map.at(key) < 0) {
			m_pg_bf_map[key] = value;
		}
	} else {
		std::cerr << "\n\nFound bf formula never declared in the Planning Graph.\n";
		exit(1);
	}
}

bool pg_state_level::pg_entailment(const fluent & f) const
{
	return get_fluent_value(f) >= 0;
}

bool pg_state_level::pg_entailment(const BeliefFormula & bf) const
{
	if (bf.get_formula_type() == BF_EMPTY) {
		return true;
	} else {
		return get_bf_value(bf) >= 0;
	}
}

bool pg_state_level::pg_entailment(const formula_list & fl) const
{
	auto it_fl = fl.begin();
	for (; it_fl != fl.end(); it_fl++) {
		if (!pg_entailment(*it_fl)) {
			return false;
		}
	}
	return true;
}

bool pg_state_level::pg_executable(const action & act) const
{
	return pg_entailment(act.get_executability());
}

void pg_state_level::get_base_fluents(const BeliefFormula & bf, fluent_set & bf_base_fluents)
{
	switch ( bf.get_formula_type() ) {

	case FLUENT_FORMULA:
	{
		fluent_formula ff = bf.get_fluent_formula();
		for (auto it_ff = ff.begin(); it_ff != ff.end(); it_ff++) {
			for (auto it_fs = it_ff->begin(); it_fs != it_ff->end(); it_fs++) {
				bf_base_fluents.insert(*it_fs);
			}
		}
		break;
	}

	case BELIEF_FORMULA:
	case C_FORMULA:
		get_base_fluents(bf.get_bf1(), bf_base_fluents);
		break;

	case PROPOSITIONAL_FORMULA:
		switch ( bf.get_operator() ) {
		case BF_NOT:
			get_base_fluents(bf.get_bf1(), bf_base_fluents);
			break;
		case BF_OR:
		case BF_AND:
			get_base_fluents(bf.get_bf1(), bf_base_fluents);
			get_base_fluents(bf.get_bf2(), bf_base_fluents);
			break;
		case BF_FAIL:
		default:
			std::cerr << "\nSomething went wrong in the generation of the subformulas while searching for fluents in belief formulas";
			exit(1);
		}
		break;
	case BF_EMPTY:
	{
		break;
	}
	case BF_TYPE_FAIL:
	default:
		std::cerr << "\nSomething went wrong in the generation of the subformulas while searching for fluents in belief formulas";
		exit(1);
	}
}

bool pg_state_level::compute_succ(const action & act, const pg_state_level & predecessor, FormulaeSet & false_bf)
{
	switch ( act.get_type() ) {
	case ONTIC:
		return exec_ontic(act, predecessor, false_bf);
	case SENSING:
	case ANNOUNCEMENT:
		return exec_epistemic(act, predecessor, false_bf);
	default:
		std::cerr << "\nAction Type not properly declared (PG building)";
		exit(1);
	}
}

bool pg_state_level::exec_ontic(const action & act, const pg_state_level & predecessor, FormulaeSet & false_bf)
{
	ObservabilitiesMap fully_observant_map = act.get_fully_observants();
	AgentSet fully_obs;
	EffectsMap effects_map = act.get_effects();
	fluent_set verified_fluents;
	fluent_set bf_base_fluents;
	bool modified_pg = false;
	fluent_formula ff_temp;
	fluent_set fs_temp;
	fluent f_tmp;


	for (auto it_fl = effects_map.begin(); it_fl != effects_map.end(); it_fl++) {
		if (predecessor.pg_entailment(it_fl->second)) {
			ff_temp = it_fl->first;
			if (ff_temp.size() == 1) {
				fs_temp = *ff_temp.begin();
				for (auto it_eff = fs_temp.begin(); it_eff != fs_temp.end(); it_eff++) {
					f_tmp = *it_eff;
					if (!pg_entailment(f_tmp)) {
						modified_pg = true;
						modify_fluent_value(f_tmp, get_score_from_depth());
					}
					verified_fluents.insert(f_tmp);
				}
			} else {
				std::cerr << "\nThe planning BisGraph does not support non-deterministic ontic action yet.";
				exit(1);
			}

		}
	}

	for (auto it_fully = fully_observant_map.begin(); it_fully != fully_observant_map.end(); it_fully++) {
		if (predecessor.pg_entailment(it_fully->second)) {
			fully_obs.insert(it_fully->first);
		}
	}


	FormulaeSet tmp_fl = false_bf;


	for (auto it_false_bf = tmp_fl.begin(); it_false_bf != tmp_fl.end(); it_false_bf++) {
		bool tmp_modified = false;
		bf_base_fluents.clear();
		get_base_fluents(*it_false_bf, bf_base_fluents);
		if (!helper::fluentset_empty_intersection(verified_fluents, bf_base_fluents)) {

			apply_ontic_effects(*it_false_bf, false_bf, fully_obs, tmp_modified);
			if (!modified_pg && tmp_modified) {
				modified_pg = true;
			}
		}
	}


	return modified_pg;

}

bool pg_state_level::exec_epistemic(const action & act, const pg_state_level & predecessor, FormulaeSet & false_bf)
{
	ObservabilitiesMap partially_observant_map = act.get_partially_observants();
	ObservabilitiesMap fully_observant_map = act.get_fully_observants();
	AgentSet fully_obs;
	AgentSet partially_obs;
	EffectsMap effects_map = act.get_effects();
	fluent_set bf_base_fluents;
	fluent_formula ff_temp;
	fluent_set fs_temp;
	fluent f_tmp;


	//observability sets (an agent can be in both)
	for (auto it_fully = fully_observant_map.begin(); it_fully != fully_observant_map.end(); it_fully++) {
		if (predecessor.pg_entailment(it_fully->second)) {
			fully_obs.insert(it_fully->first);
		}
	}

	for (auto it_partially = partially_observant_map.begin(); it_partially != partially_observant_map.end(); it_partially++) {
		if (predecessor.pg_entailment(it_partially->second)) {
			partially_obs.insert(it_partially->first);
		}
	}

	bool modified_pg = false;
	FormulaeSet tmp_fl = false_bf;
	for (auto it_false_bf = tmp_fl.begin(); it_false_bf != tmp_fl.end(); it_false_bf++) {
		bool tmp_modified = false;
		bf_base_fluents.clear();
		get_base_fluents(*it_false_bf, bf_base_fluents);

		for (auto it_fl = effects_map.begin(); it_fl != effects_map.end(); it_fl++) {
			if (predecessor.pg_entailment(it_fl->second)) {
				ff_temp = it_fl->first;
				if (ff_temp.size() == 1) {
					fs_temp = *ff_temp.begin();
					for (auto it_eff = fs_temp.begin(); it_eff != fs_temp.end(); it_eff++) {
						f_tmp = *it_eff;
						//Single fluent each time! The negation for partially must contain also the non-negated version
						if (bf_base_fluents.find(f_tmp) != bf_base_fluents.end()) {
							apply_epistemic_effects(f_tmp, *it_false_bf, false_bf, fully_obs, partially_obs, tmp_modified, 0);
							if (!modified_pg && tmp_modified) {
								modified_pg = true;
							}
						}
					}
				}
			}
		}
	}
	return modified_pg;
}

bool pg_state_level::apply_ontic_effects(const BeliefFormula & bf, FormulaeSet & fl, const AgentSet & fully, bool & modified_pg)
{
	if (pg_entailment(bf)) {
		return true;
	}

	switch ( bf.get_formula_type() ) {

	case FLUENT_FORMULA:
	{
		//We are "generous".
		//If a fluent has been set to true and it is in this belief formula then the fully observants agent will know it even if it was not an action's effect
		//
		//In preprocessing we should separate the fluent formula in Belief formula with unique fluents -- B(a, f && g) -> B(a, f) && B(a, g)
		bool ret;
		fluent_formula ff = bf.get_fluent_formula();
		for (auto it_ff = ff.begin(); it_ff != ff.end(); it_ff++) {
			ret = true;
			for (auto it_fs = it_ff->begin(); it_fs != it_ff->end();) {
				if (!pg_entailment(*it_fs)) {
					ret = false;
					it_fs = it_ff->end();
				} else {
					it_fs++;
				}
			}
			if (ret) {
				return true;
			}
		}
		break;
	}

	case BELIEF_FORMULA:
	{
		if (fully.find(bf.get_agent()) != fully.end()) {
			//We are "generous". Maybe it is necessary to check if the chain is of fully every time.
			if (apply_ontic_effects(bf.get_bf1(), fl, fully, modified_pg)) {
				//Alreaady checked at the beginning
				//if (!get_bf_value(bf))
				//{
				modified_pg = true;
				//}
				modify_bf_value(bf, get_score_from_depth());
				fl.erase(bf);
				return true;
			}
		}
		break;
	}

	case PROPOSITIONAL_FORMULA:
		switch ( bf.get_operator() ) {
			//Without attitudes the beliefs are monotonic, so it should be true in the initial state and cannot ever
		case BF_NOT:
		{
			return pg_entailment(bf);
			break;
		}
		case BF_OR:
		{
			if (apply_ontic_effects(bf.get_bf1(), fl, fully, modified_pg) || apply_ontic_effects(bf.get_bf2(), fl, fully, modified_pg)) {
				modified_pg = true;
				modify_bf_value(bf, get_score_from_depth());
				fl.erase(bf);
				return true;
			}
			break;
		}
		case BF_AND:
		{
			if (apply_ontic_effects(bf.get_bf1(), fl, fully, modified_pg) && apply_ontic_effects(bf.get_bf2(), fl, fully, modified_pg)) {
				modified_pg = true;
				modify_bf_value(bf, get_score_from_depth());
				fl.erase(bf);
				return true;
			}
			break;
		}
		case BF_FAIL:
		default:
			std::cerr << "\nSomething went wrong in the generation of the subformulas while searching for fluents in belief formulas";
			exit(1);
		}
		break;

	case C_FORMULA:
	{
		for (auto it_ags = bf.get_group_agents().begin(); it_ags != bf.get_group_agents().end(); it_ags++) {
			if (fully.find(*it_ags) == fully.end()) {
				return false;
			}
		}


		//We are "generous". Maybe it is necessary to check if the chain is of fully every time.
		if (apply_ontic_effects(bf.get_bf1(), fl, fully, modified_pg)) {
			modified_pg = true;
			modify_bf_value(bf, get_score_from_depth());
			fl.erase(bf);
			return true;
		}
		break;
	}

	case BF_EMPTY:
	{
		return true;
		break;
	}
	case BF_TYPE_FAIL:
	default:
		std::cerr << "\nSomething went wrong in the generation of the subformulas while searching for fluents in belief formulas";
		exit(1);
	}

	return false;
}

bool pg_state_level::apply_epistemic_effects(fluent effect, const BeliefFormula & bf, FormulaeSet & fl, const AgentSet & fully, const AgentSet & partially, bool & modified_pg, unsigned short vis_cond)
{
	if (pg_entailment(bf)) {
		return true;
	}

	unsigned short temp_vis_cond = vis_cond;
	bool tmp_ret1 = false;
	bool tmp_ret2 = false;


	switch ( bf.get_formula_type() ) {

	case FLUENT_FORMULA:
	{
		//CHECK IF COME FROM A CHAIN WITH PARTIALLY THAT THE BELIEF FORMULA HAS THE RIGTH FORM? Or just generalize
		bool ret;
		fluent_formula ff = bf.get_fluent_formula();
		for (auto it_ff = ff.begin(); it_ff != ff.end(); it_ff++) {
			ret = true;
			for (auto it_fs = it_ff->begin(); it_fs != it_ff->end();) {

				if (!pg_entailment(*it_fs)) {
					if (*it_fs == effect && (vis_cond < 2)) {
						it_fs++;
					} else if (*it_fs == helper::negate_fluent(effect) && (vis_cond == 1)) {
						it_fs++;
					} else {
						ret = false;
						it_fs = it_ff->end();
					}
				} else {
					it_fs++;
				}
			}
			if (ret) {
				return true;
			}
		}
		break;
	}

	case BELIEF_FORMULA:
	{

		//Should we execute both the version and return the "or" of the results
		if (fully.find(bf.get_agent()) != fully.end()) {
			if (vis_cond == 2) {
				temp_vis_cond = 1;
			}
			if (apply_epistemic_effects(effect, bf.get_bf1(), fl, fully, partially, modified_pg, temp_vis_cond)) {
				modified_pg = true;
				modify_bf_value(bf, get_score_from_depth());
				fl.erase(bf);
				tmp_ret1 = true;
			}
		}
		if (partially.find(bf.get_agent()) != partially.end()) {
			if (apply_epistemic_effects(effect, bf.get_bf1(), fl, fully, partially, modified_pg, 2)) {
				modified_pg = true;
				modify_bf_value(bf, get_score_from_depth());
				fl.erase(bf);
				tmp_ret1 = false;
			}
		}
		return tmp_ret1 || tmp_ret2;
		break;
	}

	case PROPOSITIONAL_FORMULA:
		switch ( bf.get_operator() ) {
			//Without attitudes the beliefs are monotonic, so it should be true in the initial state and cannot ever
		case BF_NOT:
			return pg_entailment(bf);
			break;
		case BF_OR:
		{
			if (apply_epistemic_effects(effect, bf.get_bf1(), fl, fully, partially, modified_pg, vis_cond)
				|| apply_epistemic_effects(effect, bf.get_bf2(), fl, fully, partially, modified_pg, vis_cond)) {
				modified_pg = true;
				modify_bf_value(bf, get_score_from_depth());
				fl.erase(bf);
				return true;
			}
			break;
		}
		case BF_AND:
		{
			if (apply_epistemic_effects(effect, bf.get_bf1(), fl, fully, partially, modified_pg, vis_cond)
				&& apply_epistemic_effects(effect, bf.get_bf2(), fl, fully, partially, modified_pg, vis_cond)) {
				modified_pg = true;
				modify_bf_value(bf, get_score_from_depth());
				fl.erase(bf);
				return true;
			}
			break;
		}
		case BF_FAIL:
		default:
			std::cerr << "\nSomething went wrong in the generation of the subformulas while searching for fluents in belief formulas";
			exit(1);
		}
		break;

	case C_FORMULA:
	{
		bool only_fully = true;
		bool one_partial = false;
		for (auto it_ags = bf.get_group_agents().begin(); it_ags != bf.get_group_agents().end(); it_ags++) {
			if (fully.find(*it_ags) == fully.end()) {
				if (partially.find(*it_ags) == partially.end()) {
					return false;
				} else {
					only_fully = false;
					one_partial = true;
				}
			} else if (!one_partial) {
				one_partial = partially.find(*it_ags) != partially.end();
			}
		}

		if (only_fully) {
			if (vis_cond == 2) {
				temp_vis_cond = 1;
			}
			if (apply_epistemic_effects(effect, bf.get_bf1(), fl, fully, partially, modified_pg, temp_vis_cond)) {
				modified_pg = true;
				modify_bf_value(bf, get_score_from_depth());
				fl.erase(bf);
				tmp_ret1 = true;
			}
		}

		if (one_partial) {
			//We are "generous". Maybe it is necessary to check if the chain is of fully every time.
			if (apply_epistemic_effects(effect, bf.get_bf1(), fl, fully, partially, modified_pg, 2)) {
				modified_pg = true;
				modify_bf_value(bf, get_score_from_depth());
				fl.erase(bf);
				tmp_ret2 = true;
			}
		}

		return tmp_ret1 || tmp_ret2;
		break;
	}

	case BF_EMPTY:
	{
		return true;
		break;
	}
	case BF_TYPE_FAIL:
	default:
		std::cerr << "\nSomething went wrong in the generation of the subformulas while searching for fluents in belief formulas";
		exit(1);
	}
	return false;

}

bool pg_state_level::operator=(const pg_state_level & to_assign)
{
	set_pg_state_level(to_assign);
	return true;
}

void pg_state_level::set_pg_state_level(const pg_state_level & to_assign)
{
	set_f_map(to_assign.get_f_map());
	set_bf_map(to_assign.get_bf_map());
	set_depth(to_assign.get_depth());
}



/*\********************Transition function based on update models**********************************/
//Given: a formula, custom update model, action definition and the map of belief formula (previous level) determines whether the given bf is true or not by exploring the update model BisGraph




/*\******************************************************************************************************************/

/*\*****START PLANNING GRAPH TIME MEASURE*******
#include <chrono>
std::chrono::duration<double> t1, t2, t3, t4;

*\******END PLANNING GRAPH TIME MEASURE********/

planning_graph::planning_graph()
{
	auto goals = domain::get_instance().get_goal_description();
	pg_state_level pg_init;
	pg_init.initialize(goals);
	init(goals, pg_init);
}

planning_graph::planning_graph(const planning_graph & pg)
{
	set_pg(pg);
}

planning_graph::planning_graph(const formula_list & goal)
{
	pg_state_level pg_init;
	pg_init.initialize(goal);
	init(goal, pg_init);
}

void planning_graph::init(const formula_list & goal, const pg_state_level & pg_init)
{
	/*\*****START PLANNING GRAPH TIME MEASURE*******
	auto start_pg_build = std::chrono::system_clock::now();
	std::cout << "start" << std::endl;
	*\******END PLANNING GRAPH TIME MEASURE********/


	set_goal(goal);
	m_state_levels.push_back(pg_init);
	auto init_bf_score = pg_init.get_bf_map();
	for (auto it_pgbf = init_bf_score.begin(); it_pgbf != init_bf_score.end(); ++it_pgbf) {
		if (it_pgbf->second < 0) {
			m_belief_formula_false.insert(it_pgbf->first);
		}
	}

	m_never_executed = domain::get_instance().get_actions();
	set_length(0);
	set_sum(0);

	/*\*****START PLANNING GRAPH TIME MEASURE*******
	auto start_pg_goal_ini = std::chrono::system_clock::now();
	*\******END PLANNING GRAPH TIME MEASURE********/


	bool not_goal = false;
	for (auto it_fl = m_goal.begin(); it_fl != m_goal.end();) {
		if (m_state_levels.back().pg_entailment(*it_fl)) {
			it_fl = m_goal.erase(it_fl);
		} else {
			not_goal = true;
			it_fl++;
		}
	}

	/*\*****START PLANNING GRAPH TIME MEASURE*******
	auto end_pg_goal_ini = std::chrono::system_clock::now();
	std::chrono::duration<double> pg_goal_ini_time = end_pg_goal_ini - start_pg_goal_ini;
	t1 = std::chrono::milliseconds::zero();
	t2 = std::chrono::milliseconds::zero();
	t3 = std::chrono::milliseconds::zero();
	t4 = std::chrono::milliseconds::zero();
	*\******END PLANNING GRAPH TIME MEASURE********/

	if (not_goal) {
		//	if (!is_single) {
		pg_build();
		//	} else {
		//		pg_build_initially(goal);
		//	}
	} else {

		set_satisfiable(true);
		std::cerr << "\nBUILDING: The initial state is goal\n";
		exit(1);
	}

	/*\*****START PLANNING GRAPH TIME MEASURE*******
	auto end_pg_build = std::chrono::system_clock::now();
	std::chrono::duration<double> pg_build_time = end_pg_build - start_pg_build;
	std::cout << "\n\nGenerated Planning Graph of length " << get_length() << " in " << pg_build_time.count() << " seconds of which:";
	std::cout << "\nFirst goal check:      " << pg_goal_ini_time.count();
	std::cout << "\nAction Level creation: " << t1.count();
	std::cout << "\n\nState Level Creation:  " << t2.count() << " of which:";
	std::cout << "\nActions Execution:     " << t4.count();
	std::cout << "\n\nGoals Check:           " << t3.count() << std::endl;
	*\******END PLANNING GRAPH TIME MEASURE********/


}

void planning_graph::set_satisfiable(bool sat)
{

	m_satisfiable = sat;
}

bool planning_graph::is_satisfiable() const
{

	return m_satisfiable;
}

void planning_graph::pg_build()
{
	pg_state_level s_level_curr;
	s_level_curr = m_state_levels.back();
	pg_action_level a_level_curr;
	a_level_curr.set_depth(get_length());

	/*\*****START PLANNING GRAPH TIME MEASURE*******
	auto start = std::chrono::system_clock::now();
	*\******END PLANNING GRAPH TIME MEASURE********/

	if (m_action_levels.size() > 0) {
		a_level_curr = m_action_levels.back();
	}
	action_set::iterator it_actset;
	for (it_actset = m_never_executed.begin(); it_actset != m_never_executed.end();) {
		if (s_level_curr.pg_executable(*it_actset)) {
			a_level_curr.add_action(*it_actset);
			it_actset = m_never_executed.erase(it_actset);
		} else {
			it_actset++;
		}
	}
	/*\*****START PLANNING GRAPH TIME MEASURE*******
	auto end = std::chrono::system_clock::now();
	t1 += end - start;
	*\******END PLANNING GRAPH TIME MEASURE********/


	add_action_level(a_level_curr);
	set_length(get_length() + 1);

	//The no-op is done with the copy
	pg_state_level s_level_next;
	s_level_next = s_level_curr;
	s_level_next.set_depth(get_length());
	bool new_state_insertion = false;

	/*\*****START PLANNING GRAPH TIME MEASURE*******
	start = std::chrono::system_clock::now();
	*\******END PLANNING GRAPH TIME MEASURE********/

	/*\*****START PLANNING GRAPH TIME MEASURE*******
	std::cerr << "\n\nPlaning Graph Length: " << get_length();
	if (get_length() > 0) {
		std::cerr << "\nAction Level creation: " << t1.count();
		std::cerr << "\nState Level Creation:  " << t2.count() << " of which:";
		std::cerr << "\nActions execution:     " << t4.count();
	}
	*\******END PLANNING GRAPH TIME MEASURE********/


	for (it_actset = a_level_curr.get_actions().begin(); it_actset != a_level_curr.get_actions().end(); it_actset++) {

		/*\*****START PLANNING GRAPH TIME MEASURE*******
		auto startN = std::chrono::system_clock::now();
		*\******END PLANNING GRAPH TIME MEASURE********/

		if (s_level_next.compute_succ(*it_actset, s_level_curr, m_belief_formula_false)) {
			new_state_insertion = true;
		}
		/*\*****START PLANNING GRAPH TIME MEASURE*******
		auto endN = std::chrono::system_clock::now();
		t4 += endN - startN;
		*\******END PLANNING GRAPH TIME MEASURE********/
	}

	/*\*****START PLANNING GRAPH TIME MEASURE*******
	end = std::chrono::system_clock::now();
	t2 += end - start;
	*\******END PLANNING GRAPH TIME MEASURE********/


	add_state_level(s_level_next);

	bool not_goal = false;

	/*\*****START PLANNING GRAPH TIME MEASURE*******
	start = std::chrono::system_clock::now();
	*\******END PLANNING GRAPH TIME MEASURE********/


	//Remove each sub goal already satisfied: it will always be and we just need to check it once
	for (auto it_fl = m_goal.begin(); it_fl != m_goal.end();) {

		if (s_level_next.pg_entailment(*it_fl)) {
			it_fl = m_goal.erase(it_fl);
			m_pg_sum += get_length();

		} else {
			it_fl++;
			not_goal = true;
		}
	}
	/*\*****START PLANNING GRAPH TIME MEASURE*******
	end = std::chrono::system_clock::now();
	t3 += end - start;
	*\******END PLANNING GRAPH TIME MEASURE********/
	if (!not_goal) {
		set_satisfiable(true);
		return;
	} else if (!new_state_insertion) {
		set_satisfiable(false);
		/*\*****START PLANNING GRAPH TIME MEASURE*******
		if (get_length() > 5) {
			print();
		}
		*\******END PLANNING GRAPH TIME MEASURE********/
		return;
	} else {
		pg_build();
	}
}

void planning_graph::add_state_level(const pg_state_level & s_level)
{
	m_state_levels.push_back(s_level);
}

void planning_graph::add_action_level(const pg_action_level & a_level)
{

	m_action_levels.push_back(a_level);
}

void planning_graph::set_length(unsigned short length)
{

	m_pg_length = length;
}

void planning_graph::set_sum(unsigned short sum)
{

	m_pg_sum = sum;
}

unsigned short planning_graph::get_length() const
{
	//if (is_satisfiable()) {

	return m_pg_length;
	//}
	//std::cerr << "\nThe planning BisGraph could not find any solution. Check for the satisfiability before calling \"get_length\"\n";
	//exit(1);
}//construct planning BisGraph and return the level that satisfied the goal.

unsigned short planning_graph::get_sum() const
{
	//if (is_satisfiable()) {

	return m_pg_sum;
	//}
	//std::cerr << "\nThe planning BisGraph could not find any solution. Check for the satisfiability before calling \"get_sum\"\n";
	//exit(1);
}//

void planning_graph::set_goal(const formula_list & goal)
{
	m_goal = goal;
}

const std::vector< pg_state_level > & planning_graph::get_state_levels() const
{
	return m_state_levels;
}

const std::vector< pg_action_level > & planning_graph::get_action_levels() const
{
	return m_action_levels;
}

const pg_f_map & planning_graph::get_f_scores() const
{
	return m_state_levels.back().get_f_map();
}

const pg_bf_map & planning_graph::get_bf_scores() const
{
	return m_state_levels.back().get_bf_map();
}

const formula_list & planning_graph::get_goal() const
{
	return m_goal;
}

const action_set & planning_graph::get_never_executed() const
{
	return m_never_executed;
}

const FormulaeSet & planning_graph::get_belief_formula_false() const
{
	return m_belief_formula_false;
}

bool planning_graph::operator=(const planning_graph & to_assign)
{
	set_pg(to_assign);
	return true;
}

void planning_graph::set_pg(const planning_graph & to_assign)
{
	m_state_levels = to_assign.get_state_levels();
	m_action_levels = to_assign.get_action_levels();
	m_pg_length = to_assign.get_length();
	m_pg_sum = to_assign.get_sum();
	m_satisfiable = to_assign.is_satisfiable();
	m_goal = to_assign.get_goal();
	//	m_bfs_score = to_assign.get_bfs_score();
	m_never_executed = to_assign.get_never_executed();
	m_belief_formula_false = to_assign.get_belief_formula_false();
}

void planning_graph::print() const
{

	std::cout << "\n\n**********ePLANNING-GRAPH PRINT**********\n";
	unsigned short count = 0;
	for (auto it_stlv = m_state_levels.begin(); it_stlv != m_state_levels.end(); ++it_stlv) {
		auto fluents_score = it_stlv->get_f_map();
		auto bf_score = it_stlv->get_bf_map();

		std::cout << "\n\t*******State Level " << count << "*******\n";
		std::cout << "\n\t\t****Fluents****\n\n";
		for (auto it_pgf = fluents_score.begin(); it_pgf != fluents_score.end(); ++it_pgf) {
			std::cout << "\t\t\t" << domain::get_instance().get_grounder().deground_fluent(it_pgf->first) << " -> " << it_pgf->second << std::endl;
		}
		std::cout << "\n\t\t****Belief Formulae****\n\n";
		for (auto it_pgbf = bf_score.begin(); it_pgbf != bf_score.end(); ++it_pgbf) {
			std::cout << "\t\t\t";
			it_pgbf->first.print();
			std::cout << " -> " << it_pgbf->second << std::endl;
		}
		std::cout << "\n\t*******End State Level " << count << "*******\n";


		if (count < m_action_levels.size()) {
			std::cout << "\n\t*******Action Level " << count << "*******\n";
			auto act_set = m_action_levels.at(count).get_actions();
			for (auto it_acts = act_set.begin(); it_acts != act_set.end(); it_acts++) {

				std::cout << "\n\t\t" << it_acts->get_name() << std::endl;
			}
			std::cout << "\n\t*******End Action Level " << count++ << "*******\n";
		}

	}
	std::cout << "\n*********END ePLANNING-GRAPH PRINT**********\n";
}
