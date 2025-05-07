/**
 * \brief Implementation of \ref state_T.h
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date April 11, 2019
 */
#include <fstream>
#include <stdlib.h>

#include "state_T.h"
#include "../domain/domain.h"

template <class T>
state<T>::state()
{
	//T m_representation;
}

/*template <class T>
state<T>::state(const state<T> & given_state)
{
	(*this) = given_state;
}*/

template <class T>
state<T>::state(const state<T> & prev_state, const action & executed_action)
{
	(*this) = prev_state.compute_succ(executed_action);
}

/*template <class T>
state<T>::state(const action_id_list & executed_actions, unsigned short length)
{
	set_executed_actions(executed_actions);
	set_plan_length(length);
	set_heuristic_value();
}

template <class T>
state<T>::state(const action_id_list & executed_actions, unsigned short length, int heuristic_value)
{
	set_executed_actions(executed_actions);
	set_plan_length(length);
	set_heuristic_value(heuristic_value);
}*/

template <class T>
const action_id_list & state<T>::get_executed_actions() const
{
	return m_executed_actions_id;
}

template <class T>
unsigned short state<T>::get_plan_length() const
{
	return m_plan_length;
}

template <class T>
short state<T>::get_heuristic_value() const
{
	return m_heuristic_value;
}

template <class T>
const T & state<T>::get_representation() const
{
	return m_representation;
}


template <class T>
bool state<T>::operator=(const state<T> & given_state)
{
	/**\warning each T must implement the operator =*/
	set_representation(given_state.get_representation());
	set_executed_actions(given_state.get_executed_actions());
	set_plan_length(given_state.get_plan_length());
	set_heuristic_value(given_state.get_heuristic_value());
	return true;
}

template <class T>
bool state<T>::operator<(const state<T> & to_compare) const
{
	/**\warning each T must implement the operator <*/
	return m_representation < to_compare.get_representation();

}

template <class T>
void state<T>::set_executed_actions(const action_id_list & executed)
{
	m_executed_actions_id = executed;
}

template <class T>
void state<T>::add_executed_action(const action & executed)
{
	m_executed_actions_id.push_back(executed.get_id());
}

template <class T>
void state<T>::set_plan_length(unsigned short length)
{
	m_plan_length = length;
}

template <class T>
void state<T>::set_heuristic_value(short heuristic_value)
{
	m_heuristic_value = heuristic_value;
}

template <class T>
void state<T>::set_representation(const T & to_set)
{
	m_representation = to_set;
}

template <class T>
bool state<T>::entails(const fluent & to_check) const
{
	return m_representation.entails(to_check);
}

template <class T>
bool state<T>::entails(const fluent_set & to_check) const
{
	return m_representation.entails(to_check);
}

template <class T>
bool state<T>::entails(const fluent_formula & to_check) const
{
	return m_representation.entails(to_check);
}

template <class T>
bool state<T>::entails(const belief_formula & to_check) const
{
	return m_representation.entails(to_check);
}

template <class T>
bool state<T>::entails(const formula_list & to_check) const
{
	//formula_list expresses CNF formula
	formula_list::const_iterator it_fl;
	for (it_fl = to_check.begin(); it_fl != to_check.end(); it_fl++) {

		if (!m_representation.entails(*it_fl)) {
			return false;
		}
	}
	return true;
}

template <class T>
void state<T>::build_initial()
{
	//To implement constructor
	//representation = T(init);
	m_representation.build_initial();
	set_plan_length(0);
	//set_heuristic_value(get_representation().compute_heuristic_value());
}

template <class T>
fluent_set state<T>::compute_succ2(const action & act) const
{
	state<T> ret;
	//if (is_executable(act)) {
	ret.set_representation(get_representation().compute_succ(act));
	ret.set_executed_actions(get_executed_actions());
	ret.add_executed_action(act);

	pworld_ptr repr = ret.get_representation();
	fluent_set repr_set = repr.get_fluent_set();
	ret.set_plan_length(get_plan_length() + 1);


	return repr_set;
}

template <class T>
state<T> state<T>::compute_succ(const action & act) const
{
	/**\todo Myabe is better if used in \ref planner or \ref domain (myabe a bool as **param[out]**.*/
	state<T> ret;
	//if (is_executable(act)) {
	ret.set_representation(get_representation().compute_succ(act));
	ret.set_executed_actions(get_executed_actions());
	ret.add_executed_action(act);
	ret.set_plan_length(get_plan_length() + 1);
	//set_heuristic_value(get_representation().compute_heuristic_value());
	return ret;
	/*} else {
		ret = (*this);
	}
	return ret;*/
}

template <class T>
void state<T>::calc_min_bisimilar()
{
	//std::cerr << "\nDEBUG: ENTRATO IN STATE<T> BISIMULATION\n";
	m_representation.calc_min_bisimilar();
	//		std::cerr << "\nDEBUG: DONE BIS\n" << std::flush;

	//std::cerr << "\nDEBUG: FINE STATE<T> BISIMULATION\n";

}

template <class T>
bool state<T>::is_executable(const action & act) const
{
	return entails(act.get_executability());
}

template <class T>
bool state<T>::is_goal() const
{
	return entails(domain::get_instance().get_goal_description());
}

template <class T>
void state<T>::print(parallel_type ptype, heuristics h) const
{
	std::string parallel_handler = "";
	if(ptype==P_CHILD){
		std::string heuristic_names[5] = {"NO_H","L_PG","S_PG","C_PG","SUBGOALS"};
		parallel_handler = "[" + heuristic_names[(int)h] + "] " ;
	}
	std::cout << "\n";
	if (domain::get_instance().get_debug()) {
		m_representation.print();
	}
	//ret.set_representation(get_representation().compute_succ(act));
	std::cout << parallel_handler+"Plan Length: " << get_plan_length();
	std::cout << "\n\n"+parallel_handler+"Executed actions: ";
	printer::get_instance().print_list(get_executed_actions());
	//std::cout << "\nHeuristic Value Length: " << get_heuristic_value();
}

template <class T>
void state<T>::print_graphviz(std::string postfix) const
{
	std::cout << "\nGraphviz-Printing of ";
	printer::get_instance().print_list(get_executed_actions());
	std::string exec_act_names;
	action_id_list::const_iterator it_act;
	if (get_executed_actions().size() == 0) {
		exec_act_names = "ini";
	} else {
		for (it_act = get_executed_actions().begin(); it_act != get_executed_actions().end(); it_act++) {
			exec_act_names += domain::get_instance().get_grounder().deground_action(*it_act);
			exec_act_names += "_";
		}
		exec_act_names.pop_back();
	}

	std::ofstream graphviz;
	std::string folder = "out/state/";
	folder += domain::get_instance().get_name();
	switch ( domain::get_instance().get_stype() ) {
	case POSSIBILITIES:
		folder += "_poss";
		break;
	default:
		folder += "_unknown";
		break;
	}
	system(("mkdir -p " + folder).c_str());
	graphviz.open(folder + "/" + exec_act_names + postfix + ".dot");
	graphviz << "digraph K_structure{\n";
	graphviz << "	rankdir=BT;\n";
	graphviz << "	size=\"8,5\"\n";
	m_representation.print_graphviz(graphviz);
	graphviz << "}";
	graphviz.close();
	std::cout << postfix << " done.";

}


template <class T>
const std::string state<T>::print_graphviz_ML_dataset(std::string postfix) const
{
	//std::cout << "\nGraphviz-Printing of ";
	//printer::get_instance().print_list(get_executed_actions());
	std::string exec_act_names;
	action_id_list::const_iterator it_act;
	if (get_executed_actions().size() == 0) {
		exec_act_names = "ini";
	} else {
		for (it_act = get_executed_actions().begin(); it_act != get_executed_actions().end(); it_act++) {
			exec_act_names += domain::get_instance().get_grounder().deground_action(*it_act);
			exec_act_names += "_";
		}
		exec_act_names.pop_back();
	}

	std::ofstream graphviz;
	std::string folder = "out/state/";
	folder += domain::get_instance().get_name();
	switch ( domain::get_instance().get_stype() ) {
	case POSSIBILITIES:
		folder += "_poss";
		break;
	default:
		folder += "_unknown";
		break;
	}
	system(("mkdir -p " + folder).c_str());
	graphviz.open(folder + "/" + exec_act_names + postfix + ".dot");
	/*graphviz << "digraph K_structure{\n";
	graphviz << "	rankdir=BT;\n";
	graphviz << "	size=\"8,5\"\n";
	m_representation.print_graphviz_explicit(graphviz);
	graphviz << "}"; */

	m_representation.print_ML_dataset(graphviz);

	graphviz.close();
	//std::cout << postfix << " done.";

	return folder + "/" + exec_act_names + postfix + ".dot";

}



//DEBUG
//
//template <class T>
//T state<T>::debug_get_representation()
//{
//	return m_representation;
//}
//
//template <class T>
//void state<T>::min_with_print(state<T> tmp)
//{
//	std::string tmp_name, state_name;
//	if (tmp.get_representation().get_edges() < get_representation().get_edges()) {
//		tmp_name = "_arg_1";
//		state_name = "_arg_2";
//	} else {
//		tmp_name = "_arg_2";
//		state_name = "_arg_1";
//	}
//	tmp.print_graphviz(tmp_name);
//	print_graphviz(state_name);
//
//	T tmp_repr = tmp.debug_get_representation();
//	tmp_repr.debug_print(get_representation());
//	//std::cerr << "\nDEBUG:HERE\n";
//	tmp.set_representation(tmp_repr);
//
//	tmp.print_graphviz("_eq_res");
//}