/* 
 * File:   state.cpp
 * Author: Francesco
 *
 * Created on March 31, 2019, 11:34 AM
 */
#include "state_A.h"

/***************************************************************************************
 * Implementation of the functions that don't require state specifications to work .
 * These constructors are called before the specific ones.
 ****************************************************************************************/
state_A::state_A()
{
}

state_A::state_A(const state_A & given_state)
{
	set_state(given_state);
}

state_A::state_A(const state_A & prev_state, const action & executed_action)
{
	set_state(prev_state.compute_succ(executed_action));
}

state_A::state_A(const action_id_list & executed_actions, unsigned short length)
{
	set_executed_actions(executed_actions);
	set_plan_length(length);
	set_heuristic_value();
}

state_A::state_A(const action_id_list & executed_actions, unsigned short length, int heuristic_value)
{
	set_executed_actions(executed_actions);
	set_plan_length(length);
	set_heuristic_value(heuristic_value);
}

const action_id_list & state_A::get_executed_actions()
{
	return m_executed_actions;
}

unsigned short state_A::get_plan_length()
{
	return m_plan_length;
}

int state_A::get_heuristic_value()
{
	return m_heuristic_value;
}

void state_A::set_state(const state_A & given_state)
{
	set_executed_actions(given_state.get_executed_actions());
	set_plan_length(given_state.get_plan_length());
	set_heuristic_value(given_state.get_heuristic_value());
}

void state_A::set_executed_actions(const action_id_list & executed)
{
	m_executed_actions_id = executed;
}

void state::set_plan_length(unsigned short length)
{
	mstate_A_plan_length = length;
}

void state_A::set_heuristic_value(int heuristic_value)
{
	m_heuristic_value = heuristic_value;
}
//@TODO: to Implement
bool state_A::is_goal();

//@TODO: for now commented because is the same in kworld -> accorpate

/*bool state_A::entails(const fluent_set & to_check)
{
	//fluent_set expresses CNF
	fluent_set::const_iterator it_fl;
	for (it_fl = to_check.begin(); it_fl != to_check.end(); it_fl++) {
		if(!entails(*it_fl)){
			return false;
		}
	}
	return true;
}
bool state_A::entails (const fluent_formula & to_check)
{
	//@TODO: Check for the size = 0?
	//fluent_formula expresses DNF
	fluent_formula::const_iterator it_fl;
	for (it_fl = to_check.begin(); it_fl != to_check.end(); it_fl++) {
		if(entails(*it_fl)){
			return true;
		}
	}
	return false;
}*/
bool state_A::entails(const formula_list & to_check)
{
	//formula_list expresses CNF
	formula_list::const_iterator it_fl;
	for (it_fl = to_check.begin(); it_fl != to_check.end(); it_fl++) {
		if (!entails(*it_fl)) {
			return false;
		}
	}
	return true;
}