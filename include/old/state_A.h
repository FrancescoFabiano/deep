/* 
 * File:   state.h
 * Author: Francesco
 *
 * Created on March 31, 2019, 10:57 AM
 */
#pragma once

#include "../actions/action.h"
#include "../domain/initially.h"
//#include "belief_formulae.h"

/***************************************************************************************
The class that encode the state of the planner; this is the #interface and will be used 
as black box from @planner: its implementation will depend on the initial choices:
 * .Kripke
 * .Possibilities
 * .OBDDs

This class should be used to check entailment and to produce successors
 ****************************************************************************************/

class state_A
{
public:

    //Implement in each class with a kstate instead of a state
    state_A();
    state_A(const state_A &);
    state_A(const state_A &, const action &);
    state_A(const action_id_list &, unsigned short);
    state_A(const action_id_list &, unsigned short, int);

    const action_id_list & get_executed_actions();
    unsigned short get_plan_length();
    int get_heuristic_value();

    void set_state(state_A);

    void set_executed_actions(const action_id_list &);
    void set_plan_length(unsigned short);
    void set_heuristic_value(int);

    //virtual state compute_succ (const action &) = 0;

    virtual bool entails(fluent) = 0;
    virtual bool entails(const fluent_set &);
    virtual bool entails(const fluent_formula &);

    bool entails(const formula_list &);
    virtual bool entails(const belief_formula &) = 0;

    //virtual state build_initial(initally) = 0;

    bool is_goal();

protected:

    action_id_list m_executed_actions_id;
    unsigned short m_plan_length;
    int m_heuristic_value;

};