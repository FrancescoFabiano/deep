/**
 * \brief Implementation of \ref State.h
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 20, 2025
 */

#include "Domain.h"
#include "State.h"

template <StateRepresentation T>
State<T>::State(const State& prev_state, const Action& executed_action)
{
    if (prev_state.is_executable(executed_action))
    {
        (*this) = prev_state.compute_successor(executed_action);
    }
    else
    {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::StateActionNotExecutableError,
            "Error: The action needed to compute the next state is not executable."
        );
    }
}

template <StateRepresentation T>
State<T>::State(const State& other)
    : m_representation(other.m_representation),
      m_executed_actions_id(other.m_executed_actions_id),
      m_heuristic_value(other.m_heuristic_value)
{
}

template <StateRepresentation T>
State<T> State<T>::compute_successor(const Action& executed_action)
{
    State<T> next_state;
    next_state.set_representation(get_representation().compute_successor(executed_action));
    next_state.add_executed_action(executed_action);

    return next_state;
}


template <StateRepresentation T>
const ActionIdsList& State<T>::get_executed_actions() const
{
    return m_executed_actions_id;
}

template <StateRepresentation T>
unsigned short State<T>::get_plan_length() const
{
    return m_executed_actions_id.size();
}

template <StateRepresentation T>
short State<T>::get_heuristic_value() const
{
    return m_heuristic_value;
}

template <StateRepresentation T>
const T& State<T>::get_representation() const
{
    return m_representation;
}


template <StateRepresentation T>
State<T>& State<T>::operator=(const State& to_assign)
{
    set_representation(to_assign.get_representation());
    set_executed_actions(to_assign.get_executed_actions());
    set_heuristic_value(to_assign.get_heuristic_value());
    return (*this);
}

template <StateRepresentation T>
bool State<T>::operator<(const State& to_compare) const
{
    return m_representation < to_compare.get_representation();
}

template <StateRepresentation T>
void State<T>::set_executed_actions(const ActionIdsList& to_set)
{
    m_executed_actions_id = to_set;
}

template <StateRepresentation T>
void State<T>::add_executed_action(const Action& to_add)
{
    m_executed_actions_id.push_back(to_add.get_id());
}

template <StateRepresentation T>
void State<T>::set_heuristic_value(const short heuristic_value)
{
    m_heuristic_value = heuristic_value;
}

template <StateRepresentation T>
void State<T>::set_representation(const T& to_set)
{
    m_representation = to_set;
}

template <StateRepresentation T>
bool State<T>::entails(const Fluent& to_check) const
{
    return m_representation.entails(to_check);
}

template <StateRepresentation T>
bool State<T>::entails(const FluentsSet& to_check) const
{
    return m_representation.entails(to_check);
}

template <StateRepresentation T>
bool State<T>::entails(const FluentFormula& to_check) const
{
    return m_representation.entails(to_check);
}

template <StateRepresentation T>
bool State<T>::entails(const BeliefFormula& to_check) const
{
    return m_representation.entails(to_check);
}

template <StateRepresentation T>
bool State<T>::entails(const FormulaeList& to_check) const
{
    return m_representation.entails(to_check);
}

template <StateRepresentation T>
void State<T>::build_initial()
{
    m_representation.build_initial();
}

template <StateRepresentation T>
void State<T>::contract_with_bisimulation()
{
    m_representation.contract_with_bisimulation();
}

template <StateRepresentation T>
bool State<T>::is_executable(const Action& act) const
{
    return entails(act.get_executability());
}

template <StateRepresentation T>
bool State<T>::is_goal() const
{
    return entails(Domain::get_instance().get_goal_description());
}

template <StateRepresentation T>
void State<T>::print() const
{
    m_representation.print();
}

template <StateRepresentation T>
void State<T>::print_dot_format(std::ofstream& ofs) const
{
    m_representation.print_dot_format(ofs);
}

template <StateRepresentation T>
void State<T>::print_dataset_format(std::ofstream& ofs) const
{
    m_representation.print_dataset_format(ofs);
}
