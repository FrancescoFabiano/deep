/**
 * \brief Implementation of \ref State.h
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 20, 2025
 */

#include "Domain.h"
#include "SpaceSearcher.h"
#include "State.h"

template <class T>
State<T>::State(const State & prev_state, const Action & executed_action)
{
	if (prev_state.is_executable(executed_action))
	{
		(*this) = SpaceSearcher<T>::compute_successor(prev_state,executed_action);
	}
	else
	{
		ExitHandler::exit_with_message(
			ExitHandler::ExitCode::StateActionNotExecutableError,
			"Error: The action needed to compute the next state is not executable."
		);
	}
}


template <class T>
const ActionIdsList & State<T>::get_executed_actions() const
{
	return m_executed_actions_id;
}

template <class T>
unsigned short State<T>::get_plan_length() const
{
	return m_executed_actions_id.size();
}

template <class T>
short State<T>::get_heuristic_value() const
{
	return m_heuristic_value;
}

template <class T>
const T & State<T>::get_representation() const
{
	return m_representation;
}


template <class T>
bool State<T>::operator=(const State & given_state)
{
	set_representation(given_state.get_representation());
	set_executed_actions(given_state.get_executed_actions());
	set_plan_length(given_state.get_plan_length());
	set_heuristic_value(given_state.get_heuristic_value());
	return true;
}

template <class T>
bool State<T>::operator<(const State & to_compare) const
{
	return m_representation < to_compare.get_representation();
}

template <class T>
void State<T>::set_executed_actions(const ActionIdsList & to_set)
{
	m_executed_actions_id = to_set;
}

template <class T>
void State<T>::add_executed_action(const Action & to_add)
{
	m_executed_actions_id.push_back(to_add.get_id());
}

template <class T>
void State<T>::set_heuristic_value(const short heuristic_value)
{
	m_heuristic_value = heuristic_value;
}

template <class T>
void State<T>::set_representation(const T & to_set)
{
	m_representation = to_set;
}

template <class T>
bool State<T>::entails(const Fluent & to_check) const
{
	return m_representation.entails(to_check);
}

template <class T>
bool State<T>::entails(const FluentsSet & to_check) const
{
	return m_representation.entails(to_check);
}

template <class T>
bool State<T>::entails(const FluentFormula & to_check) const
{
	return m_representation.entails(to_check);
}

template <class T>
bool State<T>::entails(const BeliefFormula & to_check) const
{
	return m_representation.entails(to_check);
}

template <class T>
bool State<T>::entails(const FormulaeList & to_check) const
{
	return m_representation.entails(to_check);
}

template <class T>
void State<T>::build_initial()
{
	m_representation.build_initial();
}

template <class T>
void State<T>::contract_with_bisimulation()
{
	m_representation.contract_with_bisimulation();
}

template <class T>
bool State<T>::is_executable(const Action & act) const
{
	return entails(act.get_executability());
}

template <class T>
bool State<T>::is_goal() const
{
	return entails(Domain::get_instance().get_goal_description());
}

template <class T>
void State<T>::print(std::ostream &os) const
{
	m_representation.print(os);
}

template <class T>
void State<T>::print_dot_format(std::ostream &os) const
{
	m_representation.print_dot_format(os);
}