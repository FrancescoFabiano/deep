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

template <StateRepresentation StateRepr>
State<StateRepr>::State(const State &prev_state,
                        const Action &executed_action) {
  if (prev_state.is_executable(executed_action)) {
    (*this) = prev_state.compute_successor(executed_action);
  } else {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::StateActionNotExecutableError,
        "Error: The action needed to compute the next state is not "
        "executable.");
  }
}

template <StateRepresentation StateRepr>
State<StateRepr>::State(const State &other)
    : m_representation(other.m_representation),
      m_executed_actions_id(other.m_executed_actions_id),
      m_heuristic_value(other.m_heuristic_value) {}

template <StateRepresentation StateRepr>
State<StateRepr>
State<StateRepr>::compute_successor(const Action &executed_action) {
  State<StateRepr> next_state;
  next_state.set_representation(
      get_representation().compute_successor(executed_action));
  next_state.set_executed_actions(get_executed_actions());
  next_state.add_executed_action(executed_action);

  return next_state;
}

template <StateRepresentation StateRepr>
const ActionIdsList &State<StateRepr>::get_executed_actions() const {
  return m_executed_actions_id;
}

template <StateRepresentation StateRepr>
unsigned short State<StateRepr>::get_plan_length() const {
  return m_executed_actions_id.size();
}

template <StateRepresentation StateRepr>
short State<StateRepr>::get_heuristic_value() const {
  return m_heuristic_value;
}

template <StateRepresentation StateRepr>
const StateRepr &State<StateRepr>::get_representation() const {
  return m_representation;
}

template <StateRepresentation StateRepr>
State<StateRepr> &State<StateRepr>::operator=(const State &to_assign) {
  set_representation(to_assign.get_representation());
  set_executed_actions(to_assign.get_executed_actions());
  set_heuristic_value(to_assign.get_heuristic_value());
  return (*this);
}

template <StateRepresentation StateRepr>
bool State<StateRepr>::operator<(const State &to_compare) const {
  return m_representation < to_compare.get_representation();
}
template <StateRepresentation StateRepr>
bool State<StateRepr>::operator==(const State<StateRepr> &to_compare) const {
  return !((*this) < to_compare) && !(to_compare < (*this));
}

template <StateRepresentation StateRepr>
bool State<StateRepr>::operator!=(const State<StateRepr> &to_compare) const {
  return ((*this) < to_compare) || (to_compare < (*this));
}

template <StateRepresentation StateRepr>
void State<StateRepr>::set_executed_actions(const ActionIdsList &to_set) {
  m_executed_actions_id = to_set;
}

template <StateRepresentation StateRepr>
void State<StateRepr>::add_executed_action(const Action &to_add) {
  m_executed_actions_id.push_back(to_add.get_id());
}

template <StateRepresentation StateRepr>
void State<StateRepr>::set_heuristic_value(const short heuristic_value) {
  m_heuristic_value = heuristic_value;
}

template <StateRepresentation StateRepr>
void State<StateRepr>::set_representation(const StateRepr &to_set) {
  m_representation = to_set;
}

template <StateRepresentation StateRepr>
bool State<StateRepr>::entails(const Fluent &to_check) const {
  return m_representation.entails(to_check);
}

template <StateRepresentation StateRepr>
bool State<StateRepr>::entails(const FluentsSet &to_check) const {
  return m_representation.entails(to_check);
}

template <StateRepresentation StateRepr>
bool State<StateRepr>::entails(const FluentFormula &to_check) const {
  return m_representation.entails(to_check);
}

template <StateRepresentation StateRepr>
bool State<StateRepr>::entails(const BeliefFormula &to_check) const {
  return m_representation.entails(to_check);
}

template <StateRepresentation StateRepr>
bool State<StateRepr>::entails(const FormulaeList &to_check) const {
  return m_representation.entails(to_check);
}

template <StateRepresentation StateRepr>
void State<StateRepr>::build_initial() {
  m_representation.build_initial();
}

template <StateRepresentation StateRepr>
void State<StateRepr>::contract_with_bisimulation() {
  m_representation.contract_with_bisimulation();
}

template <StateRepresentation StateRepr>
bool State<StateRepr>::is_executable(const Action &act) const {
  return entails(act.get_executability());
}

template <StateRepresentation StateRepr>
bool State<StateRepr>::is_goal() const {
  return entails(Domain::get_instance().get_goal_description());
}

template <StateRepresentation StateRepr> void State<StateRepr>::print() const {
  m_representation.print();
}

template <StateRepresentation StateRepr>
void State<StateRepr>::print_dot_format(std::ofstream &ofs) const {
  m_representation.print_dot_format(ofs);
}

template <StateRepresentation StateRepr>
void State<StateRepr>::print_dataset_format(std::ofstream &ofs,
                                            const bool use_hash,
                                            const bool is_merged) const {
  m_representation.print_dataset_format(ofs, use_hash, is_merged);
}
