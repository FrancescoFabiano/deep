#include "Action.h"

#include <stdexcept>
#include <utility>

Action::Action(std::string name, ActionId id)
    : m_name(std::move(name)), m_id(std::move(id)) {}

// === Identity ===

const std::string &Action::get_name() const noexcept { return m_name; }

void Action::set_name(const std::string &name) { m_name = name; }

const ActionId &Action::get_id() const noexcept { return m_id; }

void Action::set_id(const ActionId &id) { m_id = id; }

// === Events ===

const Events &Action::get_events() const noexcept { return m_events; }

const Event &Action::get_event(const EventId event_id) const {

  return m_events.at(event_id);
}

bool Action::has_event(const EventId event_id) const noexcept {

  return m_events.contains(event_id);
}

void Action::add_event(const Event &event) {

  const auto [iterator, inserted] = m_events.emplace(event.get_id(), event);

  if (!inserted) {
    throw std::invalid_argument("Duplicate event id in action '" + m_name +
                                "'.");
  }
}

// === Designated Events ===

const DesignatedEvents &Action::get_designated_events() const noexcept {

  return m_designated_events;
}

bool Action::is_designated(EventId event_id) const noexcept {

  return m_designated_events.contains(event_id);
}

void Action::add_designated_event(EventId event_id) {

  if (!has_event(event_id)) {
    throw std::invalid_argument("Cannot designate unknown event in action '" +
                                m_name + "'.");
  }

  m_designated_events.insert(event_id);
}

// === Operators ===

bool Action::operator<(const Action &other) const { return m_id < other.m_id; }

bool Action::operator==(const Action &other) const {

  return m_id == other.m_id;
}

const ObservabilityRelations &
Action::get_observability_relations() const noexcept {
  return m_observability_relations;
}

const EventRelation &
Action::get_observability_relation(const ObservabilityType type) const {

  const auto it = m_observability_relations.find(type);

  if (it == m_observability_relations.end()) {
    throw std::out_of_range("Observability type does not exist in action '" +
                            m_name + "'.");
  }

  return it->second;
}

void Action::add_observability_edge(const ObservabilityType type,
                                    const EventId from, const EventId to) {

  m_observability_relations[type][from].insert(to);
}

const ObservabilityConditions &
Action::get_observability_conditions() const noexcept {
  return m_observability_conditions;
}

const AgentObservabilityConditions &
Action::get_observability_conditions(const Agent &agent) const {

  const auto it = m_observability_conditions.find(agent);

  if (it == m_observability_conditions.end()) {
    throw std::out_of_range(
        "Agent has no observability conditions in action '" + m_name + "'.");
  }

  return it->second;
}

void Action::add_observability_condition(const Agent &agent,
                                         const ObservabilityType type,
                                         const BeliefFormula &condition) {

  m_observability_conditions[agent][type] = condition;
}