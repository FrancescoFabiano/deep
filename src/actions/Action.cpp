#include "Action.h"

#include <stdexcept>

Action::Action(
    const std::string &name,
    ActionId id)
    : m_name(name),
      m_id(std::move(id)) {}

// === Identity ===

const std::string &
Action::get_name() const noexcept {
  return m_name;
}

void Action::set_name(
    const std::string &name) {
  m_name = name;
}

const ActionId &
Action::get_id() const noexcept {
  return m_id;
}

void Action::set_id(
    const ActionId &id) {
  m_id = id;
}

// === Events ===

const Events &
Action::get_events() const noexcept {
  return m_events;
}

const Event &
Action::get_event(
    const EventId event_id) const {

  return m_events.at(event_id);
}


bool Action::has_event(
    const EventId event_id) const noexcept {

  return m_events.contains(event_id);
}

void Action::add_event(
    const Event &event) {

  const auto [iterator, inserted] =
      m_events.emplace(
          event.get_id(),
          event);

  if (!inserted) {
    throw std::invalid_argument(
        "Duplicate event id in action '" +
        m_name + "'.");
  }
}

// === Designated Events ===

const DesignatedEvents &
Action::get_designated_events() const noexcept {

  return m_designated_events;
}

bool Action::is_designated(
    EventId event_id) const noexcept {

  return m_designated_events.contains(
      event_id);
}

void Action::add_designated_event(
    EventId event_id) {

  if (!has_event(event_id)) {
    throw std::invalid_argument(
        "Cannot designate unknown event in action '" +
        m_name + "'.");
  }

  m_designated_events.insert(
      event_id);
}

// === Accessibility Relations ===

const EventRelations &
Action::get_event_relations() const noexcept {

  return m_event_relations;
}

const EventRelation &
Action::get_event_relation(
    const Agent &agent) const {

  static const EventRelation empty_relation;

  const auto iterator =
      m_event_relations.find(agent);

  if (iterator == m_event_relations.end()) {
    return empty_relation;
  }

  return iterator->second;
}

bool Action::has_event_edge(
    const Agent &agent,
    EventId from,
    EventId to) const {

  const auto relation =
      m_event_relations.find(agent);

  if (relation == m_event_relations.end()) {
    return false;
  }

  return relation->second.contains(
      EventEdge{from, to});
}

void Action::add_event_edge(
    const Agent &agent,
    EventId from,
    EventId to) {

  if (!has_event(from) ||
      !has_event(to)) {

    throw std::invalid_argument(
        "Event edge references unknown event "
        "in action '" +
        m_name + "'.");
  }

  m_event_relations[agent].insert(
      EventEdge{from, to});
}

// === Operators ===

bool Action::operator<(
    const Action &other) const {

  return m_id < other.m_id;
}

bool Action::operator==(
    const Action &other) const {

  return m_id == other.m_id;
}