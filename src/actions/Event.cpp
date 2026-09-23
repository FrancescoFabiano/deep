#include "Event.h"

#include <utility>

Event::Event(const EventId id, std::string name)
    : m_id(id), m_name(std::move(name)) {}

EventId Event::get_id() const noexcept { return m_id; }

void Event::set_id(const EventId id) noexcept { m_id = id; }

const std::string &Event::get_name() const noexcept { return m_name; }

void Event::set_name(const std::string &name) { m_name = name; }

const BeliefFormula &Event::get_precondition() const noexcept {
  return m_precondition;
}

void Event::set_precondition(const BeliefFormula &precondition) {
  m_precondition = precondition;
}

const Postconditions &Event::get_postconditions() const noexcept {
  return m_postconditions;
}

void Event::add_postcondition(const Fluent &fluent,
                              const BeliefFormula &postcondition) {

  const auto [iterator, inserted] =
      m_postconditions.emplace(fluent, postcondition);

  if (!inserted) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::ActionDuplicatePostcondition,
        "Duplicate postcondition for fluent in event '" + m_name + "'.");
  }
}

bool Event::has_postcondition(const Fluent &fluent) const {

  return m_postconditions.contains(fluent);
}

bool Event::operator<(const Event &other) const noexcept {

  return m_id < other.m_id;
}

bool Event::operator==(const Event &other) const noexcept {

  return m_id == other.m_id;
}