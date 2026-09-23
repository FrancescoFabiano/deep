#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "formulae/BeliefFormula.h"
#include "utilities/Define.h"

/**
 * \class Event
 * \brief A single event in a DEL event model.
 *
 * Each event has:
 *  - an identifier;
 *  - a name;
 *  - a precondition;
 *  - postconditions.
 *
 * A postcondition p -> phi means:
 *
 *   after execution of this event,
 *   p is true iff phi was true in the source world.
 */
class Event {
public:
  Event() = default;

  Event(EventId id, std::string name);

  Event(const Event &) = default;
  Event(Event &&) noexcept = default;

  ~Event() = default;

  [[nodiscard]] EventId get_id() const noexcept;

  void set_id(EventId id) noexcept;

  [[nodiscard]] const std::string &get_name() const noexcept;

  void set_name(const std::string &name);

  [[nodiscard]] const BeliefFormula &get_precondition() const noexcept;

  void set_precondition(const BeliefFormula &precondition);

  [[nodiscard]] const Postconditions &get_postconditions() const noexcept;

  void add_postcondition(const Fluent &fluent,
                         const BeliefFormula &postcondition);

  [[nodiscard]] bool has_postcondition(const Fluent &fluent) const;

  Event &operator=(const Event &) = default;
  Event &operator=(Event &&) noexcept = default;

  [[nodiscard]] bool operator<(const Event &other) const noexcept;

  [[nodiscard]] bool operator==(const Event &other) const noexcept;

private:
  EventId m_id = 0;

  std::string m_name;

  BeliefFormula m_precondition;

  Postconditions m_postconditions;
};