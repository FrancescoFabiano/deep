#pragma once

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "Event.h"
#include "utilities/Define.h"

/**
 * \class Action
 * \brief A grounded multi-pointed DEL event model.
 *
 * An Action is:
 *
 *   A = (E, R, D)
 *
 * where:
 *
 *   E = events;
 *   R = accessibility relations between events for each agent;
 *   D = designated events.
 *
 * Event preconditions and postconditions are stored inside Event.
 */
class Action {
public:

  Action() = default;

  Action(
      const std::string &name,
      ActionId id);

  Action(const Action &) = default;
  Action(Action &&) noexcept = default;

  ~Action() = default;

  // === Identity ===

  [[nodiscard]] const std::string &
  get_name() const noexcept;

  void set_name(
      const std::string &name);

  [[nodiscard]] const ActionId &
  get_id() const noexcept;

  void set_id(
      const ActionId &id);

  // === Events ===

  [[nodiscard]] const Events &
  get_events() const noexcept;

  [[nodiscard]] const Event &
  get_event(EventId event_id) const;

  [[nodiscard]] bool
  has_event(EventId event_id) const noexcept;

  void add_event(
      const Event &event);

  // === Designated Events ===

  [[nodiscard]] const DesignatedEvents &
  get_designated_events() const noexcept;

  [[nodiscard]] bool
  is_designated(EventId event_id) const noexcept;

  void add_designated_event(
      EventId event_id);

  // === Accessibility Relations ===

  [[nodiscard]] const EventRelations &
  get_event_relations() const noexcept;

  [[nodiscard]] const EventRelation &
  get_event_relation(
      const Agent &agent) const;

  [[nodiscard]] bool
  has_event_edge(
      const Agent &agent,
      EventId from,
      EventId to) const;

  void add_event_edge(
      const Agent &agent,
      EventId from,
      EventId to);

  // === Operators ===

  Action &operator=(const Action &) = default;
  Action &operator=(Action &&) noexcept = default;

  [[nodiscard]] bool
  operator<(const Action &other) const;

  [[nodiscard]] bool
  operator==(const Action &other) const;

private:
  std::string m_name;

  ActionId m_id;

  /**
   * E: events of the event model.
   */
  Events m_events;

  /**
   * D: designated events.
   *
   * More than one event may be designated.
   */
  DesignatedEvents m_designated_events;

  /**
   * R_a: accessibility relation over events for each agent.
   */
  EventRelations m_event_relations;
};

using ActionsSet = std::set<Action>;

using ActionList = std::vector<Action>;