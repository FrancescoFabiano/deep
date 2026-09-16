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
      std::string name,
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



  // === Operators ===

  Action &operator=(const Action &) = default;
  Action &operator=(Action &&) noexcept = default;

  [[nodiscard]] bool
  operator<(const Action &other) const;

  [[nodiscard]] bool
  operator==(const Action &other) const;


    // === EPDDL Observability ===

    [[nodiscard]]
    const ObservabilityRelations &
    get_observability_relations() const noexcept;

    [[nodiscard]]
    const EventRelation &
    get_observability_relation(
        ObservabilityType type) const;

    void add_observability_edge(
        ObservabilityType type,
        EventId from,
        EventId to);

    [[nodiscard]]
    const ObservabilityConditions &
    get_observability_conditions() const noexcept;

    [[nodiscard]]
    const AgentObservabilityConditions &
    get_observability_conditions(
        const Agent &agent) const;

    void add_observability_condition(
        const Agent &agent,
        ObservabilityType type,
        const BeliefFormula &condition);

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
 * Relations over events defined for each EPDDL observability type.
 *
 * These are part of the grounded action specification.
 */
    ObservabilityRelations m_observability_relations;

    /**
     * Formula-valued observability conditions for each agent.
     *
     * These determine which observability relation applies when
     * executing this action in a particular epistemic state.
     */
    ObservabilityConditions m_observability_conditions;
};

using ActionsSet = std::set<Action>;

using ActionList = std::vector<Action>;