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
 * \brief A grounded multi-pointed DEL action model.
 *
 * \details An Action stores the grounded DEL structure imported from Plank for
 * one EPDDL action declaration. The semantic core is the multi-pointed event
 * model:
 *
 *   A = (E, R, D)
 *
 * where:
 *
 *   E = events;
 *   R = event accessibility relations selected for each agent at execution
 *       time;
 *   D = designated events.
 *
 * Event preconditions and postconditions are stored inside \ref Event.
 *
 * Unlike the legacy single-relation view, DEEP keeps the EPDDL observability
 * information explicitly:
 *   - each observability type owns its own event relation;
 *   - each agent is associated with formula-valued conditions selecting which
 *     observability type applies in the current epistemic state.
 *
 * During successor generation, one observability type is resolved per agent on
 * the source state and the corresponding event relation is used in the product
 * update.
 */
class Action {
public:
  Action() = default;

  Action(std::string name, ActionId id);

  Action(const Action &) = default;
  Action(Action &&) noexcept = default;

  ~Action() = default;

  // === Identity ===

  [[nodiscard]] const std::string &get_name() const noexcept;

  void set_name(const std::string &name);

  [[nodiscard]] const ActionId &get_id() const noexcept;

  void set_id(const ActionId &id);

  // === Events ===

  [[nodiscard]] const Events &get_events() const noexcept;

  [[nodiscard]] const Event &get_event(EventId event_id) const;

  [[nodiscard]] bool has_event(EventId event_id) const noexcept;

  void add_event(const Event &event);

  // === Designated Events ===

  [[nodiscard]] const DesignatedEvents &get_designated_events() const noexcept;

  [[nodiscard]] bool is_designated(EventId event_id) const noexcept;

  void add_designated_event(EventId event_id);

  // === Operators ===

  Action &operator=(const Action &) = default;
  Action &operator=(Action &&) noexcept = default;

  [[nodiscard]] bool operator<(const Action &other) const;

  [[nodiscard]] bool operator==(const Action &other) const;

  // === EPDDL Observability ===

  /**
   * \brief Get the event relation stored for each EPDDL observability type.
   * \return Mapping from observability type to event accessibility relation.
   */
  [[nodiscard]]
  const ObservabilityRelations &get_observability_relations() const noexcept;

  /**
   * \brief Get the event relation associated with one observability type.
   * \param type The observability type to retrieve.
   * \return The event relation used when that type is selected.
   */
  [[nodiscard]]
  const EventRelation &get_observability_relation(ObservabilityType type) const;

  /**
   * \brief Add one event-accessibility edge to the relation of an
   * observability type.
   * \param type The observability type being extended.
   * \param from Source event id.
   * \param to Target event id.
   */
  void add_observability_edge(ObservabilityType type, EventId from, EventId to);

  /**
   * \brief Get all formula-valued observability conditions for all agents.
   * \return Mapping agent -> (observability type -> condition).
   */
  [[nodiscard]]
  const ObservabilityConditions &get_observability_conditions() const noexcept;

  /**
   * \brief Get the observability conditions registered for one agent.
   * \param agent The grounded agent to query.
   * \return The conditions indexed by observability type for \p agent.
   */
  [[nodiscard]]
  const AgentObservabilityConditions &
  get_observability_conditions(const Agent &agent) const;

  /**
   * \brief Register the condition under which an agent uses an
   * observability type.
   * \param agent The grounded agent.
   * \param type The observability type enabled by the condition.
   * \param condition The epistemic condition evaluated on the source state.
   */
  void add_observability_condition(const Agent &agent, ObservabilityType type,
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
   * \brief Event accessibility relation stored for each EPDDL observability
   * type.
   *
   * Each entry represents the relation that becomes active if that
   * observability type is selected for an agent while executing this action.
   */
  ObservabilityRelations m_observability_relations;

  /**
   * \brief Formula-valued observability conditions for each agent.
   *
   * These determine which observability type, and therefore which event
   * relation, applies to each agent when executing this action in a specific
   * epistemic state.
   */
  ObservabilityConditions m_observability_conditions;
};

using ActionsSet = std::set<Action>;

using ActionList = std::vector<Action>;
