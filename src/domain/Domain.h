#pragma once
#include "Grounder.h"
#include "actions/Action.h"
#include "utilities/Define.h"
#include <string>

#include "del/semantics/planning_task.h"

/**
 * \class Domain
 * \brief Singleton class that stores and manages all domain-specific
 * information for the planner.
 *
 * \details The Domain class is DEEP's repository for the grounded planning
 * task loaded through Plank. During \ref build it asks Plank to parse, type
 * check, and ground the EPDDL specification, then translates the resulting
 * planning task into DEEP's internal structures:
 *   - grounded fluents and their positive ordering;
 *   - grounded agents;
 *   - grounded DEL action models;
 *   - the initial epistemic state handle;
 *   - the grounded epistemic goal.
 *
 * This class follows the Singleton pattern: only one instance exists during the
 * application's lifetime. All access to domain data should be performed through
 * this single instance.
 *
 * \note The domain is initialized once at startup, and its data remains
 * constant throughout execution.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 14, 2025
 */
class Domain {
public:
  /** \brief To get always (the same instance of) *this* and the same
   * instantiated fields.*/
  static Domain &get_instance();

  /** \brief Build DEEP's internal view of the grounded Plank planning task. */
  void build();

  /** \brief Getter of the field \ref m_fluents. */
  [[nodiscard]] const FluentsSet &get_fluents() const noexcept;
  /** \brief Getter of the field \ref m_positive_fluents. (Useful for bitmask)
   */
  [[nodiscard]] const std::vector<Fluent> &
  get_positive_fluents() const noexcept;
  /** \brief Function that returns the number of Fluent in the domain. */
  [[nodiscard]] unsigned int get_fluent_number() const noexcept;
  /** \brief Function that returns the size of the fluent set (which also
   * include negations). */
  [[nodiscard]] unsigned int get_size_fluent() const noexcept;
  /** \brief Getter of the field \ref m_actions. */
  [[nodiscard]] const ActionsSet &get_actions() const noexcept;
  /** \brief Getter of the field \ref m_agents. */
  [[nodiscard]] const AgentsSet &get_agents() const noexcept;
  /** \brief Function that returns the number of agents in the domain. */
  [[nodiscard]] unsigned int get_agent_number() const noexcept;
  /** \brief Getter of the field \ref m_name. */
  [[nodiscard]] const std::string &get_name() const noexcept;

  /** \brief Get the grounded initial epistemic state produced by Plank. */
  [[nodiscard]] const plank::del::state_ptr &get_initial_state() const noexcept;

  /** \brief Getter of the field \ref m_goal_description. */
  [[nodiscard]] const FormulaeList &get_goal_description() const noexcept;

  /** \brief Copy constructor removed since is Singleton class. */
  Domain(const Domain &) = delete;
  /** \brief Copy operator removed since Singleton class. */
  Domain &operator=(const Domain &) = delete;

private:
  std::string
      m_name; ///< The name of the file that contains the description of *this*.

  /// \brief Grounded planning task imported from Plank.
  plank::del::planning_task m_plank_task;

  FluentsSet m_fluents; ///< Set containing all the (grounded) Fluent of
                        ///< the domain.
  std::vector<Fluent>
      m_positive_fluents; ///< Vector containing all the (grounded) POSITIVE
                          ///< Fluent of the domain ordered (useful for bitmask)
  ActionsSet m_actions;   ///< Set containing all the Action (with effects,
                          ///< conditions, obs etc.) of the domain.
  AgentsSet
      m_agents; ///< Set containing all the (grounded) Agent of the domain.

  /// \brief Agents stored in Plank's grounded order for id-based conversions.
  std::vector<Agent> m_ordered_agents;
  FormulaeList m_goal_description; ///< The formula that describes the goal.

  /**
   * \brief Stores agent information from the input file.
   * \param grounder The Grounder object being filled with agent information,
   * which will later be assigned to the helper print.
   */
  void build_agents(Grounder &grounder);

  const std::vector<Agent> &get_ordered_agents() const noexcept;

  /** \brief Function that stores the fluent information from the file.
   * \param grounder The Grounder object being filled with agent information,
   * which will later be assigned to the helper print.
   */
  void build_fluents(Grounder &grounder);

  /** \brief Function that stores the action information (with effects,
   * conditions, etc.) from the file. \param grounder The Grounder object being
   * filled with agent information, which will later be assigned to the helper
   * print.
   */
  void build_actions(Grounder &grounder);

  /** \brief Function that builds the goal description.     */
  void build_goal();

  /** Private constructor since it is a Singleton class. */
  Domain();
};
