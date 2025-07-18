#pragma once
#include "Grounder.h"
#include "InitialStateInformation.h"
#include "actions/Action.h"
#include "utilities/Define.h"
#include <string>

/**
 * \class Domain
 * \brief Singleton class that stores and manages all domain-specific
 * information for the planner.
 *
 * The Domain class is responsible for reading, storing, and processing all
 * relevant information about the planning domain from the input file. This
 * includes fluents, actions, agents, initial State descriptions, and goal
 * conditions. The class ensures that all domain data is available in structured
 * form for the rest of the planner.
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

  /** \brief Function that builds all the domain information.     */
  void build();

  /** \brief Getter of the field \ref m_fluents. */
  [[nodiscard]] const FluentsSet &get_fluents() const noexcept;
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
  /** \brief Getter of the field \ref m_initial_description. */
  [[nodiscard]] const InitialStateInformation &
  get_initial_description() const noexcept;
  /** \brief Getter of the field \ref m_goal_description. */
  [[nodiscard]] const FormulaeList &get_goal_description() const noexcept;

  /** \brief Copy constructor removed since is Singleton class. */
  Domain(const Domain &) = delete;
  /** \brief Copy operator removed since Singleton class. */
  Domain &operator=(const Domain &) = delete;

private:
  std::string
      m_name; ///< The name of the file that contains the description of *this*.
  // Grounder m_grounder; ///< A \ref grounder object used to store the name of
  // the information.
  FluentsSet m_fluents; ///< Set containing all the (grounded) Fluent of
                        ///< the domain.
  ActionsSet m_actions; ///< Set containing all the Action (with effects,
                        ///< conditions, obs etc.) of the domain.
  AgentsSet
      m_agents; ///< Set containing all the (grounded) Agent of the domain.
  InitialStateInformation
      m_initial_description;       ///< The description of the initial State.
  FormulaeList m_goal_description; ///< The formula that describes the goal.

  /**
   * \brief Stores agent information from the input file.
   * \param grounder The Grounder object being filled with agent information,
   * which will later be assigned to the helper print.
   */
  void build_agents(Grounder &grounder);

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

  /** \brief Function that adds each proposition to the correct action.
   */
  void build_propositions();

  /** \brief Function that builds the initial state static description.
   */
  void build_initially();

  /** \brief Function that builds the goal description.     */
  void build_goal();

  /** Private constructor since it is a Singleton class. */
  Domain();
};
