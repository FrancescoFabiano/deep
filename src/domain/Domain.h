#pragma once
#include <memory>
#include <string>
#include "grounder.h"
#include "initially.h"
#include "../utilities/define.h"
#include "../utilities/reader.h"
#include "../actions/action.h"
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

/**
 * \class Domain
 * \brief Singleton class that stores and manages all domain-specific information for the planner.
 *
 * The Domain class is responsible for reading, storing, and processing all relevant information
 * about the planning domain from the input file. This includes fluents, actions, agents, initial
 * state descriptions, and goal conditions. The class ensures that all domain data is available
 * in structured form for the rest of the planner.
 *
 * This class follows the Singleton pattern: only one instance exists during the application's lifetime.
 * All access to domain data should be performed through this single instance.
 *
 * \note The domain is initialized once at startup, and its data remains constant throughout execution.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 14, 2025
 */
class Domain
{
public:
    /** \brief To get always (the same instance of) *this* and the same instantiated fields.
     * \warning the \ref set_domain has to called in the main file only.*/
    static Domain& get_instance();

    /** \brief Setter for the domains parameters. */
    static void create_instance();

    /** \brief Function that builds all the domain information. */
    void build();

    /** \brief Getter of the field \ref m_grounder. */
    [[nodiscard]] const grounder& get_grounder() const noexcept;
    /** \brief Getter of the field \ref m_fluents. */
    [[nodiscard]] const fluent_set& get_fluents() const noexcept;
    /** \brief Function that returns the number of \ref fluent in the domain. */
    [[nodiscard]] unsigned int get_fluent_number() const noexcept;
    /** \brief Function that returns the size of the fluent set (which also include negations). */
    [[nodiscard]] unsigned int get_size_fluent() const noexcept;
    /** \brief Getter of the field \ref m_actions. */
    [[nodiscard]] const action_set& get_actions() const noexcept;
    /** \brief Getter of the field \ref m_agents. */
    [[nodiscard]] const agent_set& get_agents() const noexcept;
    /** \brief Function that returns the number of agents in the domain. */
    [[nodiscard]] unsigned int get_agent_number() const noexcept;
    /** \brief Getter of the field \ref m_name. */
    [[nodiscard]] const std::string& get_name() const noexcept;
    /** \brief Getter of the field \ref m_intial_description. */
    [[nodiscard]] const initially& get_initial_description() const noexcept;
    /** \brief Getter of the field \ref m_goal_description. */
    [[nodiscard]] const formula_list& get_goal_description() const noexcept;

    /** \brief Copy constructor removed since is Singleton class. */
    Domain(const Domain&) = delete;
    /** \brief Copy operator removed since Singleton class. */
    Domain& operator=(const Domain&) = delete;

private:
    std::string m_name; ///< The name of the file that contains the description of *this*.
    boost::shared_ptr<reader> m_reader; ///< The pointer to a \ref reader object.
    grounder m_grounder; ///< A \ref grounder object used to store the name of the information.
    fluent_set m_fluents; ///< Set containing all the (grounded) \ref fluent of the domain.
    action_set m_actions; ///< Set containing all the \ref action (with effects, conditions, obsv etc.) of the domain.
    agent_set m_agents; ///< Set containing all the (grounded) \ref agent of the domain.
    initially m_intial_description; ///< The description of the initial state.
    formula_list m_goal_description; ///< The formula that describes the goal.

    void build_agents(); ///< Function that from the file stores the \ref agent information.
    void build_fluents(); ///< Function that from the file stores the \ref fluent information.
    void build_actions(); ///< Function that from the file stores the \ref action information.
    void build_propositions(); ///< Function that adds to the right \ref action each \ref proposition.
    void build_initially(); ///< Function that builds \ref m_intial_description.
    void build_goal(); ///< Function that builds \ref m_goal_description.

    Domain(); ///< Private constructor since it is a Singleton class.

    static std::unique_ptr<Domain> instance; ///< Singleton instance
};