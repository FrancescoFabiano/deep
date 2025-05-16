#pragma once

#include "utilities/Define.h"
#include <string>

/**
 * \class Grounder
 * \brief Class that grounds all the strings of the domain to their numerical ids.
 *
 *  - Actions are associated to an \ref ActionId.
 *  - Agents (string) are associated to an \ref Agent.
 *  - Fluents (string) are associated to a \ref Fluent.
 *
 * \see Define, Action.
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date April 5, 2019
 */
class Grounder {
public:
    /// \name Constructors
    ///@{
    Grounder() = default;
    Grounder(const FluentMap& given_fluent_map, const AgentsMap& given_agent_map, const ActionNamesMap& given_action_name_map);
    ///@}

    /// \name Setters
    ///@{
    void set_fluent_map(const FluentMap& given_fluent_map);
    void set_agent_map(const AgentsMap& given_agent_map);
    void set_action_name_map(const ActionNamesMap& given_action_name_map);
    ///@}

    /// \name Getters
    ///@{
    [[nodiscard]] const FluentMap& get_fluent_map() const;
    [[nodiscard]] const AgentsMap& get_agent_map() const;
    [[nodiscard]] const ActionNamesMap& get_action_name_map() const;
    ///@}

    /// \name Grounding functions
    ///@{
    /**
     * \brief Returns the grounded value of a fluent name.
     * \param[in] to_ground The fluent name to ground.
     * \return The grounded value.
     */
    [[nodiscard]] Fluent ground_fluent(const std::string& to_ground) const;

    /**
     * \brief Returns the grounded value of a set of fluent names.
     * \param[in] to_ground The set of fluent names to ground.
     * \return The set of grounded values.
     */
    [[nodiscard]] FluentsSet ground_fluent(const StringsSet& to_ground) const;

    /**
     * \brief Returns the grounded value of a set of sets of fluent names.
     * \param[in] to_ground The set of sets of fluent names to ground.
     * \return The set of sets of grounded values.
     */
    [[nodiscard]] FluentFormula ground_fluent(const StringSetsSet& to_ground) const;

    /**
     * \brief Returns the grounded value of an agent name.
     * \param[in] to_ground The agent name to ground.
     * \return The grounded value.
     */
    [[nodiscard]] Agent ground_agent(const std::string& to_ground) const;

    /**
     * \brief Returns the grounded value of a set of agent names.
     * \param[in] to_ground The set of agent names to ground.
     * \return The set of grounded values.
     */
    [[nodiscard]] AgentsSet ground_agent(const StringsSet& to_ground) const;

    /**
     * \brief Returns the grounded value of an action name.
     * \param[in] to_ground The action name to ground.
     * \return The grounded value.
     */
    [[nodiscard]] ActionId ground_action(const std::string& to_ground) const;
    ///@}

    /// \name De-grounding functions
    ///@{
    /**
     * \brief Returns the name of a grounded fluent.
     * \param[in] to_deground The grounded fluent value.
     * \return The name of the fluent.
     */
    [[nodiscard]] std::string deground_fluent(const Fluent& to_deground) const;

    /**
     * \brief Returns the set of names of grounded fluents.
     * \param[in] to_deground The set of grounded fluent values.
     * \return The set of fluent names.
     */
    [[nodiscard]] StringsSet deground_fluent(const FluentsSet& to_deground) const;

    /**
     * \brief Returns the set of sets of names of grounded fluents.
     * \param[in] to_deground The set of sets of grounded fluent values.
     * \return The set of sets of fluent names.
     */
    [[nodiscard]] StringSetsSet deground_fluent(const FluentFormula& to_deground) const;

    /**
     * \brief Returns the name of a grounded agent.
     * \param[in] to_deground The grounded agent value.
     * \return The name of the agent.
     */
    [[nodiscard]] std::string deground_agent(const Agent& to_deground) const;

    /**
     * \brief Returns the set of names of grounded agents.
     * \param[in] to_deground The set of grounded agent values.
     * \return The set of agent names.
     */
    [[nodiscard]] StringsSet deground_agents(const AgentsSet& to_deground) const;

    /**
     * \brief Returns the name of a grounded action.
     * \param[in] to_deground The grounded action value.
     * \return The name of the action.
     */
    [[nodiscard]] std::string deground_action(const ActionId& to_deground) const;
    ///@}

    /// \name Operators
    ///@{
    /**
     * \brief Copy assignment operator.
     * \param[in] to_copy The Grounder to assign from.
     * \return true if the assignment went through, false otherwise.
     */
    Grounder& operator=(const Grounder& to_copy);
    ///@}

private:
    FluentMap m_fluent_map;           ///< Maps fluent names to grounded values.
    AgentsMap m_agent_map;            ///< Maps agent names to grounded values.
    ActionNamesMap m_action_name_map; ///< Maps action names to grounded values.

    ReverseFluentsMap r_fluent_map;        ///< Maps grounded fluents to names.
    ReverseAgentsMap r_agent_map;          ///< Maps grounded agents to names.
    ReverseActionNamesMap r_action_name_map; ///< Maps grounded actions to names.

    // Reverse map creation helpers (used for printing/debugging)
    void reverse();
    void create_reverse_fl();
    void create_reverse_ag();
    void create_reverse_ac();
};