#pragma once

#include <set>
#include <vector>
#include <string>
#include <utility>

#include "Proposition.h"
#include "utilities/define.h"

/**
 * \class Action
 * \brief Stores an action and all its information.
 * \author Francesco Fabiano
 * \date May 16, 2025
 * \copyright GNU Public License.
 */
class Action {
public:
    /// \name Constructors
    ///@{
    /** \brief Default constructor. */
    Action() = default;

    /**
     * \brief Constructor with a given name and id.
     * \param[in] name The value to assign to \ref m_name.
     * \param[in] id The value to assign to \ref m_id.
     */
    Action(const std::string& name, ActionId id);
    ///@}

    /// \name Getters and Setters
    ///@{
    /** \brief Gets the name of this action. */
    [[nodiscard]] std::string get_name() const;

    /** \brief Sets the name of this action.
     *  \param[in] name The value to assign to \ref m_name.
     */
    void set_name(const std::string& name);

    /** \brief Gets the executor agent of this action. */
    [[nodiscard]] Agent get_executor() const;

    /** \brief Sets the executor agent of this action.
     *  \param[in] executor The value to assign to \ref m_executor.
     */
    void set_executor(const Agent& executor);

    /** \brief Gets the unique id of this action. */
    [[nodiscard]] ActionId get_id() const;

    /** \brief Sets the unique id of this action.
     *  \param[in] id The value to assign to \ref m_id.
     */
    void set_id(ActionId id);

    /** \brief Gets the proposition type of this action. */
    [[nodiscard]] PropositionType get_type() const;

    /** \brief Sets the proposition type of this action.
     *  \param[in] type The value to assign to \ref m_type.
     */
    void set_type(PropositionType type);

    /** \brief Gets the executability conditions of this action. */
    [[nodiscard]] const formula_list& get_executability() const;

    /** \brief Gets the effects of this action. */
    [[nodiscard]] const effects_map& get_effects() const;

    /** \brief Gets the fully observant agents and their conditions. */
    [[nodiscard]] const observability_map& get_fully_observants() const;

    /** \brief Gets the partially observant agents and their conditions. */
    [[nodiscard]] const observability_map& get_partially_observants() const;
    ///@}

    /// \name Main Methods
    ///@{
    /**
     * \brief Parses a proposition and adds its information to this action.
     *
     * Uses add_executability, add_effect, add_fully_observant, and add_partially_observant
     * to add the appropriate behavior to this action.
     * \param[in] to_add The proposition to add.
     */
    void add_proposition(Proposition& to_add);

    /** \brief Prints this action. */
    void print() const;

    /** \brief Operator < implemented to use Action in std::set. */
    bool operator<(const Action&) const;

    /** \brief Assignment operator. */
    Action& operator=(const Action&);
    ///@}

private:
    /// \name Fields
    ///@{
    std::string m_name;                ///< The name of this action.
    ActionId m_id;                     ///< The unique id of this action (calculated with grounder).
    Agent m_executor;                  ///< The agent that executes the action.
    PropositionType m_type = NOTSET;  ///< The proposition type of this action.

    formula_list m_executability;      ///< Executability conditions.
    observability_map m_fully_observants;    ///< Fully observant agents and their conditions.
    observability_map m_partially_observants;///< Partially observant agents and their conditions.
    effects_map m_effects;             ///< Effects and their conditions.
    ///@}

    /// \name Private Methods
    ///@{
    /**
     * \brief Adds an executability condition to this action.
     * \param[in] to_add The belief_formula representing the executability condition to add.
     */
    void add_executability(const belief_formula& to_add);

    /**
     * \brief Adds an effect (with its conditions) to this action.
     * \param[in] to_add The fluent_formula representing the effect to add.
     * \param[in] condition The condition of to_add.
     */
    void add_effect(const FluentFormula& to_add, const belief_formula& condition);

    /**
     * \brief Adds a fully observant agent (with its conditions) to this action.
     * \param[in] ag The agent that is fully observant if condition holds.
     * \param[in] condition The condition for ag to be fully observant.
     */
    void add_fully_observant(const Agent& ag, const belief_formula& condition);

    /**
     * \brief Adds a partially observant agent (with its conditions) to this action.
     * \param[in] ag The agent that is partially observant if condition holds.
     * \param[in] condition The condition for ag to be partially observant.
     */
    void add_partially_observant(const Agent& ag, const belief_formula& condition);
    ///@}
};

/// \brief A set of Action objects.
using ActionsSet = std::set<Action>;

/// \brief A sequential execution of Action objects.
using ActionList = std::vector<Action>;