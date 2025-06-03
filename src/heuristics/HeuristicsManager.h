/**
 * \class HeuristicsManager
 * \brief Assigns heuristic scores to states using the selected heuristic.
 *
 * \copyright GNU Public License.
 * 
 * \author Francesco Fabiano.
 * \date May 31, 2025
 */

#pragma once

#include "utilities/Define.h"
#include "strategies/SatisfiedGoals.h"
#include "epg/PlanningGraph.h"
#include "neuralnets/GraphNN.h"

/**
 * \brief Manages the computation and assignment of heuristic values to states.
 */
template <StateRepresentation StateRepr>
class HeuristicsManager
{
public:
    /**
     * \brief Constructs a HeuristicsManager with the chosen heuristic.
     */
    explicit HeuristicsManager();

    /**
     * \brief Computes and sets the heuristic value for a given state.
     * \param[in,out] eState The state to update with the calculated heuristic value.
     *
     * \note If the heuristic value is negative, it indicates that the heuristic is not applicable or the state does not satisfy the goals and needs to be discarded.
     */
    void set_heuristic_value(State<StateRepr>& eState);

    /**
     * \brief Sets the heuristic to use.
     * \param[in] used_h The heuristic to assign.
     */
    void set_used_h(Heuristics used_h) noexcept;

    /**
     * \brief Gets the currently used heuristic.
     * \return The heuristic in use.
     */
    [[nodiscard]] Heuristics get_used_h() const noexcept;

    /**
     * \brief Gets the name of the currently used heuristic.
     * \return The name of the heuristic in use.
     */
    [[nodiscard]] std::string get_used_h_name() const noexcept;

    /**
     * \brief Sets the goals (CNF of expanded subgoals).
     * \param[in] to_set The CNF of expanded subgoals.
     */
    void set_goals(const FormulaeList& to_set);

    /**
     * \brief Gets the goals (CNF of expanded subgoals).
     * \return The CNF of expanded subgoals.
     */
    [[nodiscard]] const FormulaeList& get_goals() const noexcept;

private:
    Heuristics m_used_heuristics = Heuristics::ERROR; ///< The type of heuristic used.
    FormulaeList m_goals{}; ///< The goal description, possibly expanded for heuristic use.

    /// \name Planning Graph Related
    ///@{
    PG_FluentsScoreMap m_fluents_score{}; ///< Map of fluent scores (used by C_PG heuristic).
    PG_BeliefFormulaeMap m_bf_score{}; ///< Map of belief formula scores (used by C_PG heuristic).
    bool m_pg_goal_not_found = false;
    ///< Flag indicating if the planning graph determined that the goal is not reachable.
    int m_pg_max_score = 0; ///< Maximum score for the planning graph (used for normalization).
    ///@}

    /**
     * \brief Expands group formulae to generate more subgoals.
     * \details For example, C([a,b], φ) is expanded into:
     *   - B(a, φ), B(b, φ), B(a, B(b, φ)), B(b, B(a, φ)), etc., up to the specified nesting depth.
     * \param[in] nesting The maximum depth for generated subgoals (default: 2).
     */
    void expand_goals(unsigned short nesting = 2);

    /**
     * \brief Recursively generates nested subgoals.
     * \param[in] nesting The maximum nesting depth.
     * \param[in] depth The current depth.
     * \param[in] to_explore The belief formula to expand.
     * \param[in] agents The agents for whom to generate subgoals.
     */
    void produce_subgoals(unsigned short nesting, unsigned short depth, const BeliefFormula& to_explore,
                          const AgentsSet& agents);
};

#include "HeuristicsManager.tpp"
