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

#include <utility>
#include "utilities/Define.h"
#include "utilities/ExitHandler.h"
#include "heuristics_strategies/SatisfiedGoals.h"
#include "EpistemicPlanningGraph/PlanningGraph.h"
#include "search_strategies/BestFirst.h"

/**
 * \brief Manages the computation and assignment of heuristic values to states.
 */
class HeuristicsManager {
public:
    /**
     * \brief Constructs a HeuristicsManager with the chosen heuristic.
     * \param[in] used_heuristics The heuristic to use.
     */
    explicit HeuristicsManager(Heuristics used_heuristics);

    /**
     * \brief Computes and sets the heuristic value for a given state.
     * \tparam T The state representation type.
     * \param[in,out] eState The state to update with the calculated heuristic value.
     *
     * \note If the heuristic value is negative, it indicates that the heuristic is not applicable or the state does not satisfy the goals so needs to be discarded.
     */
    template<StateRepresentation T>
    void set_heuristic_value(const State<T> &eState) {
        switch (m_used_heuristics) {
            case Heuristics::L_PG: {
                const PlanningGraph pg(m_goals, eState);
                eState.set_heuristic_value(pg.is_satisfiable() ? pg.get_length() : -1);
                break;
            }
            case Heuristics::S_PG: {
                const PlanningGraph pg(m_goals, eState);
                eState.set_heuristic_value(pg.is_satisfiable() ? pg.get_sum() : -1);
                break;
            }
            case Heuristics::C_PG: {
                short h_value = 0;

                if (m_pg_goal_not_found) {
                    h_value = -1; // Goal not reachable, set heuristic to -1.
                } else {
                    for (const auto &[fluent, score] : m_fluents_score) {
                        if (eState.entails(fluent) && score > 0) h_value += score;
                    }
                    for (const auto &[belief, score] : m_bf_score) {
                        if (eState.entails(belief) && score > 0) h_value += score;
                    }
                    h_value = static_cast<short>(100 - ((static_cast<float>(h_value) / m_pg_max_score) * 100)); // Invert: 0 is 100%, 100 is 0%
                }

                eState.set_heuristic_value(h_value);
                break;
            }
            case Heuristics::SUBGOALS: {
                eState.set_heuristic_value(SatisfiedGoals::get_instance().get_unsatisfied_goals(eState));
                break;
            }
            case Heuristics::GNN: {
                eState.set_heuristic_value(get_gnn_score(eState));
                break;
            }
            default: {
                ExitHandler::exit_with_message(
                    ExitHandler::ExitCode::HeuristicsBadDeclaration,
                    "Wrong Heuristic Selection in HeuristicsManager. Please check the heuristic type."
                );
                break;
            }
        }
    }

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
     * \brief Sets the goals (CNF of expanded subgoals).
     * \param[in] to_set The CNF of expanded subgoals.
     */
    void set_goals(const FormulaeList &to_set);

    /**
     * \brief Gets the goals (CNF of expanded subgoals).
     * \return The CNF of expanded subgoals.
     */
    [[nodiscard]] const FormulaeList &get_goals() const noexcept;

    /**
     * \brief Assignment operator.
     * \param[in] to_copy The HeuristicsManager to assign from.
     * \return true if the assignment succeeded, false otherwise.
     */
    bool operator=(const HeuristicsManager &to_copy);

private:
    Heuristics m_used_heuristics = Heuristics::ERROR; ///< The type of heuristic used.
    FormulaeList m_goals {}; ///< The goal description, possibly expanded for heuristic use.

    /// \name Planning Graph Related
    ///@{
    pg_f_map m_fluents_score {}; ///< Map of fluent scores (used by C_PG heuristic).
    pg_bf_map m_bf_score {}; ///< Map of belief formula scores (used by C_PG heuristic).
    bool m_pg_goal_not_found = false; ///< Flag indicating if the planning graph determined that the goal is not reachable.
    int m_pg_max_score = 0; ///< Maximum score for the planning graph (used for normalization).
    ///@}


    /**
     * \brief Expands group formulae to generate more subgoals.
     * \details For example, C([a,b], φ) is expanded into:
     *   - B(a, φ), B(b, φ), B(a, B(b, φ)), B(b, B(a, φ)), etc., up to the specified nesting depth.
     * \param[in] nesting The maximum depth for generated subgoals (default: 2).
     */
    void expand_goals(unsigned short nesting);

    /**
     * \brief Recursively generates nested subgoals.
     * \param[in] nesting The maximum nesting depth.
     * \param[in] depth The current depth.
     * \param[in] to_explore The belief formula to expand.
     * \param[in] agents The agents for whom to generate subgoals.
     */
    void produce_subgoals(unsigned short nesting, unsigned short depth, const BeliefFormula &to_explore,
                          const AgentsSet &agents);
};
