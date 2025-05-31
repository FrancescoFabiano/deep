#include "HeuristicsManager.h"

#include <ranges>

#include "Domain.h"

HeuristicsManager::HeuristicsManager(const Heuristics used_heuristics)
{
    set_used_h(used_heuristics);
    m_goals = Domain::get_instance().get_goal_description();
    switch (m_used_heuristics) {
        case Heuristics::GNN:
            break;
        case Heuristics::L_PG:
        case Heuristics::S_PG:
            expand_goals();
            break;
        case Heuristics::C_PG: {
            if (const planning_graph pg; pg.is_satisfiable()) {
                m_pg_max_score = 0;
                m_fluents_score = pg.get_f_scores();
                m_bf_score = pg.get_bf_scores();
                for (const auto &score_f : m_fluents_score | std::views::values) {
                    m_pg_max_score += score_f; // Accumulate positive scores for normalization.
                }
                for (const auto &score_bf: m_bf_score | std::views::values) {
                    m_pg_max_score += score_bf; // Accumulate positive scores for normalization.
                }
            } else {
                m_pg_goal_not_found = true;
            }
            break;
        }
        case Heuristics::SUBGOALS:
            expand_goals();
            satisfied_goals::get_instance().set(m_goals);
            break;
        default:
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::HeuristicsBadDeclaration,
                "Wrong Heuristic Selection in HeuristicsManager. Please check the heuristic type."
            );
            break;
    }
}

void HeuristicsManager::expand_goals(const unsigned short nesting = 2)
{
    FormulaeList original_goal = m_goals;
    for (const auto& formula : original_goal) {
        produce_subgoals(nesting, 0, formula, formula.get_group_agents());
    }
}

void HeuristicsManager::produce_subgoals(unsigned short nesting, unsigned short depth, const BeliefFormula& to_explore, const AgentsSet& agents)
{
    if ((to_explore.get_formula_type() == BeliefFormulaType::C_FORMULA && depth == 0)
        || (to_explore.get_formula_type() == BeliefFormulaType::BELIEF_FORMULA && depth > 0)) {
        for (const auto& agent : agents) {
            if ((to_explore.get_agent() != agent) || (depth == 0)) {
                BeliefFormula new_subgoal;
                new_subgoal.set_formula_type(BeliefFormulaType::BELIEF_FORMULA);
                if (depth == 0) {
                    new_subgoal.set_bf1(to_explore.get_bf1());
                } else {
                    new_subgoal.set_bf1(to_explore);
                }
                new_subgoal.set_agent(agent);
                m_goals.push_back(new_subgoal);

                if (nesting > (depth + 1)) {
                    produce_subgoals(nesting, depth + 1, new_subgoal, agents);
                }
            }
        }
    }
}

void HeuristicsManager::set_used_h(Heuristics used_h) noexcept
{
    m_used_heuristics = used_h;
}

[[nodiscard]] Heuristics HeuristicsManager::get_used_h() const noexcept
{
    return m_used_heuristics;
}

[[nodiscard]] const FormulaeList& HeuristicsManager::get_goals() const noexcept
{
    return m_goals;
}

void HeuristicsManager::set_goals(const FormulaeList& to_set)
{
    m_goals = to_set;
}

bool HeuristicsManager::operator=(const HeuristicsManager& to_copy)
{
    set_used_h(to_copy.get_used_h());
    set_goals(to_copy.get_goals());
    return true;
}
