#include "HeuristicsManager.h"
#include "Domain.h"

void HeuristicsManager::expand_goals(const unsigned short nesting)
{
    FormulaeList original_goal = m_goals;
    for (const auto& formula : original_goal)
    {
        produce_subgoals(nesting, 0, formula, formula.get_group_agents());
    }
}

void HeuristicsManager::produce_subgoals(const unsigned short nesting, const unsigned short depth,
                                         const BeliefFormula& to_explore, const AgentsSet& agents)
{
    if ((to_explore.get_formula_type() == BeliefFormulaType::C_FORMULA && depth == 0)
        || (to_explore.get_formula_type() == BeliefFormulaType::BELIEF_FORMULA && depth > 0))
    {
        for (const auto& agent : agents)
        {
            if ((to_explore.get_agent() != agent) || (depth == 0))
            {
                BeliefFormula new_subgoal;
                new_subgoal.set_formula_type(BeliefFormulaType::BELIEF_FORMULA);
                if (depth == 0)
                {
                    new_subgoal.set_bf1(to_explore.get_bf1());
                }
                else
                {
                    new_subgoal.set_bf1(to_explore);
                }
                new_subgoal.set_agent(agent);
                m_goals.push_back(new_subgoal);

                if (nesting > (depth + 1))
                {
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
