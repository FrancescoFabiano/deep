#include "Domain.h"
#include "HeuristicsManager.h"
#include <ranges>

#ifdef USE_NEURALNETS
#include "neuralnets/GraphNN.h"
#endif

template <StateRepresentation StateRepr>
HeuristicsManager<StateRepr>::HeuristicsManager(
    const State<StateRepr> &initial_state) {
  set_used_h(Configuration::get_instance().get_heuristic_opt());
  m_goals = Domain::get_instance().get_goal_description();
  switch (m_used_heuristics) {
  case Heuristics::L_PG:
  case Heuristics::S_PG:
    expand_goals();
    break;
  case Heuristics::C_PG: {
    expand_goals();
    if (const PlanningGraph pg(m_goals, initial_state); pg.is_satisfiable()) {
      m_pg_max_score = 0;
      m_fluents_score = pg.get_f_scores();
      m_bf_score = pg.get_bf_scores();
      for (const auto &score_f : m_fluents_score | std::views::values) {
        if (score_f > 0) {
          m_pg_max_score +=
              score_f; // Accumulate positive scores for normalization.
        }
      }
      for (const auto &score_bf : m_bf_score | std::views::values) {
        if (score_bf > 0) {
          m_pg_max_score +=
              score_bf; // Accumulate positive scores for normalization.
        }
      }
    } else {
      m_pg_goal_not_found = true;
    }
    break;
  }
  case Heuristics::SUBGOALS:
    expand_goals();
    SatisfiedGoals::get_instance().set(m_goals);
    break;
  case Heuristics::GNN:
#ifdef USE_NEURALNETS
    GraphNN<StateRepr>::create_instance();
    break;
#else
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::HeuristicsBadDeclaration,
        "GNN heuristics selected, but neural network support (torch) is not "
        "enabled or linked. Please recompile with the nn option.");
    break;
#endif
  default:
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::HeuristicsBadDeclaration,
        "Wrong Heuristic Selection in HeuristicsManager. Please check the "
        "heuristic type.");
    break;
  }
}

template <StateRepresentation StateRepr>
int HeuristicsManager<StateRepr>::get_heuristic_value(
    State<StateRepr> &eState) {
  switch (m_used_heuristics) {
  case Heuristics::L_PG: {
    const PlanningGraph pg(m_goals, eState);
    return (pg.is_satisfiable() ? pg.get_length() : -1);
    break;
  }
  case Heuristics::S_PG: {
    const PlanningGraph pg(m_goals, eState);
    return (pg.is_satisfiable() ? pg.get_sum() : -1);
    break;
  }
  case Heuristics::C_PG: {
    int h_value = 0;

    if (m_pg_goal_not_found) {
      h_value = -1; // Goal not reachable, set heuristic to -1.
    } else {
      for (const auto &[fluent, score] : m_fluents_score) {
        if (eState.entails(fluent) && score > 0)
          h_value += score;
      }
      for (const auto &[belief, score] : m_bf_score) {
        if (eState.entails(belief) && score > 0)
          h_value += score;
      }
      h_value = static_cast<int>(100 - ((static_cast<float>(h_value) /
                                           static_cast<float>(m_pg_max_score)) *
                                          100)); // Invert: 0 is 100%, 100 is 0%
    }

    return h_value;
    break;
  }
  case Heuristics::SUBGOALS: {
    return SatisfiedGoals::get_instance().get_unsatisfied_goals(eState);
    break;
  }
  case Heuristics::GNN: {
#ifdef USE_NEURALNETS
    return GraphNN<StateRepr>::get_instance().get_score(eState);
    break;
#else
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::HeuristicsBadDeclaration,
        "GNN heuristics selected, but neural network support (torch) is not "
        "enabled or linked. Please recompile with the nn option.");
    break;
#endif
  }
  default: {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::HeuristicsBadDeclaration,
        "Wrong Heuristic Selection in HeuristicsManager. Please check the "
        "heuristic type.");
    break;
  }
  }
  return -1; // Default return value, should not be reached.
}

template <StateRepresentation StateRepr>
void HeuristicsManager<StateRepr>::set_heuristic_value(
    State<StateRepr> &eState) {
  eState.set_heuristic_value(compute_heuristic_value(eState));
}

template <StateRepresentation StateRepr>
void HeuristicsManager<StateRepr>::expand_goals(const unsigned short nesting) {
  const FormulaeList original_goal = m_goals;
  for (const auto &formula : original_goal) {
    produce_subgoals(nesting, 0, formula, formula.get_group_agents());
  }
}

template <StateRepresentation StateRepr>
void HeuristicsManager<StateRepr>::produce_subgoals(
    const unsigned short nesting, const unsigned short depth,
    const BeliefFormula &to_explore, const AgentsSet &agents) {
  if ((to_explore.get_formula_type() == BeliefFormulaType::C_FORMULA &&
       depth == 0) ||
      (to_explore.get_formula_type() == BeliefFormulaType::BELIEF_FORMULA &&
       depth > 0)) {
    for (const auto &agent : agents) {
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

template <StateRepresentation StateRepr>
void HeuristicsManager<StateRepr>::set_used_h(
    const Heuristics used_h) noexcept {
  m_used_heuristics = used_h;
}

template <StateRepresentation StateRepr>
Heuristics HeuristicsManager<StateRepr>::get_used_h() const noexcept {
  return m_used_heuristics;
}

template <StateRepresentation StateRepr>
std::string HeuristicsManager<StateRepr>::get_used_h_name() const noexcept {
  switch (m_used_heuristics) {
  case Heuristics::L_PG:
    return "L-PG";
  case Heuristics::S_PG:
    return "S-PG";
  case Heuristics::C_PG:
    return "C-PG";
  case Heuristics::SUBGOALS:
    return "SubGoals";
  case Heuristics::GNN:
    return "GNN";
  default: {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::HeuristicsBadDeclaration,
        "Wrong Heuristic Selection in HeuristicsManager. Please check the "
        "heuristic type.");
    // This line will never be reached, but added to avoid compiler warning.
    std::exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));
  }
  }
}

template <StateRepresentation StateRepr>
const FormulaeList &HeuristicsManager<StateRepr>::get_goals() const noexcept {
  return m_goals;
}

template <StateRepresentation StateRepr>
void HeuristicsManager<StateRepr>::set_goals(const FormulaeList &to_set) {
  m_goals = to_set;
}
