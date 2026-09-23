#include "PlankFormulaConverter.h"

#include <type_traits>
#include <utility>

#include "utilities/ExitHandler.h"

PlankFormulaConverter::PlankFormulaConverter(const std::vector<Fluent> &fluents,
                                             const std::vector<Agent> &agents)
    : m_fluents(fluents), m_agents(agents) {}

BeliefFormula
PlankFormulaConverter::convert(const plank::del::formula_ptr &formula) const {

  return std::visit(
      [this]<typename T0>(const T0 &typed_formula) -> BeliefFormula {
        using FormulaPtr = std::decay_t<T0>;

        // Logical constants
        if constexpr (std::is_same_v<FormulaPtr,
                                     plank::del::true_formula_ptr>) {

          BeliefFormula result;
          result.set_formula_type(BeliefFormulaType::TRUE_FORMULA);

          return result;

        } else if constexpr (std::is_same_v<FormulaPtr,
                                            plank::del::false_formula_ptr>) {

          BeliefFormula result;
          result.set_formula_type(BeliefFormulaType::FALSE_FORMULA);

          return result;

          // Atomic formula
        } else if constexpr (std::is_same_v<FormulaPtr,
                                            plank::del::atom_formula_ptr>) {

          return convert_atom(typed_formula);

          // Propositional formulae
        } else if constexpr (std::is_same_v<FormulaPtr,
                                            plank::del::not_formula_ptr>) {

          return convert_not(typed_formula);

        } else if constexpr (std::is_same_v<FormulaPtr,
                                            plank::del::and_formula_ptr>) {

          return convert_and(typed_formula);

        } else if constexpr (std::is_same_v<FormulaPtr,
                                            plank::del::or_formula_ptr>) {

          return convert_or(typed_formula);

        } else if constexpr (std::is_same_v<FormulaPtr,
                                            plank::del::imply_formula_ptr>) {

          return convert_imply(typed_formula);

          // Epistemic formulae
        } else if constexpr (std::is_same_v<FormulaPtr,
                                            plank::del::box_formula_ptr>) {

          return convert_box(typed_formula);

        } else if constexpr (std::is_same_v<FormulaPtr,
                                            plank::del::c_box_formula_ptr>) {

          return convert_common(typed_formula);

        } else if constexpr (std::is_same_v<FormulaPtr,
                                            plank::del::diamond_formula_ptr>) {

          return convert_diamond(typed_formula);

        } else if constexpr (std::is_same_v<FormulaPtr,
                                            plank::del::kw_box_formula_ptr>) {

          return convert_kw_box(typed_formula);

        } else if constexpr (std::is_same_v<
                                 FormulaPtr,
                                 plank::del::kw_diamond_formula_ptr>) {

          return convert_kw_diamond(typed_formula);

        } else if constexpr (std::is_same_v<
                                 FormulaPtr,
                                 plank::del::c_diamond_formula_ptr>) {

          return convert_common_diamond(typed_formula);

        } else {
          ExitHandler::exit_with_message(
              ExitHandler::ExitCode::DomainBuildError,
              "PlankFormulaConverter: unknown formula type.");
        }

        return {};
      },
      formula);
}

BeliefFormula PlankFormulaConverter::convert_atom(
    const plank::del::atom_formula_ptr &formula) const {

  const auto atom_id = formula->get_atom();

  if (atom_id >= m_fluents.size()) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::DomainBuildError,
        "Invalid plank atom id while converting formula.");
  }

  BeliefFormula result;

  result.set_formula_type(BeliefFormulaType::FLUENT_FORMULA);

  result.set_fluent_formula_from_fluent(m_fluents[atom_id]);

  return result;
}

BeliefFormula PlankFormulaConverter::convert_not(
    const plank::del::not_formula_ptr &formula) const {

  BeliefFormula result;

  result.set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);

  result.set_operator(BeliefFormulaOperator::BF_NOT);

  result.set_bf1(convert(formula->get_formula()));

  return result;
}

BeliefFormula
PlankFormulaConverter::convert_nary(const plank::del::formula_deque &formulas,
                                    const BeliefFormulaOperator op) const {

  if (formulas.empty()) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::DomainBuildError,
        "Cannot convert an empty EPDDL AND/OR formula.");
  }

  auto it = formulas.begin();

  BeliefFormula result = convert(*it);

  ++it;

  for (; it != formulas.end(); ++it) {
    BeliefFormula combined;

    combined.set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);

    combined.set_operator(op);

    combined.set_bf1(result);
    combined.set_bf2(convert(*it));

    result = std::move(combined);
  }

  return result;
}

BeliefFormula PlankFormulaConverter::convert_and(
    const plank::del::and_formula_ptr &formula) const {

  return convert_nary(formula->get_formulas(), BeliefFormulaOperator::BF_AND);
}

BeliefFormula PlankFormulaConverter::convert_or(
    const plank::del::or_formula_ptr &formula) const {

  return convert_nary(formula->get_formulas(), BeliefFormulaOperator::BF_OR);
}

AgentsSet PlankFormulaConverter::convert_agents(
    const plank::del::agent_set &plank_agents) const {

  AgentsSet result;

  for (const plank::del::agent plank_agent : plank_agents) {

    if (plank_agent >= m_agents.size()) {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::DomainBuildError,
          "Invalid plank agent id while converting formula.");
    }

    result.insert(m_agents[plank_agent]);
  }

  return result;
}

BeliefFormula PlankFormulaConverter::convert_box(
    const plank::del::box_formula_ptr &formula) const {

  const AgentsSet agents = convert_agents(formula->get_mod_index());

  if (agents.empty()) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::DomainBuildError,
        "Cannot convert a box formula with an empty agent set.");
  }

  BeliefFormula result;

  if (agents.size() == 1) {
    result.set_formula_type(BeliefFormulaType::BELIEF_FORMULA);

    result.set_agent(*agents.begin());

  } else {
    result.set_formula_type(BeliefFormulaType::E_FORMULA);

    result.set_group_agents(agents);
  }

  result.set_bf1(convert(formula->get_formula()));

  return result;
}

BeliefFormula PlankFormulaConverter::convert_common(
    const plank::del::c_box_formula_ptr &formula) const {

  const AgentsSet agents = convert_agents(formula->get_mod_index());

  if (agents.empty()) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::DomainBuildError,
        "Cannot convert common knowledge with an empty agent set.");
  }

  BeliefFormula result;

  result.set_formula_type(BeliefFormulaType::C_FORMULA);

  result.set_group_agents(agents);

  result.set_bf1(convert(formula->get_formula()));

  return result;
}

BeliefFormula PlankFormulaConverter::convert_imply(
    const plank::del::imply_formula_ptr &formula) const {

  // phi -> psi  ===  !phi OR psi

  BeliefFormula negated_antecedent;
  negated_antecedent.set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);
  negated_antecedent.set_operator(BeliefFormulaOperator::BF_NOT);

  negated_antecedent.set_bf1(convert(formula->get_first_formula()));

  BeliefFormula result;
  result.set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);
  result.set_operator(BeliefFormulaOperator::BF_OR);

  result.set_bf1(negated_antecedent);
  result.set_bf2(convert(formula->get_second_formula()));

  return result;
}

BeliefFormula PlankFormulaConverter::make_not(const BeliefFormula &formula) {

  BeliefFormula result;

  result.set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);

  result.set_operator(BeliefFormulaOperator::BF_NOT);

  result.set_bf1(formula);

  return result;
}

BeliefFormula PlankFormulaConverter::make_or(const BeliefFormula &left,
                                             const BeliefFormula &right) {

  BeliefFormula result;

  result.set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);

  result.set_operator(BeliefFormulaOperator::BF_OR);

  result.set_bf1(left);
  result.set_bf2(right);

  return result;
}

BeliefFormula PlankFormulaConverter::convert_diamond(
    const plank::del::diamond_formula_ptr &formula) const {

  const AgentsSet agents = convert_agents(formula->get_mod_index());

  if (agents.empty()) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::DomainBuildError,
        "Cannot convert a diamond formula with an empty agent set.");
  }

  const BeliefFormula inner = convert(formula->get_formula());

  const BeliefFormula negated_inner = make_not(inner);

  BeliefFormula box;

  if (agents.size() == 1) {
    box.set_formula_type(BeliefFormulaType::BELIEF_FORMULA);

    box.set_agent(*agents.begin());
  } else {
    box.set_formula_type(BeliefFormulaType::E_FORMULA);

    box.set_group_agents(agents);
  }

  box.set_bf1(negated_inner);

  // <> phi === !(box !phi)
  return make_not(box);
}

BeliefFormula PlankFormulaConverter::convert_common_diamond(
    const plank::del::c_diamond_formula_ptr &formula) const {

  const AgentsSet agents = convert_agents(formula->get_mod_index());

  if (agents.empty()) {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::DomainBuildError,
                                   "Cannot convert a common-knowledge diamond "
                                   "with an empty agent set.");
  }

  const BeliefFormula inner = convert(formula->get_formula());

  const BeliefFormula negated_inner = make_not(inner);

  BeliefFormula common;

  common.set_formula_type(BeliefFormulaType::C_FORMULA);

  common.set_group_agents(agents);
  common.set_bf1(negated_inner);

  // <C> phi === !C(!phi)
  return make_not(common);
}

BeliefFormula PlankFormulaConverter::convert_kw_box(
    const plank::del::kw_box_formula_ptr &formula) const {

  const AgentsSet agents = convert_agents(formula->get_mod_index());

  if (agents.empty()) {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::DomainBuildError,
                                   "Cannot convert a knowing-whether formula "
                                   "with an empty agent set.");
  }

  const BeliefFormula inner = convert(formula->get_formula());

  const BeliefFormula negated_inner = make_not(inner);

  BeliefFormula knows_true;
  BeliefFormula knows_false;

  if (agents.size() == 1) {
    const Agent &agent = *agents.begin();

    knows_true.set_formula_type(BeliefFormulaType::BELIEF_FORMULA);

    knows_true.set_agent(agent);
    knows_true.set_bf1(inner);

    knows_false.set_formula_type(BeliefFormulaType::BELIEF_FORMULA);

    knows_false.set_agent(agent);
    knows_false.set_bf1(negated_inner);

  } else {
    knows_true.set_formula_type(BeliefFormulaType::E_FORMULA);

    knows_true.set_group_agents(agents);
    knows_true.set_bf1(inner);

    knows_false.set_formula_type(BeliefFormulaType::E_FORMULA);

    knows_false.set_group_agents(agents);
    knows_false.set_bf1(negated_inner);
  }

  return make_or(knows_true, knows_false);
}

BeliefFormula PlankFormulaConverter::convert_kw_diamond(
    const plank::del::kw_diamond_formula_ptr &formula) const {

  const AgentsSet agents = convert_agents(formula->get_mod_index());

  if (agents.empty()) {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::DomainBuildError,
                                   "Cannot convert a knowing-whether diamond "
                                   "with an empty agent set.");
  }

  const BeliefFormula inner = convert(formula->get_formula());

  /*
   * Build Kw(!phi):
   *
   *   box(!phi) OR box(!!phi)
   *
   * and negate the result.
   */

  const BeliefFormula negated_inner = make_not(inner);

  const BeliefFormula double_negated_inner = make_not(negated_inner);

  BeliefFormula first;
  BeliefFormula second;

  if (agents.size() == 1) {
    const Agent &agent = *agents.begin();

    first.set_formula_type(BeliefFormulaType::BELIEF_FORMULA);
    first.set_agent(agent);
    first.set_bf1(negated_inner);

    second.set_formula_type(BeliefFormulaType::BELIEF_FORMULA);
    second.set_agent(agent);
    second.set_bf1(double_negated_inner);

  } else {
    first.set_formula_type(BeliefFormulaType::E_FORMULA);
    first.set_group_agents(agents);
    first.set_bf1(negated_inner);

    second.set_formula_type(BeliefFormulaType::E_FORMULA);
    second.set_group_agents(agents);
    second.set_bf1(double_negated_inner);
  }

  return make_not(make_or(first, second));
}