#include "KripkeEntailmentHelper.h"

#include "BeliefFormula.h"
#include "KripkeReachabilityHelper.h"
#include "KripkeState.h"
#include "KripkeWorld.h"
#include "utilities/ExitHandler.h"

/**
 * \file KripkeEntailmentHelper.cpp
 * \brief Implementation of KripkeEntailmentHelper.h
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 2025
 */

bool KripkeEntailmentHelper::entails(const Fluent &to_check,
                                     const KripkeWorld &world) {
  return world.get_fluent_set().contains(to_check);
}

bool KripkeEntailmentHelper::entails(const FluentsSet &to_check,
                                     const KripkeWorld &world) {
  // Maybe just set to True (It was like that in EFP)
  if (to_check.empty()) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::KripkeWorldEntailmentError,
        "Attempted to check entailment of an empty FluentFormula in "
        "KripkeEntailmentHelper::entails(FluentsSet,KripkeWorld).\n");
  }
  // Returns true if the formula is entailed in all reachable worlds.
  return std::ranges::all_of(
      to_check, [&](const Fluent &fluent) { return entails(fluent, world); });
}

bool KripkeEntailmentHelper::entails(const FluentFormula &to_check,
                                     const KripkeWorld &world) {
  // Maybe just set to True (It was like that in EFP)
  if (to_check.empty()) {
    return true;
    // ExitHandler::exit_with_message(
    //     ExitHandler::ExitCode::KripkeWorldEntailmentError,
    //     "Attempted to check entailment of an empty FluentFormula in "
    //     "KripkeEntailmentHelper::entails(FluentFormula, world).\n");
  }
  // Returns true if the formula is entailed in at least one reachable world.
  return std::ranges::any_of(to_check, [&](const FluentsSet &fluentSet) {
    return entails(fluentSet, world);
  });
}

bool KripkeEntailmentHelper::entails(const Fluent &to_check,
                                     const KripkeWorldPointer &world) {
  if (!world.get_ptr()) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::KripkeWorldPointerNullError,
        "Null KripkeWorldPointer in "
        "KripkeEntailmentHelper::entails(Fluent, KripkeWorldPointer).");
  }
  return entails(to_check, *world.get_ptr());
}

bool KripkeEntailmentHelper::entails(const FluentsSet &to_check,
                                     const KripkeWorldPointer &world) {
  if (!world.get_ptr()) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::KripkeWorldPointerNullError,
        "Null KripkeWorldPointer in "
        "KripkeEntailmentHelper::entails(FluentsSet, KripkeWorldPointer).");
  }
  return entails(to_check, *world.get_ptr());
}

bool KripkeEntailmentHelper::entails(const FluentFormula &to_check,
                                     const KripkeWorldPointer &world) {
  if (!world.get_ptr()) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::KripkeWorldPointerNullError,
        "Null KripkeWorldPointer in "
        "KripkeEntailmentHelper::entails(FluentFormula, KripkeWorldPointer).");
  }
  return entails(to_check, *world.get_ptr());
}

bool KripkeEntailmentHelper::entails(const BeliefFormula &to_check,
                                     const KripkeWorldPointersSet &reachable,
                                     const KripkeState &kstate) {
  // Returns true if the formula is entailed in all reachable worlds.
  return std::ranges::all_of(reachable, [&](const auto &world) {
    return entails(to_check, world, kstate);
  });
}

bool KripkeEntailmentHelper::entails(const BeliefFormula &to_check,
                                     const KripkeState &kstate) {
  return entails(to_check, kstate.get_pointed(), kstate);
}

/**
 * \brief Check if a BeliefFormula is entailed in a given Kripke world pointer,
 * using a KripkeState.
 */
bool KripkeEntailmentHelper::entails(const BeliefFormula &to_check,
                                     const KripkeWorldPointer &world,
                                     const KripkeState &kstate) {
  KripkeWorldPointersSet D_reachable;
  switch (to_check.get_formula_type()) {
  case BeliefFormulaType::FLUENT_FORMULA:
    return entails(to_check.get_fluent_formula(), world);

  case BeliefFormulaType::BELIEF_FORMULA:
    return entails(to_check.get_bf1(),
                   KripkeReachabilityHelper::get_B_reachable_worlds(
                       to_check.get_agent(), world, kstate),
                   kstate);

  case BeliefFormulaType::PROPOSITIONAL_FORMULA:
    switch (to_check.get_operator()) {
    case BeliefFormulaOperator::BF_NOT:
      return !entails(to_check.get_bf1(), world, kstate);
    case BeliefFormulaOperator::BF_OR:
      return entails(to_check.get_bf1(), world, kstate) ||
             entails(to_check.get_bf2(), world, kstate);
    case BeliefFormulaOperator::BF_AND:
      return entails(to_check.get_bf1(), world, kstate) &&
             entails(to_check.get_bf2(), world, kstate);
    case BeliefFormulaOperator::BF_FAIL:
    default:
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::BeliefFormulaOperatorUnset,
          "Invalid operator in propositional formula during "
          "entailment.");
    }
    break;

  case BeliefFormulaType::E_FORMULA:
    return entails(to_check.get_bf1(),
                   KripkeReachabilityHelper::get_E_reachable_worlds(
                       to_check.get_group_agents(), world, kstate),
                   kstate);

  case BeliefFormulaType::C_FORMULA:
    return entails(to_check.get_bf1(),
                   KripkeReachabilityHelper::get_C_reachable_worlds(
                       to_check.get_group_agents(), world, kstate),
                   kstate);

  case BeliefFormulaType::BF_EMPTY:
    return true;

  case BeliefFormulaType::BF_TYPE_FAIL:
  default:
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::BeliefFormulaTypeUnset,
        "Invalid formula type in BeliefFormula during entailment.");
  }

  return false;
}

bool KripkeEntailmentHelper::entails(const FormulaeList &to_check,
                                     const KripkeState &kstate) {
  return std::ranges::all_of(to_check, [&](const auto &belief_formula) {
    return entails(belief_formula, kstate);
  });
}

bool KripkeEntailmentHelper::check_properties(const AgentsSet &fully,
                                              const AgentsSet &partially,
                                              const FluentFormula &effects,
                                              const KripkeState &updated) {
  if (!fully.empty()) {
    BeliefFormula effects_formula;
    effects_formula.set_formula_type(BeliefFormulaType::FLUENT_FORMULA);
    effects_formula.set_fluent_formula(effects);

    BeliefFormula property1;
    property1.set_group_agents(fully);
    property1.set_formula_type(BeliefFormulaType::C_FORMULA);
    property1.set_bf1(effects_formula);

    if (!entails(property1, updated)) {
      std::cerr << "\nDEBUG: First property not respected";
      return false;
    }

    if (!partially.empty()) {
      BeliefFormula inner_nested2, nested2, disjunction, property2;
      inner_nested2.set_group_agents(fully);
      inner_nested2.set_formula_type(BeliefFormulaType::C_FORMULA);
      inner_nested2.set_bf1(effects_formula);

      nested2.set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);
      nested2.set_operator(BeliefFormulaOperator::BF_NOT);
      nested2.set_bf1(inner_nested2);

      disjunction.set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);
      disjunction.set_operator(BeliefFormulaOperator::BF_OR);
      disjunction.set_bf1(property1);
      disjunction.set_bf2(nested2);

      property2.set_group_agents(partially);
      property2.set_formula_type(BeliefFormulaType::C_FORMULA);
      property2.set_bf1(disjunction);

      BeliefFormula property3;
      property3.set_group_agents(fully);
      property3.set_formula_type(BeliefFormulaType::C_FORMULA);
      property3.set_bf1(property2);

      if (!entails(property2, updated)) {
        std::cerr << "\nDEBUG: Second property not respected in the formula: ";
        return false;
      }
      if (!entails(property3, updated)) {
        std::cerr << "\nDEBUG: Third property not respected in the formula: ";
        return false;
      }
    }
  }
  return true;
}
