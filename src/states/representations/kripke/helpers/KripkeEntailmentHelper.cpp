//
// Created by franc on 5/17/2025.
//

#include "KripkeEntailmentHelper.h"

#include "BeliefFormula.h"
#include "KripkeReachabilityHelper.h"
#include "KripkeState.h"
#include "utilities/ExitHandler.h"
#include "KripkeWorld.h"

/**
 * \file KripkeEntailmentHelper.cpp
 * \brief Implementation of KripkeEntailmentHelper.h
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 2025
 */

/// \name Entailment for KripkeWorld
///@{
bool KripkeEntailmentHelper::entails(const Fluent &to_check, const KripkeWorld& world) {
    return world.get_fluent_set().contains(to_check);
}

bool KripkeEntailmentHelper::entails(const FluentsSet& to_check, const KripkeWorld& world){
    //Maybe just set to True (It was like this in EFP)
    if (to_check.empty()) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::KripkeWorldEntailmentError,
            "Error: Attempted to check entailment of an empty FluentFormula in KripkeEntailmentHelper::entails().\n"
        );
    }
    // Returns true if the formula is entailed in all reachable worlds.
    return std::ranges::all_of(to_check, [&](const Fluent& fluent) {
        return entails(fluent, world);
    });
}

bool KripkeEntailmentHelper::entails(const FluentFormula& to_check, const KripkeWorld& world) {
    //Maybe just set to True (It was like this in EFP)
    if (to_check.empty()) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::KripkeWorldEntailmentError,
            "Error: Attempted to check entailment of an empty FluentFormula in KripkeEntailmentHelper::entails().\n"
        );
    }
    // Returns true if the formula is entailed in all reachable worlds.
    return std::ranges::all_of(to_check, [&](const FluentsSet& fluentSet) {
        return entails(fluentSet, world);
    });
}
///@}

/// \name Entailment for KripkeWorldPointer
///@{
bool KripkeEntailmentHelper::entails(const Fluent& to_check, const KripkeWorldPointer& pworld)
{
    if (!pworld.get_ptr()) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::KripkeWorldPointerNullError,
            "Error: Null KripkeWorldPointer in KripkeEntailmentHelper::entails(Fluent, KripkeWorldPointer)."
        );
    }
    return entails(to_check,*pworld.get_ptr());
}

bool KripkeEntailmentHelper::entails(const FluentsSet& to_check, const KripkeWorldPointer& pworld)
{
    if (!pworld.get_ptr()) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::KripkeWorldPointerNullError,
            "Error: Null KripkeWorldPointer in KripkeEntailmentHelper::entails(FluentsSet, KripkeWorldPointer)."
        );
    }
    return entails(to_check, *pworld.get_ptr());
}

bool KripkeEntailmentHelper::entails(const FluentFormula& to_check, const KripkeWorldPointer& pworld)
{
    if (!pworld.get_ptr()) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::KripkeWorldPointerNullError,
            "Error: Null KripkeWorldPointer in KripkeEntailmentHelper::entails(FluentFormula, KripkeWorldPointer)."
        );
    }
    return entails(to_check,*pworld.get_ptr());
}
///@}

/// \name Entailment KripkeState
///@{
bool KripkeEntailmentHelper::entails(const BeliefFormula& to_check, const KripkeWorldPointersSet& reachable, const KripkeState & kstate)
{
    // Returns true if the formula is entailed in all reachable worlds.
    return std::ranges::all_of(reachable, [&](const auto& world) {
        return entails(to_check, world, kstate);
    });
}


bool KripkeEntailmentHelper::entails(const BeliefFormula& to_check, const KripkeState& kstate) {
    return entails(to_check, kstate.get_pointed(), kstate);
}

/**
 * \brief Check if a BeliefFormula is entailed in a given Kripke world pointer, using a KripkeState.
 */
bool KripkeEntailmentHelper::entails(const BeliefFormula& to_check, const KripkeWorldPointer& world, const KripkeState& kstate) {
    KripkeWorldPointersSet D_reachable;
    switch (to_check.get_formula_type()) {

    case BeliefFormulaType::FLUENT_FORMULA:
        return entails(to_check.get_fluent_formula(), world);

    case BeliefFormulaType::BELIEF_FORMULA:
        return entails(to_check.get_bf1(), KripkeReachabilityHelper::get_B_reachable_worlds(to_check.get_agent(), world, kstate),kstate);

    case BeliefFormulaType::PROPOSITIONAL_FORMULA:
        switch (to_check.get_operator()) {
        case BeliefFormulaOperator::BF_NOT:
            return !entails(to_check.get_bf1(), world,kstate);
        case BeliefFormulaOperator::BF_OR:
            return entails(to_check.get_bf1(), world,kstate) || entails(to_check.get_bf2(), world,kstate);
        case BeliefFormulaOperator::BF_AND:
            return entails(to_check.get_bf1(), world,kstate) && entails(to_check.get_bf2(), world,kstate);
        case BeliefFormulaOperator::BF_FAIL:
        default:
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::BeliefFormulaOperatorUnset,
                "Error: Invalid operator in propositional formula during entailment."
            );
        }
        break;

    case BeliefFormulaType::E_FORMULA:
        return entails(to_check.get_bf1(), KripkeReachabilityHelper::get_E_reachable_worlds(to_check.get_group_agents(), world, kstate),kstate);

    case BeliefFormulaType::C_FORMULA:
        return entails(to_check.get_bf1(), KripkeReachabilityHelper::get_C_reachable_worlds(to_check.get_group_agents(), world, kstate),kstate);

    case BeliefFormulaType::BF_EMPTY:
        return true;

    case BeliefFormulaType::BF_TYPE_FAIL:
    default:
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::BeliefFormulaTypeUnset,
            "Error: Invalid formula type in BeliefFormula during entailment."
        );
    }

    return false;
}

bool KripkeEntailmentHelper::entails(const FormulaeList& to_check, const KripkeState & kstate)
{
    return std::ranges::all_of(to_check, [&](const auto& belief_formula) {
    return entails(belief_formula, kstate);
});
}
///@}
