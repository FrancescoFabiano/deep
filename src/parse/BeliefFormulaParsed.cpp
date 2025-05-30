/**
* \file BeliefFormulaParsed.cpp
 * \brief Implementation of the \ref BeliefFormulaParsed class.
 *
 * \author Francesco Fabiano.
 * \date May 2025
 */
#include "BeliefFormulaParsed.h"

void BeliefFormulaParsed::set_string_fluent_formula(const StringSetsSet & to_set)
{
    m_string_fluent_formula = to_set;
}

void BeliefFormulaParsed::set_string_agent(const std::string & to_set)
{
    m_string_agent = to_set;
}

void BeliefFormulaParsed::set_string_group_agents(const StringsSet & to_set)
{
    m_string_group_agents = to_set;
}

void BeliefFormulaParsed::set_bf1(const BeliefFormulaParsed & to_set)
{
    m_bf1 = std::make_unique<BeliefFormulaParsed>(to_set);
}

void BeliefFormulaParsed::set_bf2(const BeliefFormulaParsed & to_set)
{
    m_bf2 = std::make_unique<BeliefFormulaParsed>(to_set);
}

void BeliefFormulaParsed::set_formula_type(const BeliefFormulaType to_set)
{
    m_formula_type = to_set;
}

void BeliefFormulaParsed::set_operator(const BeliefFormulaOperator to_set)
{
    m_operator = to_set;
}

void BeliefFormulaParsed::set_from_ff(const StringSetsSet & to_build)
{
    set_formula_type(BeliefFormulaType::FLUENT_FORMULA);
    set_string_fluent_formula(to_build);
}

BeliefFormulaType BeliefFormulaParsed::get_formula_type() const noexcept
{
    return m_formula_type;
}

const StringSetsSet & BeliefFormulaParsed::get_string_fluent_formula() const noexcept
{
    return m_string_fluent_formula;
}

const std::string & BeliefFormulaParsed::get_string_agent() const noexcept
{
    return m_string_agent;
}

const StringsSet & BeliefFormulaParsed::get_string_group_agents() const noexcept
{
    return m_string_group_agents;
}

const BeliefFormulaParsed & BeliefFormulaParsed::get_bf1() const
{
    return *m_bf1;
}

const BeliefFormulaParsed & BeliefFormulaParsed::get_bf2() const
{
    return *m_bf2;
}

BeliefFormulaOperator BeliefFormulaParsed::get_operator() const noexcept
{
    return m_operator;
}

const StringsSet & BeliefFormulaParsed::get_group_agents() const noexcept
{
    return m_string_group_agents;
}

BeliefFormulaParsed& BeliefFormulaParsed::operator=(const BeliefFormulaParsed & to_copy) // NOLINT(*-no-recursion, *-unhandled-self-assignment)
{
    set_formula_type(to_copy.get_formula_type());
    switch (m_formula_type) {
    case BeliefFormulaType::FLUENT_FORMULA:
            set_from_ff(to_copy.get_string_fluent_formula());
        break;
    case BeliefFormulaType::BELIEF_FORMULA:
        set_string_agent(to_copy.get_string_agent());
        set_bf1(to_copy.get_bf1());
        break;
    case BeliefFormulaType::PROPOSITIONAL_FORMULA:
        set_operator(to_copy.get_operator());
        set_bf1(to_copy.get_bf1());
        switch (m_operator) {
        case BeliefFormulaOperator::BF_AND:
        case BeliefFormulaOperator::BF_OR:
            set_bf2(to_copy.get_bf2());
            break;
        case BeliefFormulaOperator::BF_NOT:
            break;
        case BeliefFormulaOperator::BF_INPAREN:
            //This CORRECTLY disrupt the self assignment by removing extra parenthesis that are not semantically necessary
            *this = to_copy.get_bf1();
            break;
        case BeliefFormulaOperator::BF_FAIL:
            //This CORRECTLY disrupt the self assignment if the BF is not set properly
        default:
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::BeliefFormulaOperatorUnset,
                "Error in copying a BeliefFormula."
            );
            break;
        }
        break;
    case BeliefFormulaType::E_FORMULA:
    case BeliefFormulaType::C_FORMULA:
            set_string_group_agents(to_copy.get_group_agents());
        set_bf1(to_copy.get_bf1());
        break;
    case BeliefFormulaType::BF_EMPTY:
        break;
    case BeliefFormulaType::BF_TYPE_FAIL:
        //This CORRECTLY disrupt the self assignment if the BF is not set properly
    default:
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::BeliefFormulaTypeUnset,
            "Error in copying a BeliefFormula."
        );
        break;
    }
    return *this;
}

BeliefFormulaParsed::BeliefFormulaParsed(const BeliefFormulaParsed & to_copy)
{
    (*this) = to_copy;
}