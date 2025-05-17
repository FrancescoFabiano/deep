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
    m_bf1 = std::make_shared<BeliefFormulaParsed>(to_set);
}

void BeliefFormulaParsed::set_bf2(const BeliefFormulaParsed & to_set)
{
    m_bf2 = std::make_shared<BeliefFormulaParsed>(to_set);
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

[[nodiscard]] BeliefFormulaType BeliefFormulaParsed::get_formula_type() const noexcept
{
    return m_formula_type;
}

[[nodiscard]] const StringSetsSet & BeliefFormulaParsed::get_string_fluent_formula() const noexcept
{
    return m_string_fluent_formula;
}

[[nodiscard]] const std::string & BeliefFormulaParsed::get_string_agent() const noexcept
{
    return m_string_agent;
}

[[nodiscard]] const StringsSet & BeliefFormulaParsed::get_string_group_agents() const noexcept
{
    return m_string_group_agents;
}

[[nodiscard]] const BeliefFormulaParsed & BeliefFormulaParsed::get_bf1() const
{
    return *m_bf1;
}

[[nodiscard]] const BeliefFormulaParsed & BeliefFormulaParsed::get_bf2() const
{
    return *m_bf2;
}

[[nodiscard]] BeliefFormulaOperator BeliefFormulaParsed::get_operator() const noexcept
{
    return m_operator;
}

[[nodiscard]] const StringsSet & BeliefFormulaParsed::get_group_agents() const noexcept
{
    return m_string_group_agents;
}
