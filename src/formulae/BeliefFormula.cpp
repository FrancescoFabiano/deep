/**
 * \file BeliefFormula.cpp
 * \brief Implementation of the \ref BeliefFormula class.
 *
 * \author Francesco Fabiano.
 * \date May 2025
 */
#include "BeliefFormula.h"
#include "domain/Domain.h"
#include "utilities/HelperPrint.h"
#include "utilities/ExitHandler.h"


void BeliefFormula::set_from_ff(const FluentFormula & to_build)
{
    set_formula_type(BeliefFormulaType::FLUENT_FORMULA);
    set_fluent_formula(to_build);
}

BeliefFormula::BeliefFormula(const BeliefFormulaParsed &to_ground) {
    const auto grounder = Domain::get_instance().get_grounder();
    set_formula_type(to_ground.get_formula_type());
    switch (m_formula_type) {
        case BeliefFormulaType::FLUENT_FORMULA:
            set_fluent_formula(grounder.ground_fluent(to_ground.get_string_fluent_formula()));
            break;
        case BeliefFormulaType::BELIEF_FORMULA:
            set_agent(grounder.ground_agent(to_ground.get_string_agent()));
            set_bf1(to_ground.get_bf1());
            break;
        case BeliefFormulaType::PROPOSITIONAL_FORMULA:
            set_operator(to_ground.get_operator());
            set_bf1(to_ground.get_bf1());
            switch (m_operator) {
                case BeliefFormulaOperator::BF_AND:
                case BeliefFormulaOperator::BF_OR:
                    set_bf2(to_ground.get_bf2());
                    break;
                case BeliefFormulaOperator::BF_NOT:
                    break;
                case BeliefFormulaOperator::BF_INPAREN:
                    (*this) = BeliefFormula(get_bf1());
                    break;
                case BeliefFormulaOperator::BF_FAIL:
                default:
                    ExitHandler::exit_with_message(
                        ExitHandler::ExitCode::BeliefFormulaOperatorUnset,
                        "Error in creating a BeliefFormula from a parsed one."
                    );
                    break;
            }
            break;
        case BeliefFormulaType::E_FORMULA:
        case BeliefFormulaType::C_FORMULA:
        case BeliefFormulaType::D_FORMULA:
            set_group_agents(grounder.ground_agent(to_ground.get_group_agents()));
            set_bf1(to_ground.get_bf1());
            break;
        case BeliefFormulaType::BF_EMPTY:
        case BeliefFormulaType::BF_TYPE_FAIL:
        default:
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::BeliefFormulaTypeUnset,
                "Error in creating a BeliefFormula from a parsed one."
            );
            break;
    }
}

BeliefFormula::BeliefFormula(const BeliefFormula & to_copy)
{
    (*this) = to_copy;
}

void BeliefFormula::set_formula_type(const BeliefFormulaType to_set)
{
    m_formula_type = to_set;
}

[[nodiscard]] BeliefFormulaType BeliefFormula::get_formula_type() const
{
    if (m_formula_type == BeliefFormulaType::BF_TYPE_FAIL) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::BeliefFormulaTypeUnset,
            "Error in reading a BeliefFormula (BeliefFormulaType not set properly)."
        );
    }
    return m_formula_type;
}

void BeliefFormula::set_fluent_formula(const FluentFormula & to_set)
{
    if (to_set.empty()) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::BeliefFormulaEmptyFluent,
            "Error in declaring a BeliefFormula: there must be at least one fluent in a formula."
        );
    }
    m_fluent_formula = to_set;
}

void BeliefFormula::set_fluent_formula_from_fluent(const Fluent& to_set)
{
    FluentsSet tmp;
    tmp.insert(to_set);
    m_fluent_formula.insert(tmp);
}

[[nodiscard]] const FluentFormula & BeliefFormula::get_fluent_formula() const
{
    if (m_fluent_formula.empty()) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::BeliefFormulaNotGrounded,
            "Error in reading a BeliefFormula, it must be grounded (FluentFormula not grounded)."
        );
    }
    return m_fluent_formula;
}

void BeliefFormula::set_agent(const Agent& to_set)
{
    m_agent = to_set;
}

[[nodiscard]] const Agent & BeliefFormula::get_agent() const
{
    return m_agent;
}

[[nodiscard]] const BeliefFormula & BeliefFormula::get_bf1() const
{
    if (!m_bf1) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::BeliefFormulaMissingNested,
            "Error in declaring a BeliefFormula: a nested belief formula has not been declared."
        );
    }
    return *m_bf1;
}

[[nodiscard]] const BeliefFormula & BeliefFormula::get_bf2() const
{
    if (!m_bf2) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::BeliefFormulaMissingNested,
            "Error in declaring a BeliefFormula: a second nested belief formula has not been declared."
        );
    }
    return *m_bf2;
}

void BeliefFormula::set_operator(const BeliefFormulaOperator to_set)
{
    m_operator = to_set;
}

bool BeliefFormula::is_bf2_null() const
{
    return m_bf2 == nullptr;
}

[[nodiscard]] BeliefFormulaOperator BeliefFormula::get_operator() const
{
    if (m_formula_type == BeliefFormulaType::BF_TYPE_FAIL) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::BeliefFormulaOperatorUnset,
            "Error in reading a BeliefFormula (BeliefFormulaOperator not set properly)."
        );
    }
    return m_operator;
}

void BeliefFormula::set_group_agents(const AgentsSet & to_set)
{
    if (to_set.empty()) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::BeliefFormulaEmptyAgentGroup,
            "Error in declaring a BeliefFormula: there must be at least one agent for group formulae."
        );
    }
    m_group_agents = to_set;
}

[[nodiscard]] const AgentsSet & BeliefFormula::get_group_agents() const
{
    return m_group_agents;
}

void BeliefFormula::set_bf1(const BeliefFormula & to_set)
{
    m_bf1 = std::make_shared<BeliefFormula>(to_set);
}

void BeliefFormula::set_bf2(const BeliefFormula & to_set)
{
    m_bf2 = std::make_shared<BeliefFormula>(to_set);
}

void BeliefFormula::set_bf1(const BeliefFormulaParsed & to_set)
{
    m_bf1 = std::make_shared<BeliefFormula>(to_set);
}

void BeliefFormula::set_bf2(const BeliefFormulaParsed & to_set)
{
    m_bf2 = std::make_shared<BeliefFormula>(to_set);
}

void BeliefFormula::print() const
{
    HelperPrint::get_instance().print_belief_formula(*this);
    std::cout << std::endl;
}

bool BeliefFormula::operator==(const BeliefFormula & to_compare) const
{
    if (m_formula_type != to_compare.m_formula_type) {
        return false;
    }

    switch (m_formula_type) {
    case BeliefFormulaType::FLUENT_FORMULA:
        return m_fluent_formula == to_compare.get_fluent_formula();

    case BeliefFormulaType::BELIEF_FORMULA:
        return m_agent == to_compare.get_agent() &&
               get_bf1() == to_compare.get_bf1();

    case BeliefFormulaType::PROPOSITIONAL_FORMULA:
        if (m_operator != to_compare.get_operator()) {
            return false;
        }
        switch (m_operator) {
        case BeliefFormulaOperator::BF_NOT:
                return get_bf1() == to_compare.get_bf1();
        case BeliefFormulaOperator::BF_AND:
        case BeliefFormulaOperator::BF_OR:
                // Commutative: (A op B) == (B op A)
                return (get_bf1() == to_compare.get_bf1() && get_bf2() == to_compare.get_bf2()) ||
                       (get_bf1() == to_compare.get_bf2() && get_bf2() == to_compare.get_bf1());
        default:
                return false;
        }

    case BeliefFormulaType::E_FORMULA:
    case BeliefFormulaType::C_FORMULA:
    case BeliefFormulaType::D_FORMULA:
        return m_group_agents == to_compare.get_group_agents() &&
               get_bf1() == to_compare.get_bf1();

    case BeliefFormulaType::BF_EMPTY:
        return true;

    default:
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::BeliefFormulaTypeUnset,
            "Unknown BeliefFormula type."
        );
        return false;
    }
}

BeliefFormula& BeliefFormula::operator=(const BeliefFormula & to_copy)
{
    set_formula_type(to_copy.get_formula_type());
    switch (m_formula_type) {
    case BeliefFormulaType::FLUENT_FORMULA:
            set_fluent_formula(to_copy.get_fluent_formula());
        break;
    case BeliefFormulaType::BELIEF_FORMULA:
        set_agent(to_copy.get_agent());
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
    case BeliefFormulaType::D_FORMULA:
            set_group_agents(to_copy.get_group_agents());
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

bool BeliefFormula::operator<(const BeliefFormula & to_compare) const
{
    // Compare formula types first
    if (get_formula_type() < to_compare.get_formula_type()) {
        return true;
    }
    if (get_formula_type() != to_compare.get_formula_type()) {
        return false;
    }

    switch (m_formula_type) {
    case BeliefFormulaType::FLUENT_FORMULA:
        return get_fluent_formula() < to_compare.get_fluent_formula();

    case BeliefFormulaType::BELIEF_FORMULA:
        if (get_agent() < to_compare.get_agent()) {
            return true;
        }
        if (get_agent() == to_compare.get_agent()) {
            return get_bf1() < to_compare.get_bf1();
        }
        return false;

    case BeliefFormulaType::PROPOSITIONAL_FORMULA:
        if (get_operator() < to_compare.get_operator()) {
            return true;
        }
        if (get_operator() != to_compare.get_operator()) {
            return false;
        }
        switch (m_operator) {
        case BeliefFormulaOperator::BF_AND:
        case BeliefFormulaOperator::BF_OR:
            if (get_bf1() < to_compare.get_bf1()) {
                return true;
            }
            if (get_bf1() == to_compare.get_bf1()) {
                return get_bf2() < to_compare.get_bf2();
            }
            return false;
        case BeliefFormulaOperator::BF_NOT:
            return get_bf1() < to_compare.get_bf1();
        case BeliefFormulaOperator::BF_INPAREN:
        case BeliefFormulaOperator::BF_FAIL:
        default:
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::BeliefFormulaOperatorUnset,
                "Error in comparing belief_formulae."
            );
            return false;
        }

    case BeliefFormulaType::E_FORMULA:
    case BeliefFormulaType::C_FORMULA:
    case BeliefFormulaType::D_FORMULA:
        if (get_group_agents() < to_compare.get_group_agents()) {
            return true;
        }
        if (get_group_agents() == to_compare.get_group_agents()) {
            return get_bf1() < to_compare.get_bf1();
        }
        return false;

    case BeliefFormulaType::BF_EMPTY:
    case BeliefFormulaType::BF_TYPE_FAIL:
    default:
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::BeliefFormulaTypeUnset,
            "Error in comparing belief_formulae."
        );
        return false;
    }
}