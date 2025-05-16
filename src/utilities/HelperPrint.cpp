#include "HelperPrint.h"
#include "ExitHandler.h"
#include <iostream>

#include "formulae/BeliefFormula.h"

/**
 * \file HelperPrint.cpp
 * \brief Implementation of HelperPrint.h
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 2025
 */

HelperPrint& HelperPrint::get_instance()
{
    static HelperPrint instance;
    return instance;
}

void HelperPrint::set_grounder(const Grounder& gr)
{
    m_grounder = gr;
    m_set_grounder = true;
}

void HelperPrint::print_list(const StringsSet& to_print)
{
    bool first = true;
    for (const auto& str : to_print) {
        if (!first) std::cout << ",";
        first = false;
        std::cout << str;
    }
}

void HelperPrint::print_list(const StringSetsSet& to_print)
{
    bool first = true;
    for (const auto& set : to_print) {
        if (!first) std::cout << " OR ";
        first = false;
        print_list(set);
    }
}

void HelperPrint::print_list(const FluentsSet& to_print) const
{
    if (m_set_grounder) {
        print_list(m_grounder.deground_fluent(to_print));
    } else {
        bool first = true;
        for (const auto& f : to_print) {
            if (!first) std::cout << ",";
            first = false;
            std::cout << f;
        }
    }
}

void HelperPrint::print_list(const FluentFormula& to_print) const
{
    if (m_set_grounder) {
        print_list(m_grounder.deground_fluent(to_print));
    } else {
        bool first = true;
        for (const auto& fs : to_print) {
            if (!first) std::cout << " OR ";
            first = false;
            print_list(fs);
        }
    }
}

void HelperPrint::print_list(const FormulaeList& to_print)
{
    bool first = true;
    for (const auto& formula : to_print) {
        if (!first) std::cout << " AND ";
        first = false;
        formula.print();
    }
}

void HelperPrint::print_list(const KripkeWorldPointersSet& to_print)
{
    bool first = true;
    for (const auto& ptr : to_print) {
        if (!(ptr == nullptr)) {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::PrintNullPointerError,
                "Null pointer encountered in KripkeWorldPointersSet during print."
            );
        }
        if (!first) std::cout << "\n";
        first = false;
        std::cout << ptr->get_id();
    }
}

void HelperPrint::print_list(const ActionIdsList& to_print) const
{
    bool first = true;
    for (const auto& id : to_print) {
        if (!first) std::cout << ", ";
        first = false;
        if (m_set_grounder) {
            std::cout << m_grounder.deground_action(id);
        } else {
            std::cout << id;
        }
    }
}

void HelperPrint::print_list_ag(const AgentsSet& to_print) const
{
    bool first = true;
    for (const auto& ag : to_print) {
        if (!first) std::cout << ", ";
        first = false;
        if (m_set_grounder) {
            std::cout << m_grounder.deground_agent(ag);
        } else {
            std::cout << ag;
        }
    }
}

void HelperPrint::print_belief_formula(const BeliefFormula& to_print) const
{
    switch (to_print.get_formula_type()) {
    case BeliefFormulaType::FLUENT_FORMULA:
        print_list(to_print.get_fluent_formula());
        break;
    case BeliefFormulaType::BELIEF_FORMULA:
        std::cout << "B(" << m_grounder.deground_agent(to_print.get_agent()) << ",(";
        print_belief_formula(to_print.get_bf1());
        std::cout << "))";
        break;
    case BeliefFormulaType::D_FORMULA:
        std::cout << "D([";
        print_list_ag(to_print.get_group_agents());
        std::cout << "],";
        print_belief_formula(to_print.get_bf1());
        std::cout << ")";
        break;
    case BeliefFormulaType::C_FORMULA:
        std::cout << "C([";
        print_list_ag(to_print.get_group_agents());
        std::cout << "],";
        print_belief_formula(to_print.get_bf1());
        std::cout << ")";
        break;
    case BeliefFormulaType::E_FORMULA:
        std::cout << "E([";
        print_list_ag(to_print.get_group_agents());
        std::cout << "],";
        print_belief_formula(to_print.get_bf1());
        std::cout << ")";
        std::cout << ")";
        break;
    case BeliefFormulaType::PROPOSITIONAL_FORMULA:
        if (to_print.get_operator() == BeliefFormulaOperator::BF_NOT)
            std::cout << "NOT(";
        print_belief_formula(to_print.get_bf1());
        if (to_print.get_operator() == BeliefFormulaOperator::BF_NOT)
            std::cout << ")";
        if (to_print.get_operator() == BeliefFormulaOperator::BF_AND)
            std::cout << " AND ";
        if (to_print.get_operator() == BeliefFormulaOperator::BF_OR)
            std::cout << " OR ";
        else if (to_print.get_operator() == BeliefFormulaOperator::BF_FAIL) {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::BeliefFormulaOperatorUnset,
                "ERROR IN DECLARATION."
            );
        }
        if (!to_print.is_bf2_null()) {
            print_belief_formula(to_print.get_bf2());
        }
        break;
    case BeliefFormulaType::BF_EMPTY:
        std::cout << "Empty\n";
        break;
    case BeliefFormulaType::BF_TYPE_FAIL:
    default:
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::BeliefFormulaTypeUnset,
            "Unknown BeliefFormula type."
        );
        break;
    }
}
