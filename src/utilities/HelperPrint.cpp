#include "HelperPrint.h"
#include "ExitHandler.h"
#include "KripkeWorld.h"

#include "formulae/BeliefFormula.h"
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>

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

void HelperPrint::print_list(const StringsSet& to_print, std::ostream& os)
{
    bool first = true;
    for (const auto& str : to_print) {
        if (!first) os << ",";
        first = false;
        os << str;
    }
}

void HelperPrint::print_list(const StringSetsSet& to_print, std::ostream& os)
{
    bool first = true;
    for (const auto& set : to_print) {
        if (!first) os << " OR ";
        first = false;
        print_list(set, os);
    }
}

void HelperPrint::print_list(const FluentsSet& to_print, std::ostream& os) const
{
    if (m_set_grounder) {
        print_list(m_grounder.deground_fluent(to_print), os);
    } else {
        bool first = true;
        for (const auto& f : to_print) {
            if (!first) os << ",";
            first = false;
            os << f;
        }
    }
}

void HelperPrint::print_list(const FluentFormula& to_print, std::ostream& os) const
{
    if (m_set_grounder) {
        print_list(m_grounder.deground_fluent(to_print), os);
    } else {
        bool first = true;
        for (const auto& fs : to_print) {
            if (!first) os << " OR ";
            first = false;
            print_list(fs, os);
        }
    }
}

void HelperPrint::print_list(const FormulaeList& to_print, std::ostream& os)
{
    bool first = true;
    for (const auto& formula : to_print) {
        if (!first) os << " AND ";
        first = false;
        formula.print(os);
    }
}

void HelperPrint::print_list(const KripkeWorldPointersSet& to_print, std::ostream& os)
{
    bool first = true;
    for (const auto& ptr : to_print) {
        if (ptr.get_ptr() != nullptr) {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::PrintNullPointerError,
                "Null pointer encountered in KripkeWorldPointersSet during print."
            );
        }
        if (!first) os << "\n";
        first = false;
        os << ptr.get_ptr()->get_id();
    }
}

void HelperPrint::print_list(const ActionIdsList& to_print, std::ostream& os) const
{
    bool first = true;
    for (const auto& id : to_print) {
        if (!first) os << ", ";
        first = false;
        if (m_set_grounder) {
            os << m_grounder.deground_action(id);
        } else {
            os << id;
        }
    }
}

void HelperPrint::print_list_ag(const AgentsSet& to_print, std::ostream& os) const
{
    bool first = true;
    for (const auto& ag : to_print) {
        if (!first) os << ", ";
        first = false;
        if (m_set_grounder) {
            os << m_grounder.deground_agent(ag);
        } else {
            os << ag;
        }
    }
}

void HelperPrint::print_belief_formula(const BeliefFormula& to_print, std::ostream& os) const
{
    switch (to_print.get_formula_type()) {
    case BeliefFormulaType::FLUENT_FORMULA:
        print_list(to_print.get_fluent_formula(), os);
        break;
    case BeliefFormulaType::BELIEF_FORMULA:
        os << "B(" << m_grounder.deground_agent(to_print.get_agent()) << ",(";
        print_belief_formula(to_print.get_bf1(), os);
        os << "))";
        break;
    case BeliefFormulaType::C_FORMULA:
        os << "C([";
        print_list_ag(to_print.get_group_agents(), os);
        os << "],";
        print_belief_formula(to_print.get_bf1(), os);
        os << ")";
        break;
    case BeliefFormulaType::E_FORMULA:
        os << "E([";
        print_list_ag(to_print.get_group_agents(), os);
        os << "],";
        print_belief_formula(to_print.get_bf1(), os);
        os << ")";
        os << ")";
        break;
    case BeliefFormulaType::PROPOSITIONAL_FORMULA:
        if (to_print.get_operator() == BeliefFormulaOperator::BF_NOT)
            os << "NOT(";
        print_belief_formula(to_print.get_bf1(), os);
        if (to_print.get_operator() == BeliefFormulaOperator::BF_NOT)
            os << ")";
        if (to_print.get_operator() == BeliefFormulaOperator::BF_AND)
            os << " AND ";
        if (to_print.get_operator() == BeliefFormulaOperator::BF_OR)
            os << " OR ";
        else if (to_print.get_operator() == BeliefFormulaOperator::BF_FAIL) {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::BeliefFormulaOperatorUnset,
                "ERROR IN DECLARATION."
            );
        }
        if (!to_print.is_bf2_null()) {
            print_belief_formula(to_print.get_bf2(), os);
        }
        break;
    case BeliefFormulaType::BF_EMPTY:
        os << "Empty\n";
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

std::string HelperPrint::generate_log_file_path(const std::string& domain_file) {
    namespace fs = std::filesystem;
    fs::create_directories("log");

    fs::path domain_path(domain_file);
    std::string domain_name = domain_path.stem().string();

    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif

    int repetition = 0;
    std::string path;
    do {
        std::ostringstream oss;
        oss << "log/"
            << domain_name << "_"
            << std::put_time(&tm, "%Y%m%d_%H%M%S");
        if (repetition > 0) {
            oss << "_" << repetition;
        }
        oss << ".log";
        path = oss.str();
        ++repetition;
    } while (fs::exists(path));
    return path;
}
