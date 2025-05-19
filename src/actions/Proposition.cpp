// src/actions/Proposition.cpp

/**
 * \file Proposition.cpp
 * \brief Implementation of Proposition class.
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 16, 2025
 */

#include "Proposition.h"

#include "HelperPrint.h"
#include "domain/Domain.h"

PropositionType Proposition::get_type() const noexcept {
    return m_type;
}

const std::string& Proposition::get_action_name() const noexcept {
    return m_action_name;
}

FluentFormula Proposition::get_action_effect() const {
    return Domain::get_instance().get_grounder().ground_fluent(m_action_effect);
}

Agent Proposition::get_agent() const {
    return Domain::get_instance().get_grounder().ground_agent(m_agent);
}

const BeliefFormula& Proposition::get_observability_conditions() const noexcept {
    return m_observability_conditions;
}

const BeliefFormula& Proposition::get_executability_conditions() const noexcept {
    return m_executability_conditions;
}

void Proposition::set_type(const PropositionType to_set) noexcept {
    m_type = to_set;
}

void Proposition::set_action_name(const std::string& to_set) {
    m_action_name = to_set;
}

void Proposition::add_action_effect(const StringsSet& to_add) {
    m_action_effect.insert(to_add);
}

void Proposition::set_action_effect(const StringSetsSet& to_set) {
    m_action_effect = to_set;
}

void Proposition::set_agent(const std::string& to_set) {
    m_agent = to_set;
}

void Proposition::set_observability_conditions(const BeliefFormula& to_set) {
    m_observability_conditions = to_set;
}

void Proposition::set_executability_conditions(const BeliefFormula& to_set) {
    m_executability_conditions = to_set;
}

std::string Proposition::type_to_string(PropositionType type) {
    switch (type) {
    case PropositionType::EXECUTABILITY: return "EXECUTABILITY";
    case PropositionType::ONTIC:         return "ONTIC";
    case PropositionType::SENSING:       return "SENSING";
    case PropositionType::ANNOUNCEMENT:  return "ANNOUNCEMENT";
    case PropositionType::OBSERVANCE:    return "OBSERVANCE";
    case PropositionType::AWARENESS:     return "AWARENESS";
    case PropositionType::NOTSET:        return "NOTSET";
    default:                             return "UNKNOWN";
    }
}

void Proposition::print(std::ostream& os) const {
    switch (m_type) {
        case PropositionType::ONTIC:
            os << m_action_name << " causes ";
            break;
        case PropositionType::EXECUTABILITY:
            os << m_action_name << " executable ";
            break;
        case PropositionType::SENSING:
            os << m_action_name << " determines ";
            break;
        case PropositionType::ANNOUNCEMENT:
            os << m_action_name << " announces ";
            break;
        case PropositionType::OBSERVANCE:
            os << m_agent << " observes " << m_action_name;
            break;
        case PropositionType::AWARENESS:
            os << m_agent << " aware of " << m_action_name;
            break;
        default:
            break;
    }

    os << "\n Effects:\n";
    HelperPrint::print_list(m_action_effect);

    // Uncomment and adapt if you want to print observability/executability conditions
    // os << "\nObservability conditions:\n";
    // m_observability_conditions.print();
    // os << "\nExecutability conditions:\n";
    // m_executability_conditions.print();

    os << std::endl;
}