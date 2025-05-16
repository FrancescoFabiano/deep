/**
 * \brief Implementation of \ref Domain.h.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 14, 2025
 */
#include "domain.h"
#include <boost/dynamic_bitset.hpp>
#include <iostream>
#include <ranges>

#include "ArgumentParser.h"
#include "utilities/FormulaHelper.h"


Domain* Domain::instance = nullptr;


Domain::Domain()
{
    auto& arg_parser = ArgumentParser::get_instance();
    std::string input_file = arg_parser.get_input_file();

    if (freopen(input_file.c_str(), "r", stdin) == nullptr) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::DomainFileOpenError,
            "File " + input_file + " cannot be opened." + std::string(ExitHandler::domain_file_error)
        );
    }

    instance->m_name = input_file.substr(input_file.find_last_of("\\/") + 1);
    instance->m_name = instance->m_name.substr(0, instance->m_name.find_last_of('.'));

    /////@TODO This will be replaced by epddl parser
    auto domain_reader = boost::make_shared<reader>(reader());
    domain_reader->read();
    instance->m_reader = domain_reader;
    instance->build();
}

Domain& Domain::get_instance() {
    if (!instance) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::DomainInstanceError,
            "Domain instance not created. Call create_instance() first."
        );
    }
    return *instance;
}


void Domain::create_instance() {
    if (!instance) {
        instance = new Domain();
    }
}
const Grounder& Domain::get_grounder() const noexcept {
    return m_grounder;
}

const FluentsSet& Domain::get_fluents() const noexcept {
    return m_fluents;
}

unsigned int Domain::get_fluent_number() const noexcept {
    return static_cast<unsigned int>(m_fluents.size() / 2);
}

unsigned int Domain::get_size_fluent() const noexcept {
    auto fluent_first = m_fluents.begin();
    return fluent_first != m_fluents.end() ? static_cast<unsigned int>(fluent_first->size()) : 0;
}

const ActionsSet& Domain::get_actions() const noexcept {
    return m_actions;
}

const AgentsSet& Domain::get_agents() const noexcept {
    return m_agents;
}

unsigned int Domain::get_agent_number() const noexcept {
    return static_cast<unsigned int>(m_agents.size());
}

const std::string& Domain::get_name() const noexcept {
    return m_name;
}

const InitialStateInformation& Domain::get_initial_description() const noexcept {
    return m_initial_description;
}

const FormulaeList& Domain::get_goal_description() const noexcept {
    return m_goal_description;
}

void Domain::build() {
    build_agents();
    build_fluents();
    build_actions();
    build_initially();
    build_goal();
}

void Domain::build_agents() {
    AgentsMap domain_agent_map;
    std::cout << "\nBuilding agent list..." << std::endl;
    int i = 0;
    int agents_length = FormulaHelper::length_to_power_two(static_cast<int>(m_reader->m_agents.size()));

    /////@TODO This will be replaced by epddl parser. Reader needs to be changed and make sure to have getter and setter
    for (const auto& agent_name : m_reader->m_agents) {
        Agent agent(agents_length, i);
        domain_agent_map.insert({agent_name, agent});
        m_agents.insert(agent);
        ++i;

        if (ArgumentParser::get_instance().get_debug()) {
            std::cout << "Agent " << agent_name << " is " << agent << std::endl;
        };
    }
    m_grounder.set_agent_map(domain_agent_map);
}

void Domain::build_fluents() {
    FluentMap domain_fluent_map;
    std::cout << "\nBuilding fluent literals..." << std::endl;
    int i = 0;
    int bit_size = FormulaHelper::length_to_power_two(static_cast<int>(m_reader->m_fluents.size()));

    /////@TODO This will be replaced by epddl parser. Reader needs to be changed and make sure to have getter and setter
    for (const auto& fluent_name : m_reader->m_fluents) {
        Fluent fluentReal(bit_size + 1, i);
        fluentReal.set(fluentReal.size() - 1, false);
        domain_fluent_map.insert({fluent_name, fluentReal});
        m_fluents.insert(fluentReal);

        Fluent fluent_negate_real(bit_size + 1, i);
        fluent_negate_real.set(fluent_negate_real.size() - 1, true);
        domain_fluent_map.insert({NEGATION_SYMBOL + fluent_name, fluent_negate_real});
        m_fluents.insert(fluent_negate_real);
        ++i;

        if (ArgumentParser::get_instance().get_debug()) {
            std::cout << "Literal " << fluent_name << " is " << " " << fluentReal << std::endl;
            std::cout << "Literal not " << fluent_name << " is " << (i - 1) << " " << fluent_negate_real << std::endl;
        }

    }
    m_grounder.set_fluent_map(domain_fluent_map);
}

void Domain::build_actions() {
    ActionNamesMap domain_action_name_map;
    std::cout << "\nBuilding action list..." << std::endl;
    int i = 0;
    int number_of_actions = static_cast<int>(m_reader->m_actions.size());
    int bit_size = FormulaHelper::length_to_power_two(number_of_actions);

    /////@TODO This will be replaced by epddl parser. Reader needs to be changed and make sure to have getter and setter
    for (const auto& action_name : m_reader->m_actions) {
        ActionId action_bitset(bit_size, i);
        Action tmp_action(action_name, action_bitset);
        domain_action_name_map.insert({action_name, action_bitset});
        m_actions.insert(tmp_action);
        ++i;

        if (ArgumentParser::get_instance().get_debug()) {
            std::cout << "Action " << tmp_action.get_name() << " is " << tmp_action.get_id() << std::endl;
        }
    }

    m_grounder.set_action_name_map(domain_action_name_map);
    printer::get_instance().set_grounder(m_grounder);

    build_propositions();


    if (ArgumentParser::get_instance().get_debug()) {
        std::cout << "\nPrinting complete action list..." << std::endl;
        for (const auto& action : m_actions) {
            action.print();
        }
    }
}

void Domain::build_propositions() {
    std::cout << "\nAdding propositions to actions..." << std::endl;

    /////@TODO This will be replaced by epddl parser. Reader needs to be changed and make sure to have getter and setter
    for (auto& prop : m_reader->m_propositions) {
        auto action_to_modify = m_grounder.ground_action(prop.get_action_name());
        for (auto it = m_actions.begin(); it != m_actions.end(); ++it) {
            ActionId actionTemp = action_to_modify;
            actionTemp.set(actionTemp.size() - 1, false);
            if (m_actions.size() > actionTemp.to_ulong() && it->get_id() == action_to_modify) {
                Action tmp = *it;
                tmp.add_proposition(prop);
                m_actions.erase(it);
                m_actions.insert(tmp);
                break;
            }
        }
    }
}

void Domain::build_initially() {
    std::cout << "\nAdding to pointed world and initial conditions..." << std::endl;

    /////@TODO This will be replaced by epddl parser. Reader needs to be changed and make sure to have getter and setter
    for (auto& formula : m_reader->m_bf_initially) {
        formula.ground();

        switch (formula.get_formula_type()) {
        case FLUENT_FORMULA: {
                m_initial_description.add_pointed_condition(formula.get_fluent_formula());
                if (ArgumentParser::get_instance().get_debug()) {
                    std::cout << "    Pointed world: ";
                    printer::get_instance().print_list(formula.get_fluent_formula());
                    std::cout << std::endl;
                }
                break;
        }
        case C_FORMULA:
        case D_FORMULA:
        case PROPOSITIONAL_FORMULA:
        case BELIEF_FORMULA:
        case E_FORMULA: {
                m_initial_description.add_initial_condition(formula);
                if (ArgumentParser::get_instance().get_debug()) {
                    std::cout << "Added to initial conditions: ";
                    formula.print();
                    std::cout << std::endl;
                }
                break;
        }
        case BF_EMPTY:
            break;
        default:
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::DomainBuildError,
                "Error in the 'initially' conditions."
            );
        }
    }
}

void Domain::build_goal() {
    std::cout << "\nAdding to Goal..." << std::endl;

    /////@TODO This will be replaced by epddl parser. Reader needs to be changed and make sure to have getter and setter
    for (auto& formula : m_reader->m_bf_goal) {
        formula.ground();
        m_goal_description.push_back(formula);
        if (ArgumentParser::get_instance().get_debug()) {
            std::cout << "    ";
            formula.print();
            std::cout << std::endl;
        }
    }
}