/**
 * \brief Implementation of \ref Domain.h.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 14, 2025
 */
#include "Domain.h"
#include <boost/dynamic_bitset.hpp>

#include "ArgumentParser.h"
#include "Configuration.h"
#include "HelperPrint.h"
#include "utilities/FormulaHelper.h"
#include "parse/Reader.h"

Domain::Domain()
{
    const auto& argument_parser = ArgumentParser::get_instance();
    const std::string input_file = argument_parser.get_input_file();
    const std::filesystem::path domain_path(input_file);
    m_name = domain_path.stem().string();

    if (freopen(input_file.c_str(), "r", stdin) == nullptr)
    {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::DomainFileOpenError,
            "File " + input_file + " cannot be opened." + std::string(ExitHandler::domain_file_error)
        );
        // Just to please the compiler
        exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));
    }

    // Global declaration so it accessed by the bison parser
    /// \todo Change this when new parser is added
    domain_reader->read();

    // Global declaration so it accessed by the bison parser
    /// \todo Change this when new parser is added
    if (ArgumentParser::get_instance().get_debug())
        domain_reader->print();

    build();
}

Domain& Domain::get_instance()
{
    static Domain instance;
    return instance;
}

const FluentsSet& Domain::get_fluents() const noexcept
{
    return m_fluents;
}

unsigned int Domain::get_fluent_number() const noexcept
{
    return static_cast<unsigned int>(m_fluents.size() / 2);
}

unsigned int Domain::get_size_fluent() const noexcept
{
    auto fluent_first = m_fluents.begin();
    return fluent_first != m_fluents.end() ? static_cast<unsigned int>(fluent_first->size()) : 0;
}

const ActionsSet& Domain::get_actions() const noexcept
{
    return m_actions;
}

const AgentsSet& Domain::get_agents() const noexcept
{
    return m_agents;
}

unsigned int Domain::get_agent_number() const noexcept
{
    return static_cast<unsigned int>(m_agents.size());
}

const std::string& Domain::get_name() const noexcept
{
    return m_name;
}

const InitialStateInformation& Domain::get_initial_description() const noexcept
{
    return m_initial_description;
}

const FormulaeList& Domain::get_goal_description() const noexcept
{
    return m_goal_description;
}

void Domain::build()
{
    auto& os = ArgumentParser::get_instance().get_output_stream();
    os << "\n\n========== DOMAIN OUTPUT BEGIN ==========\n";

    Grounder grounder;
    build_agents(grounder);
    build_fluents(grounder);
    build_actions(grounder);
    build_initially();
    build_goal();
    os << "========== DOMAIN OUTPUT END ==========\n\n";

}

void Domain::build_agents(Grounder & grounder)
{
    AgentsMap domain_agent_map;
    auto& os = ArgumentParser::get_instance().get_output_stream();
    os << "Building agent list..." << std::endl;
    int i = 0;
    const int agents_length = FormulaHelper::length_to_power_two(static_cast<int>(domain_reader->m_agents.size()));

    /////@TODO This will be replaced by epddl parser. Reader needs to be changed and make sure to have getter and setter
    for (const auto& agent_name : domain_reader->m_agents)
    {
        Agent agent(agents_length, i);
        domain_agent_map.insert({agent_name, agent});
        m_agents.insert(agent);
        ++i;

        if (ArgumentParser::get_instance().get_debug())
        {
            os << "Agent " << agent_name << " is " << agent << std::endl;
        };
    }
    grounder.set_agent_map(domain_agent_map);
}

void Domain::build_fluents(Grounder & grounder)
{
    FluentMap domain_fluent_map;
    auto& os = ArgumentParser::get_instance().get_output_stream();

    os << "Building fluent literals..." << std::endl;
    int i = 0;
    int bit_size = FormulaHelper::length_to_power_two(static_cast<int>(domain_reader->m_fluents.size()));

    /////@TODO This will be replaced by epddl parser. Reader needs to be changed and make sure to have getter and setter
    for (const auto& fluent_name : domain_reader->m_fluents)
    {
        Fluent fluentReal(bit_size + 1, i);
        fluentReal.set(fluentReal.size() - 1, false);
        domain_fluent_map.insert({fluent_name, fluentReal});
        m_fluents.insert(fluentReal);

        Fluent fluent_negate_real(bit_size + 1, i);
        fluent_negate_real.set(fluent_negate_real.size() - 1, true);
        domain_fluent_map.insert({NEGATION_SYMBOL + fluent_name, fluent_negate_real});
        m_fluents.insert(fluent_negate_real);
        ++i;

        if (ArgumentParser::get_instance().get_debug())
        {
            os << "Literal " << fluent_name << " is " << " " << fluentReal << std::endl;
            os << "Literal not " << fluent_name << " is " << (i - 1) << " " << fluent_negate_real << std::endl;
        }
    }
    grounder.set_fluent_map(domain_fluent_map);
}

void Domain::build_actions(Grounder & grounder)
{
    ActionNamesMap domain_action_name_map;
    auto& os = ArgumentParser::get_instance().get_output_stream();

    os << "Building action list..." << std::endl;
    int i = 0;
    int number_of_actions = static_cast<int>(domain_reader->m_actions.size());
    int bit_size = FormulaHelper::length_to_power_two(number_of_actions);

    /////@TODO This will be replaced by epddl parser. Reader needs to be changed and make sure to have getter and setter
    for (const auto& action_name : domain_reader->m_actions)
    {
        ActionId action_bitset(bit_size, i);
        Action tmp_action(action_name, action_bitset);
        domain_action_name_map.insert({action_name, action_bitset});
        m_actions.insert(tmp_action);
        ++i;

        if (ArgumentParser::get_instance().get_debug())
        {
            os << "Action " << tmp_action.get_name() << " is " << tmp_action.get_id() << std::endl;
        }
    }

    grounder.set_action_name_map(domain_action_name_map);

    HelperPrint::get_instance().set_grounder(grounder);
    build_propositions();


    if (ArgumentParser::get_instance().get_debug())
    {
        os << "\nPrinting complete action list..." << std::endl;
        for (const auto& action : m_actions)
        {
            action.print();
        }
    }
}

void Domain::build_propositions()
{
    auto& os = ArgumentParser::get_instance().get_output_stream();
    os << "Adding propositions to actions..." << std::endl;

    /////@TODO This will be replaced by epddl parser. Reader needs to be changed and make sure to have getter and setter
    for (auto& prop : domain_reader->m_propositions)
    {
        auto action_to_modify = HelperPrint::get_instance().get_grounder().ground_action(prop.get_action_name());
        for (auto it = m_actions.begin(); it != m_actions.end(); ++it)
        {
            ActionId actionTemp = action_to_modify;
            actionTemp.set(actionTemp.size() - 1, false);
            if (m_actions.size() > actionTemp.to_ulong() && it->get_id() == action_to_modify)
            {
                Action tmp = *it;
                tmp.add_proposition(prop);
                m_actions.erase(it);
                m_actions.insert(tmp);
                break;
            }
        }
    }
}

void Domain::build_initially()
{
    auto& os = ArgumentParser::get_instance().get_output_stream();
    os << "Adding to pointed world and initial conditions..." << std::endl;

    /////@TODO This will be replaced by epddl parser. Reader needs to be changed and make sure to have getter and setter
    for (auto& formula_parsed : domain_reader->m_bf_initially)
    {
        const auto formula = BeliefFormula(formula_parsed);


        switch (formula.get_formula_type())
        {
        case BeliefFormulaType::FLUENT_FORMULA:
            {
                m_initial_description.add_pointed_condition(formula.get_fluent_formula());
                if (ArgumentParser::get_instance().get_debug())
                {
                    os << "    Pointed world: ";
                    HelperPrint::get_instance().print_list(formula.get_fluent_formula());
                    os << std::endl;
                }
                break;
            }
        case BeliefFormulaType::C_FORMULA:
        case BeliefFormulaType::PROPOSITIONAL_FORMULA:
        case BeliefFormulaType::BELIEF_FORMULA:
        case BeliefFormulaType::E_FORMULA:
            {
                m_initial_description.add_initial_condition(formula);
                if (ArgumentParser::get_instance().get_debug())
                {
                    os << "Added to initial conditions: ";
                    formula.print();
                    os << std::endl;
                }
                break;
            }
        case BeliefFormulaType::BF_EMPTY:
            break;
        default:
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::DomainBuildError,
                "Error in the 'initially' conditions."
            );
        }
    }

    m_initial_description.set_ff_forS5();

}

void Domain::build_goal()
{
    auto& os = ArgumentParser::get_instance().get_output_stream();
    os << "Adding to Goal..." << std::endl;

    /////@TODO This will be replaced by epddl parser. Reader needs to be changed and make sure to have getter and setter
    for (auto& formula_parsed : domain_reader->m_bf_goal)
    {
        const auto formula = BeliefFormula(formula_parsed);
        m_goal_description.push_back(formula);
        if (ArgumentParser::get_instance().get_debug())
        {
            os << "    ";
            formula.print();
            os << std::endl;
        }
    }
}
