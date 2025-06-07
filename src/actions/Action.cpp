/**
* \brief Implementation of \ref Action.h.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 14, 2025
 */


#include "Action.h"

#include <utility>

#include "ArgumentParser.h"
#include "Domain.h"
#include "HelperPrint.h"
#include "utilities/ExitHandler.h"
#include "actions/Proposition.h"


// Constructor
Action::Action(const std::string& name, ActionId id)
{
    set_name(name);
    set_id(std::move(id));
}

Action::Action(const Action& other)
    : m_name(other.m_name),
      m_id(other.m_id),
      m_executor(other.m_executor),
      m_type(other.m_type),
      m_executability(other.m_executability),
      m_fully_observants(other.m_fully_observants),
      m_partially_observants(other.m_partially_observants),
      m_effects(other.m_effects)
{
}

std::string Action::get_name() const
{
    return m_name;
}

void Action::set_name(const std::string& name)
{
    m_name = name;
}

Agent Action::get_executor() const
{
    return m_executor;
}

void Action::set_executor(const Agent& executor)
{
    m_executor = executor;
}

ActionId Action::get_id() const
{
    return m_id;
}

void Action::set_id(ActionId id)
{
    m_id = std::move(id);
}

PropositionType Action::get_type() const
{
    return m_type;
}

void Action::set_type(PropositionType type)
{
    if (type != PropositionType::NOTSET)
    {
        if (m_type == PropositionType::NOTSET)
        {
            m_type = type;
        }
        else if (m_type != type)
        {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::ActionTypeConflict,
                "Conflicting action types for action '" + m_name + "'."
            );
        }
    }
}

const FormulaeList& Action::get_executability() const
{
    return m_executability;
}

const EffectsMap& Action::get_effects() const
{
    return m_effects;
}

const ObservabilitiesMap& Action::get_fully_observants() const
{
    return m_fully_observants;
}

const ObservabilitiesMap& Action::get_partially_observants() const
{
    return m_partially_observants;
}

void Action::add_executability(const BeliefFormula& exec)
{
    m_executability.push_back(exec);
}

void Action::add_effect(const FluentFormula& effect, const BeliefFormula& condition)
{
    auto [it, inserted] = m_effects.insert(EffectsMap::value_type(effect, condition));
    if (!inserted)
    {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::ActionEffectError,
            "Failed to add effect to action '" + m_name + "'."
        );
    }
}

void Action::add_fully_observant(const Agent& fully, const BeliefFormula& condition)
{
    m_fully_observants.insert(ObservabilitiesMap::value_type(fully, condition));
}

void Action::add_partially_observant(const Agent& partial, const BeliefFormula& condition)
{
    m_partially_observants.insert(ObservabilitiesMap::value_type(partial, condition));
}

void Action::add_proposition(const Proposition& to_add)
{
    switch (to_add.get_type())
    {
    case PropositionType::ONTIC:
        set_type(PropositionType::ONTIC);
        add_effect(to_add.get_action_effect(), BeliefFormula(to_add.get_executability_conditions()));
        break;
    case PropositionType::SENSING:
        set_type(PropositionType::SENSING);
        add_effect(to_add.get_action_effect(), BeliefFormula(to_add.get_executability_conditions()));
        break;
    case PropositionType::ANNOUNCEMENT:
        set_type(PropositionType::ANNOUNCEMENT);
        add_effect(to_add.get_action_effect(), BeliefFormula(to_add.get_executability_conditions()));
        break;
    case PropositionType::OBSERVANCE:
        set_type(PropositionType::NOTSET);
        add_fully_observant(to_add.get_agent(), BeliefFormula(to_add.get_observability_conditions()));
        break;
    case PropositionType::AWARENESS:
        set_type(PropositionType::NOTSET);
        add_partially_observant(to_add.get_agent(), BeliefFormula(to_add.get_observability_conditions()));
        break;
    case PropositionType::EXECUTABILITY:
        set_type(PropositionType::NOTSET);
        add_executability(BeliefFormula(to_add.get_executability_conditions()));
        break;
    default:
        break;
    }
}

bool Action::operator<(const Action& act) const
{
    return m_id < act.get_id();
}

Action& Action::operator=(const Action& act)
{
    if (this != &act)
    {
        set_name(act.get_name());
        set_id(act.get_id());
        m_type = act.get_type();
        m_executability = act.get_executability();
        m_executor = act.get_executor();
        m_fully_observants = act.get_fully_observants();
        m_partially_observants = act.get_partially_observants();
        m_effects = act.get_effects();
    }
    return *this;
}

void Action::print() const
{
    auto& os = ArgumentParser::get_instance().get_output_stream();
    const auto grounder = HelperPrint::get_instance().get_grounder();
    os << "\nAction " << get_name() << ":" << std::endl;
    os << "    ID: " << get_id() << ":" << std::endl;
    os << "    Type: " << Proposition::type_to_string(get_type()) << std::endl;

    os << "    Executability:";
    for (const auto& exec : m_executability)
    {
        os << " | ";
        exec.print();
    }

    os << "\n    Effects:";
    for (const auto& [effect, condition] : m_effects)
    {
        os << " | ";
        HelperPrint::get_instance().print_list(effect);
        os << " if ";
        condition.print();
    }

    os << "\n    Fully Observant:";
    for (const auto& [agent, condition] : m_fully_observants)
    {
        os << " | " << grounder.deground_agent(agent) << " if ";
        condition.print();
    }

    os << "\n    Partially Observant:";
    for (const auto& [agent, condition] : m_partially_observants)
    {
        os << " | " << grounder.deground_agent(agent) << " if ";
        condition.print();
    }
    os << std::endl;
}
