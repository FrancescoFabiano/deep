// src/domain/Grounder.cpp

#include "Grounder.h"

#include "utilities/ExitHandler.h"
#include <boost/lexical_cast.hpp>

#include "ArgumentParser.h"

/**
 * \file Grounder.cpp
 * \brief Implementation of the Grounder class.
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 16, 2025
 */


Grounder::Grounder(const FluentMap& given_fluent_map, const AgentsMap& given_agent_map, const ActionNamesMap& given_action_name_map)
{
    set_fluent_map(given_fluent_map);
    set_agent_map(given_agent_map);
    set_action_name_map(given_action_name_map);
    reverse();
}



void Grounder::reverse()
{
    create_reverse_fl();
    create_reverse_ag();
    create_reverse_ac();
}

void Grounder::create_reverse_fl()
{
    r_fluent_map.clear();
    for (const auto& [name, id] : m_fluent_map)
        r_fluent_map[id] = name;
}

void Grounder::create_reverse_ag()
{
    r_agent_map.clear();
    for (const auto& [name, id] : m_agent_map)
        r_agent_map[id] = name;
}

void Grounder::create_reverse_ac()
{
    r_action_name_map.clear();
    for (const auto& [name, id] : m_action_name_map)
        r_action_name_map[id] = name;
}

void Grounder::set_fluent_map(const FluentMap& given_fluent_map)
{
    m_fluent_map = given_fluent_map;
    create_reverse_fl();
}

void Grounder::set_agent_map(const AgentsMap& given_agent_map)
{
    m_agent_map = given_agent_map;
    create_reverse_ag();
}

void Grounder::set_action_name_map(const ActionNamesMap& given_action_name_map)
{
    m_action_name_map = given_action_name_map;
    create_reverse_ac();
}

[[nodiscard]] const FluentMap& Grounder::get_fluent_map() const
{
    return m_fluent_map;
}

[[nodiscard]] const AgentsMap& Grounder::get_agent_map() const
{
    return m_agent_map;
}

[[nodiscard]] const ActionNamesMap& Grounder::get_action_name_map() const
{
    return m_action_name_map;
}

[[nodiscard]] Fluent Grounder::ground_fluent(const std::string& to_ground) const
{
    if (const auto it = m_fluent_map.find(to_ground); it != m_fluent_map.end())
        return it->second;

    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::DomainUndeclaredFluent,
        "ERROR (ground): Fluent '" + to_ground + "' is undeclared (grounding)."
    );
    return Fluent{}; // Unreachable, but avoids compiler warning
}

[[nodiscard]] FluentsSet Grounder::ground_fluent(const StringsSet& to_ground) const
{
    FluentsSet y;
    for (const auto& name : to_ground)
        y.insert(ground_fluent(name));
    return y;
}

[[nodiscard]] FluentFormula Grounder::ground_fluent(const StringSetsSet& to_ground) const
{
    FluentFormula y;
    for (const auto& names : to_ground)
        y.insert(ground_fluent(names));
    return y;
}

[[nodiscard]] Agent Grounder::ground_agent(const std::string& to_ground) const
{
    auto it = m_agent_map.find(to_ground);
    if (it != m_agent_map.end())
        return it->second;

    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::DomainUndeclaredAgent,
        "ERROR (ground): Agent '" + to_ground + "' is undeclared."
    );
    return Agent{}; // Unreachable
}

[[nodiscard]] AgentsSet Grounder::ground_agent(const StringsSet& x) const
{
    AgentsSet y;
    for (const auto& name : x)
        y.insert(ground_agent(name));
    return y;
}

[[nodiscard]] ActionId Grounder::ground_action(const std::string& to_ground) const
{
    auto it = m_action_name_map.find(to_ground);
    if (it != m_action_name_map.end())
        return it->second;

    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::DomainUndeclaredAction,
        "ERROR (ground): Action '" + to_ground + "' is undeclared."
    );
    return ActionId{}; // Unreachable
}

[[nodiscard]] std::string Grounder::deground_fluent(const Fluent& to_deground) const
{
    auto it = r_fluent_map.find(to_deground);
    if (it != r_fluent_map.end())
        return it->second;

    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::DomainUndeclaredFluent,
        "ERROR (deground): Fluent '" + boost::lexical_cast<std::string>(to_deground) + "' is undeclared."
    );
    return {};
}

[[nodiscard]] StringsSet Grounder::deground_fluent(const FluentsSet& to_deground) const
{
    StringsSet y;
    for (const auto& id : to_deground)
        y.insert(deground_fluent(id));
    return y;
}

[[nodiscard]] StringSetsSet Grounder::deground_fluent(const FluentFormula& to_deground) const
{
    StringSetsSet y;
    for (const auto& ids : to_deground)
        y.insert(deground_fluent(ids));
    return y;
}

[[nodiscard]] std::string Grounder::deground_agent(const Agent& to_deground) const
{
    auto it = r_agent_map.find(to_deground);
    if (it != r_agent_map.end())
        return it->second;

    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::DomainUndeclaredAgent,
        "ERROR (deground): Agent '" + boost::lexical_cast<std::string>(to_deground) + "' is undeclared."
    );
    return {};
}

[[nodiscard]] StringsSet Grounder::deground_agents(const AgentsSet& to_deground) const
{
    StringsSet y;
    for (const auto& id : to_deground)
        y.insert(deground_agent(id));
    return y;
}

[[nodiscard]] std::string Grounder::deground_action(const ActionId& to_deground) const
{
    if (const auto it = r_action_name_map.find(to_deground); it != r_action_name_map.end())
        return it->second;

    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::DomainUndeclaredAction,
        "ERROR (deground): Action '" + boost::lexical_cast<std::string>(to_deground) + "' is undeclared."
    );
    return {};
}

Grounder& Grounder::operator=(const Grounder& to_copy)
{
    if (this != &to_copy) {
        m_fluent_map = to_copy.m_fluent_map;
        m_agent_map = to_copy.m_agent_map;
        m_action_name_map = to_copy.m_action_name_map;
        r_fluent_map = to_copy.r_fluent_map;
        r_agent_map = to_copy.r_agent_map;
        r_action_name_map = to_copy.r_action_name_map;
    }
    return *this;
}

Grounder::Grounder(const Grounder& other)
    : m_fluent_map(other.m_fluent_map),
      m_agent_map(other.m_agent_map),
      m_action_name_map(other.m_action_name_map),
      r_fluent_map(other.r_fluent_map),
      r_agent_map(other.r_agent_map),
      r_action_name_map(other.r_action_name_map)
{
    // Nothing else needed; all members copied.
}