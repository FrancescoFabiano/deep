/*
 * File:   grounder.h
 * Author: Francesco
 *
 * Created on April 5, 2019, 12:18 PM
 */

#include "grounder.h"

grounder::grounder()
{
}

grounder::grounder(const FluentMap & fluent_map, const AgentsMap & agent_map, const ActionNamesMap & action_name_map)
{
	set_fluent_map(fluent_map);
	set_agent_map(agent_map);
	set_action_name_map(action_name_map);

	//@TODO:Remove for efficency; just for printing reasons (pass debug maybe)
	//reverse();

}

/*void grounder::reverse()
{
	create_reverse_fl(m_fluent_map);
	create_reverse_ag(m_agent_map);
	create_reverse_ac(m_action_name_map);
}*/

void grounder::create_reverse_fl()
{
	FluentMap::iterator it;
	for (it = m_fluent_map.begin(); it != m_fluent_map.end(); it++)
		r_fluent_map[it->second] = it->first;
}

void grounder::create_reverse_ag()
{
	AgentsMap::iterator it;
	for (it = m_agent_map.begin(); it != m_agent_map.end(); it++)
		r_agent_map[it->second] = it->first;
}

void grounder::create_reverse_ac()
{
	ActionNamesMap::iterator it;
	for (it = m_action_name_map.begin(); it != m_action_name_map.end(); it++)
		r_action_name_map[it->second] = it->first;
}

void grounder::set_fluent_map(const FluentMap & fluent_map)
{
	m_fluent_map = fluent_map;
	create_reverse_fl();
}

void grounder::set_agent_map(const AgentsMap & agent_map)
{
	m_agent_map = agent_map;
	create_reverse_ag();
}

void grounder::set_action_name_map(const ActionNamesMap & action_name_map)
{
	m_action_name_map = action_name_map;
	create_reverse_ac();
}

const FluentMap & grounder::get_fluent_map() const
{
	return m_fluent_map;
}

const AgentsMap & grounder::get_agent_map() const
{
	return m_agent_map;
}

const ActionNamesMap & grounder::get_action_name_map() const
{
	return m_action_name_map;
}

Fluent grounder::ground_fluent(const std::string& x) const
{
	FluentMap::const_iterator p = m_fluent_map.find(x);

	if (p != m_fluent_map.end()) {
		return(p->second);
	}

	std::cerr << "ERROR: Fluent " << x << " is undeclared." << std::endl;
	exit(1);
}

FluentsSet grounder::ground_fluent(const StringsSet& x) const
{
	StringsSet::iterator it;
	FluentsSet y;

	for (it = x.begin(); it != x.end(); it++) {
		y.insert(ground_fluent(*it));
	}

	return y;
}

FluentFormula grounder::ground_fluent(const StringSetsSet& x) const
{
	StringSetsSet::iterator it;
	FluentFormula y;

	for (it = x.begin(); it != x.end(); it++) {
		y.insert(ground_fluent(*it));
	}

	return y;
}

Agent grounder::ground_agent(const std::string& x) const
{
	AgentsMap::const_iterator p = m_agent_map.find(x);

	if (p != m_agent_map.end()) {
		return(p->second);
	}
	
	std::cerr << "ERROR: Agent " << x << " is undeclared." << std::endl;
	exit(1);
}

AgentsSet grounder::ground_agent(const StringsSet& x) const
{
	StringsSet::iterator it;
	AgentsSet y;

	for (it = x.begin(); it != x.end(); it++) {
		y.insert(ground_agent(*it));
	}

	return y;
}

ActionId grounder::ground_action(const std::string& x) const
{
	ActionNamesMap::const_iterator p = m_action_name_map.find(x);

	if (p != m_action_name_map.end()) {
		return(p->second);
	}

	std::cerr << "ERROR: Action " << x << " is undeclared." << std::endl;
	exit(1);
}

std::string grounder::deground_fluent(Fluent x) const
{
	reverse_fluent_map::const_iterator p = r_fluent_map.find(x);

	if (p != r_fluent_map.end()) {
		return(p->second);
	}

	std::cerr << "ERROR: Fluent " << x << " is undeclared." << std::endl;
	exit(1);
}

StringsSet grounder::deground_fluent(const FluentsSet& x) const
{

	FluentsSet::iterator it;
	StringsSet y;

	for (it = x.begin(); it != x.end(); it++) {
		y.insert(deground_fluent(*it));
	}

	return y;
}

StringSetsSet grounder::deground_fluent(const FluentFormula& x) const
{
	FluentFormula::iterator it;
	StringSetsSet y;

	for (it = x.begin(); it != x.end(); it++) {
		y.insert(deground_fluent(*it));
	}

	return y;
}

std::string grounder::deground_agent(Agent x) const
{
	reverse_agent_map::const_iterator p = r_agent_map.find(x);

	if (p != r_agent_map.end()) {
		return(p->second);
	}

	std::cerr << "ERROR: Agent " << x << " is undeclared." << std::endl;
	exit(1);
}

StringsSet grounder::deground_agents(const AgentsSet & x) const
{
	AgentsSet::iterator it;
	StringsSet y;

	for (it = x.begin(); it != x.end(); it++) {
		y.insert(deground_agent(*it));
	}
	
	return y;
}

std::string grounder::deground_action(ActionId x) const
{
	reverse_action_name_map::const_iterator p = r_action_name_map.find(x);

	if (p != r_action_name_map.end()) {
		return(p->second);
	}

	std::cerr << "ERROR: Action " << x << " is undeclared." << std::endl;
	exit(1);
}

/*void grounder::print_ff(const fluent_set& to_print) const
{
	printer::get_instance().print_list(deground_fluent(to_print));
}

void grounder::print_ff(const fluent_formula& to_print) const
{
	printer::get_instance().print_list(deground_fluent(to_print));
}*/

bool grounder::operator=(const grounder& to_assign)
{
	set_fluent_map(to_assign.get_fluent_map());
	set_agent_map(to_assign.get_agent_map());
	set_action_name_map(to_assign.get_action_name_map());
	return true;
}