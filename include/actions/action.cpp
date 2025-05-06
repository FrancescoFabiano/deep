/*
 * File:   action.cpp
 * Author: Francesco
 *
 * Created on April 4, 2019, 2:58 PM
 */
#include "action.h"
#include "../domain/domain.h"

/*********************************************************************
 Action implementation
 **********************************************************************/

/* constructor and destructor */

action::action()
{
	m_type = NOTSET;
}

action::action(const std::string& name, action_id id)
{
	set_name(name);
	set_id(id);
}

std::string action::get_name() const
{
	return m_name;
}

void action::set_name(const std::string & name)
{
	m_name = name;
}

agent action::get_executor() const
{
	return m_executor;
}

void action::set_executor(agent executor)
{
	m_executor = executor;
}

action_id action::get_id() const
{
	return m_id;
}

void action::set_id(action_id id)
{
	m_id = id;
}

const proposition_type action::get_type() const
{
	return m_type;
}

void action::set_type(proposition_type type)
{
	if (type != NOTSET) {
		if (m_type == NOTSET) {
			/*std::cerr << "Two different actions with the same name." << std::endl;
			exit(1);*/
			m_type = type;
		}
	}
}

const formula_list& action::get_executability() const
{
	return m_executability;
}

const effects_map& action::get_effects() const
{
	return m_effects;
}

const observability_map& action::get_fully_observants() const
{
	return m_fully_observants;
}

const observability_map& action::get_partially_observants() const
{
	return m_partially_observants;
}

void action::add_executability(const belief_formula& exec)
{
	///Parameter Passing ok because push_back makes copy.
	m_executability.push_back(exec);
}

void action::add_effect(const fluent_formula& effect, const belief_formula &condition)
{
	///Parameter Passing ok because is map::value_type and it makes copy

	m_effects.insert(effects_map::value_type(effect, condition));
}

void action::add_fully_observant(agent fully, const belief_formula &condition)
{
	////Parameter Passing ok because is map::value_type and it makes copy

	m_fully_observants.insert(observability_map::value_type(fully, condition));
}

void action::add_partially_observant(agent partial, const belief_formula &condition)
{
	m_partially_observants.insert(observability_map::value_type(partial, condition));
}

void action::add_proposition(proposition & prop)
{
	///Parameter Passing ok because its methods all make copies

	switch ( prop.get_type() ) {

		//Add action to the the list (name as identifier, then set id) then update the conditions and the awareness of the action so it's complete)
	case ONTIC:
		set_type(ONTIC);
		add_effect(prop.get_action_effect(), prop.get_grounded_executability_conditions());
		break;

	case SENSING:
		set_type(SENSING);
		add_effect(prop.get_action_effect(), prop.get_grounded_executability_conditions());
		break;

	case ANNOUNCEMENT:
		set_type(ANNOUNCEMENT);
		add_effect(prop.get_action_effect(), prop.get_grounded_executability_conditions());
		break;
		/***************DOXASTIC REASONING***************/
	case EXECUTOR:
		set_type(NOTSET);
		//@TODO:What if there is more than one? Then CNF or DNF
		set_executor(prop.get_agent());
		break;
	case LIES:
		set_type(LIES);
		add_effect(prop.get_action_effect(), prop.get_grounded_executability_conditions());
		break;
		/***************END DOXASTIC***************/

	case OBSERVANCE:
		set_type(NOTSET);
		add_fully_observant(prop.get_agent(), prop.get_grounded_observability_conditions());
		break;

	case AWARENESS:
		set_type(NOTSET);
		add_partially_observant(prop.get_agent(), prop.get_grounded_observability_conditions());
		break;

	case EXECUTABILITY:
		set_type(NOTSET);
		//@TODO:What if there is more than one? Then CNF or DNF
		add_executability(prop.get_grounded_executability_conditions());
		break;
	default:
		break;
	}
}

bool action::operator<(const action& act) const
{
	return m_id < act.get_id();
}

bool action::operator=(const action& act)
{
	set_name(act.get_name());
	set_id(act.get_id());
	m_type = act.get_type();

	m_executability = act.get_executability();
	m_executor = act.get_executor();
	m_fully_observants = act.get_fully_observants();
	m_partially_observants = act.get_partially_observants();
	m_effects = act.get_effects();

	return true;
}

void action::print() const
{
	grounder grounder = domain::get_instance().get_grounder();
	std::cout << "\nAction " << get_name() << ":" << std::endl;
	std::cout << "	ID: " << get_id() << ":" << std::endl;
	std::cout << "	Type: " << get_type() << std::endl;
	
	std::cout << "	Executability:";
	formula_list::const_iterator it_fl;
	for (it_fl = m_executability.begin(); it_fl != m_executability.end(); ++it_fl) {
		std::cout << " | ";
		it_fl->print();
		//std::cout << std::endl;
	}

	std::cout << "\n	Effects:";
	effects_map::const_iterator it_effmap;
	for (it_effmap = m_effects.begin(); it_effmap != m_effects.end(); ++it_effmap) {
		std::cout << " | ";
		printer::get_instance().print_list(it_effmap->first);
		std::cout << " if ";
		it_effmap->second.print();
	}


	std::cout << "\n	Fully Observant:";
	observability_map::const_iterator it_obsmap;
	for (it_obsmap = m_fully_observants.begin(); it_obsmap != m_fully_observants.end(); ++it_obsmap) {
		std::cout << " | " << grounder.deground_agent(it_obsmap->first) << " if ";
		//printer::get_instance().print_list(it_obsmap->second);
		it_obsmap->second.print();

	}

	std::cout << "\n	Partially Observant:";
	for (it_obsmap = m_partially_observants.begin(); it_obsmap != m_partially_observants.end(); ++it_obsmap) {
		std::cout << " | " << grounder.deground_agent(it_obsmap->first) << " if ";
		//printer::get_instance().print_list(it_obsmap->second);
		it_obsmap->second.print();
	}
	std::cout << std::endl;


}
//
//  cout << "    Imposibility:\n";
//  for (it1 = get_imposs()->begin();
//       it1 != get_imposs()->end(); ++it1) {
//    cout << "\t";
//    m_planner->print(*it1);
//    cout << endl;
//  }
//
//  // print conditional effects
//  cout << "    Conditional Effects\n";
//  for (it2 = get_effects()->begin();
//       it2 != get_effects()->end(); ++it2) {
//    cout << "\t";
//    m_planner->print(*it2->get_head());
//    cout << " <- ";
//    m_planner->print(*it2->get_body());
//    cout << endl;
//  }
//}

