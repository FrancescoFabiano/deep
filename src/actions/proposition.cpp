/*
 * \brief Implementation of \ref proposition.h.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date April 1, 2019
 */
#include <iostream>

#include "proposition.h"
#include "../domain/domain.h"

proposition_type proposition::get_type() const
{
	return m_type;
}

const std::string & proposition::get_action_name() const
{
	return m_action_name;
}

fluent_formula proposition::get_action_effect() const
{
	return domain::get_instance().get_grounder().ground_fluent(m_action_effect);
}

agent proposition::get_agent() const
{
	return domain::get_instance().get_grounder().ground_agent(m_agent);
}

const belief_formula & proposition::get_observability_conditions() const
{
	return m_observability_conditions;
}

const belief_formula & proposition::get_executability_conditions() const
{
	return m_executability_conditions;
}


const belief_formula & proposition::get_grounded_observability_conditions()
{
	m_observability_conditions.ground();
	return m_observability_conditions;
}

const belief_formula & proposition::get_grounded_executability_conditions()
{
	m_executability_conditions.ground();
	return m_executability_conditions;
}

void proposition::set_type(proposition_type to_set)
{
	m_type = to_set;
}

void proposition::set_action_name(const std::string & to_set)
{
	m_action_name = to_set;
}

void proposition::add_action_effect(const string_set & to_add)
{
	m_action_effect.insert(to_add);
}

void proposition::set_action_effect(const string_set_set & to_set)
{
	m_action_effect = to_set;
}

void proposition::set_agent(const std::string & to_set)
{
	m_agent = to_set;
}

void proposition::set_observability_conditions(const belief_formula & to_set)
{
	m_observability_conditions = to_set;
}

void proposition::set_executability_conditions(const belief_formula & to_set)
{
	m_executability_conditions = to_set;
	//m_executability_conditions.ground();
}

void proposition::print() const
{
	switch ( m_type ) {
	case ONTIC:
		std::cout << m_action_name << " causes ";
		break;
	case EXECUTABILITY:
		std::cout << m_action_name << " executable ";
		break;
	case SENSING:
		std::cout << m_action_name << " determines ";
		break;
	case ANNOUNCEMENT:
		std::cout << m_action_name << " announces ";
		break;
	case OBSERVANCE:
		std::cout << m_agent << " observes " << m_action_name;
		break;
	case AWARENESS:
		std::cout << m_agent << " aware of " << m_action_name;
		break;
	case EXECUTOR:
		std::cout << m_agent << " executor of " << m_action_name;
		break;
	default: /* static */
		break;
	}

	/**
	 * \todo check implementation, the print should be moved inside the switch.
	 */

	std::cout << "\n Effects:\n";
	printer::get_instance().print_list(m_action_effect);


	/*std::cout << "\n Conditional Execution for act -> incorporated with Executability conditions:\n";
	first = true;
	for (it = m_action_precondition.begin(); it != m_action_precondition.end(); it++) {
		if (!first)
			std::cout << ",";
		else
			std::cout << " if ";
		first = false;
		std::cout << *it;
	}*/

	//For Printing Info
	//	std::cout << "\n Observability conditions:\n";
	//	printer::get_instance().print_list(m_observability_conditions);
	//	/*first2 = true;
	//	for (it2 = m_observability_conditions.begin(); it2 != m_observability_conditions.end(); it2++) {
	//		if ((m_type == AWARENESS || m_type == OBSERVANCE) && !if_fluform) {
	//			std::cout << " if ";
	//			if_fluform = true;
	//		}
	//		if (!first2)
	//			std::cout << " | ";
	//		first2 = false;
	//		first = true;
	//		std::cout << "(";
	//		for (it = it2->begin(); it != it2->end(); it++) {
	//			if (!first)
	//				std::cout << " AND ";
	//			first = false;
	//			std::cout << *it;
	//		}
	//		std::cout << ")";
	//	}*/
	//
	//	std::cout << "\n Executability conditions:\n";
	//	//@TODO: why this or?
	//	if (m_executability_conditions.m_bf1 != nullptr || m_type == ONTIC) {
	//		std::cout << " if ";
	//		m_executability_conditions.print();
	//	}
	//	std::cout << std::endl;

};

