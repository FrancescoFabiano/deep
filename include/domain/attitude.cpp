/*
 * \brief Implementation of \ref attitude.h.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date December 17, 2020
 */
#include "attitude.h"
#include "domain.h"

agents_attitudes attitude::get_type() const
{
	return m_type;
}

std::string attitude::get_string_type() const
{

	switch ( m_type ) {
	case P_KEEPER:
		return "P_KEEPER";
	case P_INSECURE:
		return "P_INSECURE";
	case F_TRUSTY:
		return "F_TRUSTY";
	case F_MISTRUSTY:
		return "F_MISTRUSTY";
	case F_UNTRUSTY:
		return "F_UNTRUSTY";
	case F_STUBBORN:
		return "F_STUBBORN";
	default:
		return "P_KEEPER";
	}
}

agent attitude::get_agent() const
{
	return domain::get_instance().get_grounder().ground_agent(m_agent);
}

agent attitude::get_executor() const
{
	return domain::get_instance().get_grounder().ground_agent(m_executor);
}

const belief_formula & attitude::get_attitude_conditions()
{
	m_attitude_conditions.ground();
	return m_attitude_conditions;
}

const belief_formula & attitude::get_original_attitude_conditions() const
{
	return m_attitude_conditions;
}

void attitude::set_type(agents_attitudes to_set)
{
	m_type = to_set;
}

void attitude::set_agent(const std::string & to_set)
{
	m_agent = to_set;

}

void attitude::set_executor(const std::string & to_set)
{
	m_executor = to_set;

}

void attitude::set_attitude_conditions(const belief_formula & to_set)
{
	m_attitude_conditions = to_set;

}

void attitude::print(const belief_formula & cond) const
{
	std::cout << m_agent << " has attitude " << get_string_type() << " when " << m_executor << " is the executor of the action if ";
	cond.print();
}

bool attitude::operator=(const attitude& to_copy)
{
	m_agent = domain::get_instance().get_grounder().deground_agent(to_copy.get_agent());
	m_executor = domain::get_instance().get_grounder().deground_agent(to_copy.get_executor());
	m_type = to_copy.get_type();
	m_attitude_conditions = to_copy.get_original_attitude_conditions();

	return true;
}
