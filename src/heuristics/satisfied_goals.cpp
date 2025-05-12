#include "satisfied_goals.h"

satisfied_goals::satisfied_goals()
{
}

void satisfied_goals::set(const formula_list & goals)
{
	set_goals(goals);
	set_goals_number(goals.size());
}

satisfied_goals& satisfied_goals::get_instance()
{
	static satisfied_goals instance;
	return instance;
}

const formula_list & satisfied_goals::get_goals() const
{
	return m_goals;
}

void satisfied_goals::set_goals(const formula_list & to_set)
{
	m_goals = to_set;
}

unsigned short satisfied_goals::get_goals_number() const
{
	return m_goals_number;
}

void satisfied_goals::set_goals_number(unsigned short to_set)
{
	m_goals_number = to_set;
}

/*template <class T>
bool satisfied_goals<T>::operator=(const satisfied_goals<T>& to_copy)
{
	set_goals(to_copy.get_goals());
	set_goals_number(to_copy.get_goals_number());
	return true;
}*/