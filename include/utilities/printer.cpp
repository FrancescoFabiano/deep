/*
 * \brief Implementation of \ref printer.h
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date April 7, 2019
 */

#include <iostream>
#include "printer.h"

printer::printer()
{
}

printer& printer::get_instance()
{
	static printer instance;
	return instance;
}

void printer::set_grounder(const grounder & gr)
{
	m_grounder = gr;
	setted_grounder = true;
}

void printer::print_list(const string_set& to_print) const
{
	bool first_print = true;

	//Iterate on the \p to_print and print its content separated by a comma to represent a conjunctive set of \ref fluent.
	string_set::const_iterator it_sl;
	for (it_sl = to_print.begin(); it_sl != to_print.end(); it_sl++) {
		if (!first_print) {
			std::cout << ",";
		}
		first_print = false;
		std::cout << *it_sl;
	}
}

void printer::print_list(const string_set_set& to_print) const
{
	bool first_print = true;
	string_set_set::const_iterator it_sll;
	//Iterate on the \p to_print and print its content using ::to_print(const string_set&) seprated by a OR to represent a DNF.

	for (it_sll = to_print.begin(); it_sll != to_print.end(); it_sll++) {
		if (!first_print) {
			std::cout << " OR ";
		}
		first_print = false;
		print_list(*it_sll);

	}
}

void printer::print_list(const fluent_set& to_print) const
{
	if (setted_grounder) {
		print_list(m_grounder.deground_fluent(to_print));
	} else {
		bool first_print = true;

		//Iterate on the \p to_print and print its content separated by a comma to represent a conjunctive set of \ref fluent.
		fluent_set::const_iterator it_sl;
		for (it_sl = to_print.begin(); it_sl != to_print.end(); it_sl++) {
			if (!first_print) {
				std::cout << ",";
			}
			first_print = false;
			std::cout << *it_sl;
		}
	}
}

void printer::print_list(const fluent_formula& to_print) const
{
	if (setted_grounder) {
		print_list(m_grounder.deground_fluent(to_print));
	} else {
		bool first_print = true;
		fluent_formula::const_iterator it_sll;
		//Iterate on the \p to_print and print its content using ::to_print(const string_set&) seprated by a OR to represent a DNF.

		for (it_sll = to_print.begin(); it_sll != to_print.end(); it_sll++) {
			if (!first_print) {
				std::cout << " OR ";
			}
			first_print = false;
			print_list(*it_sll);

		}
	}
}

void printer::print_list(const formula_list& to_print) const
{
	bool first_print = true;
	formula_list::const_iterator it_sll;
	//Iterate on the \p to_print and print its content using ::to_print(const string_set&) seprated by a OR to represent a DNF.

	for (it_sll = to_print.begin(); it_sll != to_print.end(); it_sll++) {
		if (!first_print) {
			std::cout << " AND ";
		}
		first_print = false;
		it_sll->print();
	}
}

/*void printer::print_list(const kworld_ptr_set& to_print)
{
	bool first_print = true;
	kworld_ptr_set::const_iterator it_sll;
	//Iterate on the \p to_print and print its content using ::to_print(const string_set&) seprated by a OR to represent a DNF.

	for (it_sll = to_print.begin(); it_sll != to_print.end(); it_sll++) {
		if (!first_print) {
			std::cout << "\n";
		}
		first_print = false;
		std::cout << it_sll->get_id();
	}
}*/

void printer::print_list(const pworld_ptr_set& to_print) const
{
	bool first_print = true;
	pworld_ptr_set::const_iterator it_sll;
	//Iterate on the \p to_print and print its content using ::to_print(const string_set&) seprated by a OR to represent a DNF.

	for (it_sll = to_print.begin(); it_sll != to_print.end(); it_sll++) {
		if (!first_print) {
			std::cout << "\n";
		}
		first_print = false;
		std::cout << it_sll->get_id();
	}
}

void printer::print_list(const action_id_list& to_print) const
{
	bool first_print = true;
	action_id_list::const_iterator it_sll;
	//Iterate on the \p to_print and print its content using ::to_print(const string_set&) seprated by a OR to represent a DNF.

	for (it_sll = to_print.begin(); it_sll != to_print.end(); it_sll++) {
		if (!first_print) {
			std::cout << ", ";
		}
		first_print = false;

		if (setted_grounder) {
			std::cout << (m_grounder.deground_action(*it_sll));
		} else {
			std::cout << *it_sll;
		}
	}
}

void printer::print_list_ag(const agent_set& to_print) const
{
	bool first_print = true;
	agent_set::const_iterator it_sll;
	//Iterate on the \p to_print and print its content using ::to_print(const string_set&) seprated by a OR to represent a DNF.

	for (it_sll = to_print.begin(); it_sll != to_print.end(); it_sll++) {
		if (!first_print) {
			std::cout << ", ";
		}
		first_print = false;

		if (setted_grounder) {
			std::cout << (m_grounder.deground_agent(*it_sll));
		} else {
			std::cout << *it_sll;
		}
	}
}

void printer::print_map_att(const transition_map_att& to_print) const
{

	std::cout << "\n*******Printing the attitudes transition map*******\n";
	auto it_mapatt = to_print.begin();
	//Iterate on the \p to_print and print its content using ::to_print(const string_set&) seprated by a OR to represent a DNF.

	for (; it_mapatt != to_print.end(); it_mapatt++) {
		auto pair = it_mapatt->first;
		pworld_ptr ptr1 = pair.first.first;
		pworld_id id1 =  ptr1.get_id();
		
		pworld_ptr ptr2 = it_mapatt->second;
		pworld_id id2 =  ptr2.get_id();
		
		std::cout << "\t The world: " << id1;
		std::cout << " updated following " << pair.second;
		std::cout << " results in the world " << id2;
		std::cout << std::endl;
	}

	std::cout << "*********************DONE*********************\n\n";

}