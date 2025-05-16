/**
 * \brief Class that implements a \ref belief_formula.h.
 *
 * \author Francesco Fabiano.
 * \date March 31, 2019
 */
#include "belief_formula.h"
#include "../domain/Domain.h"
#include "../utilities/printer.h"

belief_formula::belief_formula()
{
}

void belief_formula::set_from_ff(const fluent_formula & to_build)
{
	set_formula_type(FLUENT_FORMULA);
	set_fluent_formula(to_build);
	set_is_grounded(true);
	deground();
}

belief_formula::belief_formula(const belief_formula & to_copy)
{
	(*this) = to_copy;
}

bool belief_formula::get_is_grounded() const
{
	return m_is_grounded;
}

void belief_formula::set_is_grounded(bool is_grounded)
{
	m_is_grounded = is_grounded;
}

void belief_formula::set_formula_type(bf_type to_set)
{
	m_formula_type = to_set;
}

bf_type belief_formula::get_formula_type() const
{
	if (m_formula_type == BF_TYPE_FAIL) {
		std::cerr << "\nError in reading a belief_formula ";
		std::cerr << "(bf_type not setted properly).";
		std::cerr << std::endl;
		exit(1);
	}
	return m_formula_type;
}

void belief_formula::set_fluent_formula(const fluent_formula & to_set)
{
	if (to_set.size() < 1) {
		std::cerr << "\nError in declaring a belief_formula ";
		std::cerr << "there must be at least one fluent in a formula.";
		std::cerr << std::endl;
		exit(1);
	}
	m_fluent_formula = to_set;
}

void belief_formula::set_fluent_formula_from_fluent(Fluent to_set)
{
	fluent_set tmp;
	tmp.insert(to_set);
	m_fluent_formula.insert(tmp);
}

const fluent_formula & belief_formula::get_fluent_formula() const
{
	if (m_fluent_formula.size() < 1) {
		std::cerr << "\nError in reading a belief_formula, it must be grounded ";
		std::cerr << "(fluent_formula not grounded).";
		std::cerr << std::endl;
		exit(1);
	}
	return m_fluent_formula;
}

void belief_formula::set_agent(Agent to_set)
{
	m_agent = to_set;
}

Agent belief_formula::get_agent() const
{
	if (!m_is_grounded) {
		std::cerr << "\nError in reading a belief_formula, it must be grounded";
		std::cerr << "(agent not grounded).";
		std::cerr << std::endl;
		exit(1);
	}
	return m_agent;
}

const belief_formula & belief_formula::get_bf1() const
{
	if (m_bf1 == nullptr) {
		std::cerr << "\nError in declaring a belief_formula ";
		std::cerr << "a nested belief formula has not been declared.";
		std::cerr << std::endl;
		exit(1);
	}
	return *m_bf1;
}

const belief_formula & belief_formula::get_bf2() const
{
	if (m_bf2 == nullptr) {
		std::cerr << "\nError in declaring a belief_formula ";
		std::cerr << "a second nested belief formula has not been declared.";
		std::cerr << std::endl;
		exit(1);
	}
	return *m_bf2;
}

void belief_formula::set_operator(bf_operator to_set)
{
	m_operator = to_set;
}

bf_operator belief_formula::get_operator() const
{
	if (m_formula_type == BF_TYPE_FAIL) {
		std::cerr << "\nError in reading a belief_formula";
		std::cerr << "(bf_operator not setted properly).";
		std::cerr << std::endl;
		exit(1);
	}
	return m_operator;
}

void belief_formula::set_group_agents(const AgentSet & to_set)
{
	if (to_set.size() < 1) {
		std::cerr << "\nError in declaring a belief_formula ";
		std::cerr << "there must be at least one agent for group formulae.";
		std::cerr << std::endl;
		exit(1);
	}
	m_group_agents = to_set;
}

const AgentSet & belief_formula::get_group_agents() const
{
	if (!m_is_grounded) {
		std::cerr << "\nError in reading a belief_formula, it must be grounded";
		std::cerr << " (agent_group not grounded).";
		std::cerr << std::endl;
		exit(1);
	}

	return m_group_agents;
}

const string_set_set & belief_formula::get_string_fluent_formula() const
{
	return m_string_fluent_formula;
}

const std::string & belief_formula::get_string_agent() const
{
	return m_string_agent;
}

const string_set & belief_formula::get_string_group_agents() const
{
	return m_string_group_agents;
}

void belief_formula::set_string_fluent_formula(const string_set_set & to_set)
{
	/*if (to_set.size() < 0) {
		std::cerr << "\nError in reading a belief_formula,";
		std::cerr << " the string formula cannot be empty.";
		std::cerr << std::endl;
		exit(1);
	}*/
	m_string_fluent_formula = to_set;

}

void belief_formula::set_string_agent(std::string to_set)
{
	m_string_agent = to_set;

}

void belief_formula::set_string_group_agents(const string_set & to_set)
{
	if (to_set.size() < 1) {
		std::cerr << "\nError in reading a belief_formula";
		std::cerr << ": the set of agents cannot be empty.";
		std::cerr << std::endl;
		exit(1);
	}
	m_string_group_agents = to_set;
}

void belief_formula::set_bf1(const belief_formula & to_set)
{
	m_bf1 = std::make_shared<belief_formula>(to_set);
	//m_bf1 = std::shared_ptr<belief_formula>(to_set);
}

void belief_formula::set_bf2(const belief_formula & to_set)
{
	m_bf2 = std::make_shared<belief_formula>(to_set);
	//m_bf2 = std::shared_ptr<belief_formula>(&to_set);
}

void belief_formula::print() const
{

	/*if (!m_is_grounded) {
		std::cerr << "\nError in reading a belief_formula, it must be grounded";
		std::cerr << std::endl;
		exit(1);
	}*/

	switch ( m_formula_type ) {

	case FLUENT_FORMULA:
		printer::get_instance().print_list(m_fluent_formula);
		break;
	case BELIEF_FORMULA:
		std::cout << "B(" << domain::get_instance().get_grounder().deground_agent(m_agent) << ",(";
		m_bf1->print();
		std::cout << "))";
		break;

	case D_FORMULA:
		std::cout << "D([";
		printer::get_instance().print_list_ag(m_group_agents);
		std::cout << "],";
		m_bf1->print();
		std::cout << ")";
		break;

	case C_FORMULA:
		std::cout << "C([";
		printer::get_instance().print_list_ag(m_group_agents);
		std::cout << "],";
		m_bf1->print();
		std::cout << ")";
		break;

	case E_FORMULA:
		std::cout << "E([";
		printer::get_instance().print_list_ag(m_group_agents);
		std::cout << "],";
		m_bf1->print();
		std::cout << ")";
		std::cout << ")";
		break;

	case PROPOSITIONAL_FORMULA:
		if (m_operator == BF_NOT)
			std::cout << "NOT(";
		m_bf1->print();
		if (m_operator == BF_NOT)
			std::cout << ")";
		if (m_operator == BF_AND)
			std::cout << " AND ";
		if (m_operator == BF_OR)
			std::cout << " OR ";
		else if (m_operator == BF_FAIL) {
			std::cerr << "\n ERROR IN DELCARATION\n.";
			exit(1);
		}
		if (m_bf2 != nullptr) {
			//std::cout << "(";
			m_bf2->print();
		}
		//std::cout << ")";
		break;
	case BF_EMPTY:
		std::cout << "Empty\n";
		break;
	case BF_TYPE_FAIL:
	default:
		std::cerr << "\n Unknown belief_formula type.";
		exit(1);
		break;

	}
};

void belief_formula::print_deground()
{

	ground();
	m_is_grounded = true;

	print();
};

/*void belief_formula::print_grounded(const grounder& grounder) const
{

	bool first = true;
	string_set::const_iterator it;
	string_set_set::const_iterator it2;
	bool first2 = true;
	std::cout << "(";
	switch ( m_formula_type ) {

	case FLUENT_FORMULA:
		for (it2 = m_string_fluent_formula.begin(); it2 != m_string_fluent_formula.end(); it2++) {
			if (!first2)
				std::cout << " | ";
			first2 = false;
			first = true;
			std::cout << "(";
			for (it = it2->begin(); it != it2->end(); it++) {
				if (!first)
					std::cout << " AND ";
				first = false;
				std::cout << grounder.ground_fluent(*it);
			}
			std::cout << ")";
		}
		std::cout << ")";
		break;

	case BELIEF_FORMULA:
		std::cout << "B(" << grounder.ground_agent(m_string_agent) << ",";
		m_bf1->print_grounded(grounder);
		std::cout << ")";
		std::cout << ")";
		break;

	case C_FORMULA:
		std::cout << "C(";
		for (it = m_string_group_agents.begin(); it != m_string_group_agents.end(); it++) {
			if (!first)
				std::cout << ",";
			first = false;
			std::cout << grounder.ground_agent(*it);
		}
		std::cout << ",";
		m_bf1->print_grounded(grounder);
		std::cout << ")";
		std::cout << ")";
		break;

	case E_FORMULA:
		std::cout << "E(";
		for (it = m_string_group_agents.begin(); it != m_string_group_agents.end(); it++) {
			if (!first)
				std::cout << ",";
			first = false;
			std::cout << grounder.ground_agent(*it);
		}
		std::cout << ",";
		m_bf1->print_grounded(grounder);
		std::cout << ")";
		std::cout << ")";
		break;

	case PROPOSITIONAL_FORMULA:
		if (m_operator == BF_NOT)
			std::cout << " NOT ";
		else if (m_operator == BF_FAIL)
			std::cout << "(";
		m_bf1->print_grounded(grounder);
		if (m_operator == BF_FAIL)
			std::cout << ")";

		if (m_operator == BF_AND)
			std::cout << " & ";
		else if (m_operator == BF_OR)
			std::cout << " | ";
		if (m_bf2 != nullptr)
			m_bf2->print_grounded(grounder);
		std::cout << ")";
		break;

	case BF_TYPE_FAIL: //Static
		std::cerr << "\n Unknown belief_formula type.";
		exit(1);
		break;

	default:
		std::cerr << "\n Unknown belief_formula type.";
		exit(1);
		break;

	} //switch
}; print*/

void belief_formula::ground()
{

	//std::cout << "\nDEBUG: ground bf..." << std::endl;

	grounder gr = domain::get_instance().get_grounder();
	if (!m_is_grounded) {
		switch ( m_formula_type ) {

		case FLUENT_FORMULA:

			set_fluent_formula(gr.ground_fluent(get_string_fluent_formula()));
			//m_fluent_formula = gr.ground_fluent(m_string_fluent_formula);
			break;

		case BELIEF_FORMULA:

			set_agent(gr.ground_agent(get_string_agent()));
			//m_agent = gr.ground_agent(m_string_agent_op);
			m_bf1->ground();
			break;

		case E_FORMULA:
		case C_FORMULA:
		case D_FORMULA:
			set_group_agents(gr.ground_agent(get_string_group_agents()));
			//m_group_agents = gr.ground_agent(m_string_group_agents);
			m_bf1->ground();
			break;

		case PROPOSITIONAL_FORMULA:
			m_bf1->ground();
			if (m_operator == BF_AND || m_operator == BF_OR) {
				m_bf2->ground();
			}
			break;
		case BF_EMPTY:
			break;
		case BF_TYPE_FAIL:
		default:
			std::cerr << "\n Unknown belief_formula type.";
			exit(1);
			break;
		}
	}
	m_is_grounded = true;

}

void belief_formula::deground()
{

	//std::cout << "\nDEBUG: ground bf..." << std::endl;

	grounder gr = domain::get_instance().get_grounder();
	switch ( m_formula_type ) {

	case FLUENT_FORMULA:

		set_string_fluent_formula(gr.deground_fluent(get_fluent_formula()));
		//m_fluent_formula = gr.ground_fluent(m_string_fluent_formula);
		break;

	case BELIEF_FORMULA:

		set_string_agent(gr.deground_agent(get_agent()));
		//m_agent = gr.ground_agent(m_string_agent_op);
		m_bf1->deground();
		break;

	case E_FORMULA:
	case C_FORMULA:
	case D_FORMULA:
		set_string_group_agents(gr.deground_agents(get_group_agents()));
		//m_group_agents = gr.ground_agent(m_string_group_agents);
		m_bf1->deground();
		break;

	case PROPOSITIONAL_FORMULA:
		m_bf1->deground();
		if (m_operator == BF_AND || m_operator == BF_OR) {
			m_bf2->deground();
		}
		break;
	case BF_EMPTY:
		break;
	case BF_TYPE_FAIL:
	default:
		std::cerr << "\n Unknown belief_formula type.";
		exit(1);
		break;
	}
}

bool belief_formula::operator==(const belief_formula & to_compare) const
{

	if (m_is_grounded == to_compare.get_is_grounded()) {
		if (m_formula_type == to_compare.m_formula_type) {

			switch ( m_formula_type ) {

			case FLUENT_FORMULA:
			{
				return(m_fluent_formula == to_compare.get_fluent_formula());
				break;
			}

			case BELIEF_FORMULA:
			{
				if (m_agent == to_compare.get_agent()) {
					return get_bf1() == to_compare.get_bf1();
				} else {
					return false;
				}
				break;
			}

			case PROPOSITIONAL_FORMULA:
			{
				if (m_operator == to_compare.get_operator()) {
					switch ( m_operator ) {
					case BF_NOT:
					{
						return get_bf1() == to_compare.get_bf1();
						break;
					}
					case BF_AND:
					case BF_OR:
					{
						if (get_bf1() == to_compare.get_bf1()) {
							return get_bf2() == to_compare.get_bf2();
						} else if (get_bf1() == to_compare.get_bf2()) {
							return get_bf2() == to_compare.get_bf1();
						} else {
							return false;
						}
						break;
					}
					default:
						return false;
						break;
					}
				} else {
					return false;
				}
				break;
			}
			case E_FORMULA:
			case C_FORMULA:
			case D_FORMULA:
			{
				if (m_group_agents == to_compare.get_group_agents()) {
					return get_bf1() == to_compare.get_bf1();
				}
				break;
			}
			case BF_EMPTY:
			{
				break;
			}
			default:
			{
				std::cerr << "\n Unknown belief_formula type.";
				exit(1);
				break;
			}
			}
		}
	}
	return false;
}

bool belief_formula::operator=(const belief_formula & to_copy)
{
	set_formula_type(to_copy.get_formula_type());

	m_is_grounded = to_copy.get_is_grounded();
	switch ( m_formula_type ) {
	case FLUENT_FORMULA:
	{
		//If has been grounded
		if (m_is_grounded) {
			set_fluent_formula(to_copy.get_fluent_formula());
		}
		set_string_fluent_formula(to_copy.get_string_fluent_formula());
		break;
	}
	case BELIEF_FORMULA:
	{

		//If has been grounded
		if (m_is_grounded) {
			set_agent(to_copy.get_agent());
		}
		set_string_agent(to_copy.get_string_agent());
		set_bf1(to_copy.get_bf1());
		break;
	}
	case PROPOSITIONAL_FORMULA:
	{
		set_operator(to_copy.get_operator());
		set_bf1(to_copy.get_bf1());
		switch ( m_operator ) {
		case BF_AND:
		case BF_OR:
		{
			set_bf2(to_copy.get_bf2());
			break;
		}
		case BF_NOT:
		{
			break;
		}
			//REMOVE EVERY INPAREN
		case BF_INPAREN:
		{
			*this = to_copy.get_bf1();
			break;
		}
		case BF_FAIL:
		default:
		{
			std::cerr << "\nError in copying a belief_formula ";
			std::cerr << std::endl;
			exit(1);
			break;
		}
		}
		break;

	}
	case E_FORMULA:
	case C_FORMULA:
	case D_FORMULA:
	{
		//If has been grounded
		if (m_is_grounded) {
			set_group_agents(to_copy.get_group_agents());
		}
		set_string_group_agents(to_copy.get_string_group_agents());
		set_bf1(to_copy.get_bf1());
		break;
	}

	case BF_EMPTY:
		break;

	case BF_TYPE_FAIL:
	default:
	{
		std::cerr << "\nError in copying a belief_formula ";
		std::cerr << std::endl;
		exit(1);
		break;
	}
	}

	return true;
}

bool belief_formula::operator<(const belief_formula& to_compare) const
{

	if (get_formula_type() < to_compare.get_formula_type()) {
		return true;
	} else if (get_formula_type() == to_compare.get_formula_type()) {
		switch ( m_formula_type ) {
		case FLUENT_FORMULA:
		{
			if (get_fluent_formula() < to_compare.get_fluent_formula()) {
				return true;
			}

			break;
		}
		case BELIEF_FORMULA:
		{

			//If has been grounded
			if (get_agent() < to_compare.get_agent()) {
				return true;
			} else if (get_agent() == to_compare.get_agent()) {
				return(get_bf1() < to_compare.get_bf1());
			}
			break;
		}
		case PROPOSITIONAL_FORMULA:
		{
			if (get_operator() < to_compare.get_operator()) {
				return true;
			} else if (get_operator() == to_compare.get_operator()) {
				switch ( m_operator ) {
				case BF_AND:
				case BF_OR:
				{
					if (get_bf1() < to_compare.get_bf1()) {
						return true;
					} else if (get_bf1() == to_compare.get_bf1()) {
						return(get_bf2() < to_compare.get_bf2());
					}
					break;
				}
				case BF_NOT:
				{
					return(get_bf1() < to_compare.get_bf1());
					break;

				}
					//REMOVE EVERY INPAREN
				case BF_INPAREN:
				case BF_FAIL:
				default:
				{
					std::cerr << "\nError in comparing belief_formulae";
					std::cerr << std::endl;
					exit(1);
					break;
				}
				}
			}
			break;
		}
		case E_FORMULA:
		case C_FORMULA:
		case D_FORMULA:
		{
			//If has been grounded
			if (get_group_agents() < to_compare.get_group_agents()) {
				return true;
			} else if (get_group_agents() == to_compare.get_group_agents()) {
				return(get_bf1() < to_compare.get_bf1());
			}
			break;
		}
		case BF_EMPTY:
			/*Maybe
			 return true;
			 */
		case BF_TYPE_FAIL:
		default:
		{
			std::cerr << "\nError in comparing belief_formulae";
			std::cerr << std::endl;
			exit(1);
			break;
		}
		}
	}
	return false;
}