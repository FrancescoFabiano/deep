/**
 * \brief Implementation of \ref domain.h.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date April 1, 2019
 */
#include "domain.h"
#include <boost/dynamic_bitset.hpp>

domain::domain()
{
}

domain& domain::get_instance()
{
	static domain instance;
	return instance;
}

void domain::set_domain(std::string name, bool debug, state_type stype, bool k_opt, boost::shared_ptr<reader> reader, domain_restriction ini_res, domain_restriction goal_res, bool is_global_obsv, action_check act_check, bool check_visited, bis_type bisimulation)
{
	m_name = name;
	m_debug = debug;
	m_stype = stype;
	m_kopt = k_opt;
	m_reader = reader;
	m_intial_description = initially(ini_res);
	m_goal_restriction = goal_res;
	m_is_global_obsv = is_global_obsv;
	m_act_check = act_check;
	m_check_visited = check_visited;
	m_bisimulation = bisimulation;
}

const state_type & domain::get_stype()
{
	return m_stype;
}

const bool & domain::get_k_optimized()
{
	return m_kopt;
}

const grounder & domain::get_grounder()
{
	return m_grounder;
}

const fluent_set & domain::get_fluents()
{
	return m_fluents;
}

unsigned int domain::get_fluent_number()
{
	return(m_fluents.size() / 2);
}

unsigned int domain::get_size_fluent()
{
	auto fluent_first = m_fluents.begin();
	return(fluent_first->size());
}

const action_set & domain::get_actions()
{
	return m_actions;
}


const agent_set & domain::get_agents()
{
	return m_agents;
}

bool domain::get_is_global_obsv()
{
	return m_is_global_obsv;
}

bool domain::get_debug()
{
	return m_debug;
}

bool domain::check_visited()
{
	return m_check_visited;
}

std::string domain::get_name()
{
	return m_name;
}

action_check domain::get_act_check()
{
	return m_act_check;
}

const initially & domain::get_initial_description()
{
	return m_intial_description;
}

domain_restriction domain::get_goal_restriction()
{
	return m_goal_restriction;
}

const formula_list & domain::get_goal_description()
{
	return m_goal_description;
}

bis_type domain::get_bisimulation()
{
	return m_bisimulation;
}

void domain::build()
{
	//std::cout << "Agenti" <<std::endl;
	build_agents();
	//std::cout << "Fluenti" <<std::endl;
	build_fluents();
	// std::cout << "Azioni" <<std::endl;
	build_actions();
	//std::cout << "Initilly" <<std::endl;
	build_initially();
	// std::cout << "Goal" <<std::endl;
	build_goal();
}

void domain::build_agents()
{
	/*
	 * This function set the grounder agent map with the correct values.
	 */
	agent_map domain_agent_map;
	std::cout << "\nBuilding agent list..." << std::endl;
	string_set::const_iterator it_agents;
	int i = 0;

	//ottengo il numero degli agenti e inizio a generare un bitset dinamico con n bit
	//posizione x se bit a 1 esiste un agente altrimenti 0.
	int agents_length = helper::lenght_to_power_two(m_reader->m_agents.size());

	for (it_agents = m_reader->m_agents.begin(); it_agents != m_reader->m_agents.end(); it_agents++) {
		boost::dynamic_bitset<> agent(agents_length, i);
		domain_agent_map.insert(agent_map::value_type(*it_agents, agent));
		m_agents.insert(agent);
		if (m_debug) {
			std::cout << "Agent " << *it_agents << " is " << agent << std::endl;

		}
		i++;
	}
	m_grounder.set_agent_map(domain_agent_map);
}

void domain::build_fluents()
{
	/*
	 * This function set the grounder fluent map with the correct values.
	 */
	fluent_map domain_fluent_map;
	std::cout << "\nBuilding fluent literals..." << std::endl;
	string_set::const_iterator it_fluents;
	int i = 0;
	int bit_size = helper::lenght_to_power_two(m_reader->m_fluents.size());
	//todo prende numero fluenti*2 e generare i bit necessari
	for (it_fluents = m_reader->m_fluents.begin();
		it_fluents != m_reader->m_fluents.end(); it_fluents++) {
		boost::dynamic_bitset<> fluentReal(bit_size + 1, i);
		fluentReal.set(fluentReal.size() - 1, 0);
		domain_fluent_map.insert(fluent_map::value_type(*it_fluents, fluentReal));
		m_fluents.insert(fluentReal);

		if (m_debug) {
			std::cout << "Literal " << *it_fluents << " is " << " " << fluentReal << std::endl;
		}

		boost::dynamic_bitset<> fluent_negate_real(bit_size + 1, i);
		fluent_negate_real.set(fluent_negate_real.size() - 1, 1);
		domain_fluent_map.insert(fluent_map::value_type(NEGATION_SYMBOL + *it_fluents, fluent_negate_real));
		m_fluents.insert(fluent_negate_real);
		i++;
		if (m_debug) {
			std::cout << "Literal not " << *it_fluents << " is " << (i - 1) << " " << fluent_negate_real << std::endl;
		}
	}
	m_grounder.set_fluent_map(domain_fluent_map);
}

void domain::build_actions()
{

	/*
	 * This function set the grounder action map with the correct values.
	 */
	action_name_map domain_action_name_map;
	std::cout << "\nBuilding action list..." << std::endl;
	string_set::const_iterator it_actions_name;
	int i = 0;
	int numberOfActions = m_reader->m_actions.size();
	int bit_size = helper::lenght_to_power_two(numberOfActions);
	for (it_actions_name = m_reader->m_actions.begin();
		it_actions_name != m_reader->m_actions.end(); it_actions_name++) {
		boost::dynamic_bitset<> action_bitset(bit_size, i);
		action tmp_action(*it_actions_name, action_bitset);
		domain_action_name_map.insert(action_name_map::value_type(*it_actions_name, action_bitset));
		i++;
		m_actions.insert(tmp_action);
		if (m_debug) {
			std::cout << "Action " << tmp_action.get_name() << " is " << tmp_action.get_id() << std::endl;
		}
	}

	m_grounder.set_action_name_map(domain_action_name_map);
	printer::get_instance().set_grounder(m_grounder);

	build_propositions();

	if (m_debug) {
		std::cout << "\nPrinting complete action list..." << std::endl;
		action_set::const_iterator it_actions;

		for (it_actions = m_actions.begin();
			it_actions != m_actions.end(); it_actions++) {
			it_actions->print();
		}
	}

}

void domain::build_propositions()
{
	//Adding propositions to actions list
	std::cout << "\nAdding propositions to actions..." << std::endl;
	proposition_list::iterator it_prop;
	action_id action_to_modify;
	for (it_prop = m_reader->m_propositions.begin();
		it_prop != m_reader->m_propositions.end(); it_prop++) {


		action_to_modify = m_grounder.ground_action(it_prop->get_action_name());

		//To change remove and add the updated --> @TODO: find better like queue

		for (std::set<action>::iterator actionsList = m_actions.begin();
			actionsList != m_actions.end(); actionsList++) {
			action_id actionTemp = action_to_modify;
			actionTemp.set(actionTemp.size() - 1, 0);
			if (m_actions.size() > actionTemp.to_ulong() && actionsList->get_id() == action_to_modify) {

				action_set::iterator it_action_set = actionsList;
				action tmp = *it_action_set;
				tmp.add_proposition(*it_prop);
				m_actions.erase(it_action_set);
				m_actions.insert(tmp);
				break;
			}
		}

		/*
		 *
		 *
		 * 	for (it_prop = m_reader->m_propositions.begin();
			it_prop != m_reader->m_propositions.end(); it_prop++) {

			action_to_modify = m_grounder.ground_action(it_prop->get_action_name());
			if (m_actions.size() > action_to_modify) {
				//To change remove and add the updated --> @TODO: find better like queue
				action_set::iterator it_action_set = std::next(m_actions.begin(), action_to_modify);
				action tmp = *it_action_set;
				tmp.add_proposition(*it_prop);
				m_actions.erase(it_action_set);
				m_actions.insert(tmp);
			}
		}
		 * */
	}
}


void domain::build_initially()
{
	std::cout << "\nAdding to pointed world and initial conditions..." << std::endl;
	formula_list::iterator it_fl;

	for (it_fl = m_reader->m_bf_initially.begin(); it_fl != m_reader->m_bf_initially.end(); it_fl++) {

		it_fl->ground();

		switch ( it_fl->get_formula_type() ) //initially phi
		{
			//S5 -> pointed world
		case FLUENT_FORMULA:
		{
			m_intial_description.add_pointed_condition(it_fl->get_fluent_formula());

			if (m_debug) {
				std::cout << "	Pointed world: ";
				printer::get_instance().print_list(it_fl->get_fluent_formula());
				std::cout << std::endl;
			}

			break;
		}
			//S5 -> Edge
		case C_FORMULA:
			//No more S5
		case D_FORMULA:
			//Possible S5 -> initally [C(phi2), C(phi2)]
		case PROPOSITIONAL_FORMULA:
			//No more S5
		case BELIEF_FORMULA:
			//No more S5
		case E_FORMULA:
		{
			m_intial_description.add_initial_condition(*it_fl);
			if (m_debug) {
				std::cout << "Added to initial conditions: ";
				it_fl->print();
				std::cout << std::endl;
			}
			break;
		}
		case BF_EMPTY:
			break;
		default:
			std::cerr << "Error in the 'initially' conditions." << std::endl;
			exit(1);
			break;
		}
	}


	//Given the initial restriction the initial state might needs different function
	switch ( m_intial_description.get_ini_restriction() ) {
	case S5:
	{
		m_intial_description.set_ff_forS5();
		break;
	}
	case K45:
	{
		break;
	}
	case NONE:
	{
		break;
	}
	default:
	{
		break;
	}
	}

}

void domain::build_goal()
{
	std::cout << "\nAdding to Goal..." << std::endl;


	formula_list::iterator it_fl;

	for (it_fl = (m_reader->m_bf_goal).begin(); it_fl != (m_reader->m_bf_goal).end(); it_fl++) {
		if (check_goal_restriction(*it_fl)) {
			it_fl->ground();
			m_goal_description.push_back(*it_fl);
			if (m_debug) {
				std::cout << "	";
				it_fl->print();
				std::cout << std::endl;

			}
		} else {
			std::cerr << "The Goal does not respect the conditions\n";
			exit(1);
		}
	}
}

bool domain::check_goal_restriction(const belief_formula& bf)//Apply the restriction
{
	/** \todo: Maybe a separated class?*/
	bool ret = false;
	switch ( m_goal_restriction ) {
		//We only admit non negative goals
	case NONEG:
		switch ( bf.get_formula_type() ) {
		case FLUENT_FORMULA:
			ret = true;
			break;
		case BELIEF_FORMULA:
			ret = true;
			break;
		case D_FORMULA:
		case C_FORMULA:
		case E_FORMULA:
			ret = check_goal_restriction(bf.get_bf1());
			break;
		case PROPOSITIONAL_FORMULA:
			if (bf.get_operator() == BF_NOT) {
				ret = false;
			} else {
				ret = check_goal_restriction(bf.get_bf1()) && check_goal_restriction(bf.get_bf2());
			}
			break;
		default:
			break;
		}
		break;
	case NONE:
		ret = true;
		break;
	default: /* static */
		ret = false;
		break;
	}

	return ret;

}



/*std::string domain::get_name();
formula_list domain::get_initial_description();
formula_list domain::get_goal_description();

void domain::set_name(std::string domain_name);
//@TODO: setter properly and constructors
void domain::set_initial_description();
void domain::set_goal_description();*/
