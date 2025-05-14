/*
 * \brief The main file.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date Aprile 1, 2019
 */


#include <iostream>
#include <stdio.h>
#include <vector>

#include <boost/make_shared.hpp>

#include "../src/argparse/ArgumentParser.h"
#include "../src/search/planner.ipp"


void launch_search(state_type state_struc, bool execute_given_action, bool results_file, parallel_input pin, heuristics used_heur, search_type used_search, ML_Dataset_Params ML_dataset, std::vector<std::string> given_actions, short max_depth, short step)
{
	switch ( state_struc ) {
	case POSSIBILITIES:
	{
		planner< state<pstate> > m_planner;
		if (execute_given_action) {
			if (results_file) {
				m_planner.execute_given_actions_timed(given_actions,pin.ptype);
			} else {
				m_planner.execute_given_actions(given_actions);
			}
			std::cout << "\n\n\n*****THE END*****\n";
		} else {
			if (m_planner.search(results_file, pin, used_heur, used_search, ML_dataset, max_depth, step)) {
				std::cout << "\n\n\n*****THE END*****\n";
			} else {
				std::cout << "\n\n\n*****THE SAD END*****\n";
			}
		}
		break;
	}
	default:
		std::cerr << "\nNot implemented yet - 0\n";
		exit(1);
	}
}


/////@TODO This will be replaced by epddl parser
reader generator()
{
	return reader();
}

boost::shared_ptr<reader> domain_reader = boost::make_shared<reader>(generator());
void generate_domain(char** argv)
{
	std::string domain_name = argv[1];
	if (freopen(argv[1], "r", stdin) == NULL) {

		std::cerr << argv[0] << ": File " << domain_name << " cannot be opened.\n";
		exit(1);
	}

	domain_name = domain_name.substr(domain_name.find_last_of("\\/") + 1);
	domain_name = domain_name.substr(0, domain_name.find_last_of("."));


	//timer.start(READ_TIMER);
	domain_reader->read();
	//	if (debug) {
	//		domain_reader->print();
	//	}
	//Domain building

	domain::get_instance().build();
}

int main(int argc, char** argv)
{

	// Parse command-line arguments
	ArgumentParser::create_instance(argc, argv);

	//Creates domain calling parsing and using ArgumentParser
	domain::createInstance();

	/////@TODO Fix: generate_domain(argv);

	//launch search planner
	/////@TODO Fix: launch_search(state_struc, execute_given_actions, results_file, pin, used_heur, used_search, ML_dataset, given_actions, max_depth, step);

	//timer.end(READ_TIMER);
	//planner.main();

	exit(0);
}

