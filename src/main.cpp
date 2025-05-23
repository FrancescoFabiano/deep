/*
 * \brief The main file.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 12, 2025
 */


#include <iostream>
#include <boost/make_shared.hpp>

#include "argparse/ArgumentParser.h"
#include "domain/Domain.h"
//#include "../src/search/planner.ipp"


/*void launch_search(state_type state_struc, bool execute_given_action, bool results_file, parallel_input pin, heuristics used_heur, search_type used_search, ML_Dataset_Params ML_dataset, std::vector<std::string> given_actions, short max_depth, short step)
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
}*/


int main(int argc, char** argv)
{


	ArgumentParser::create_instance(argc, argv);
	auto output = ArgumentParser::get_instance().get_log_enabled() ? ArgumentParser::get_instance().get_log_file_path() : std::cout;
	Domain::create_instance(output);

	//you can run a single search or use portfolio search
	//Understand if portfolio needs to be achieved through an extenral script (massive overhead for parsing and launching)
	//or if it can be done through the planner itself. maybe vreate a new class for this (which also set the various parameter of a search)
	//this way we can create initla (stored somewhere, find a way to address this) once and then use it for all the search, as well as creaing the wrods etc (make the worlds pointer accessible by all threads?. This imply generation and then readonly)
	return 0;

	/////@TODO Fix: generate_domain(argv);

	//launch search planner
	/////@TODO Fix: launch_search(state_struc, execute_given_actions, results_file, pin, used_heur, used_search, ML_dataset, given_actions, max_depth, step);

	//timer.end(READ_TIMER);
	//planner.main();

	exit(0);
}

