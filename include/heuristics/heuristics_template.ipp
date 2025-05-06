
#include <map>
//#include "planning_graph_template.ipp"
#include "heuristics_manager.h"
//heuristic Manager
template <class T>
heuristics_manager::heuristics_manager(heuristics used_heur, const T & eState)
{
	set_used_h(used_heur);
	m_goals = domain::get_instance().get_goal_description();
	switch ( m_used_heur ) {
	case GNN:
		break;
	case L_PG:
	case S_PG:
	{
		expand_goals();
		break;

	}
	case C_PG:
	{
		planning_graph pg(eState);
		if (pg.is_satisfiable()) {
			m_fluents_score = pg.get_f_scores();
			m_bf_score = pg.get_bf_scores();

		} else {
			std::cout << "\n\nIt does not exists any Plan for this domain instance:(\n";
			exit(1);
		}
		break;
	}
	case SUBGOALS:
	{
		expand_goals();
		satisfied_goals::get_instance().set(m_goals);
		break;
	}
	default:
	{
		std::cerr << "\nWrong Heuristic Selection\n";
		exit(1);
	}

	}
}

template <class T>
void heuristics_manager::set_heuristic_value(T & eState)
{

	switch ( m_used_heur ) {
	case L_PG:
	{
		planning_graph pg(m_goals, eState);
		if (pg.is_satisfiable()) {
			eState.set_heuristic_value(pg.get_length());
		} else {
			eState.set_heuristic_value(-1);
		}
		break;
	}
	case S_PG:
	{
		planning_graph pg(m_goals, eState);
		if (pg.is_satisfiable()) {
			eState.set_heuristic_value(pg.get_sum());
		} else {
			eState.set_heuristic_value(-1);
		}
		break;
	}
	case C_PG:
	{
		short h_value = 0;

		for (auto it_pgf = m_fluents_score.begin(); it_pgf != m_fluents_score.end(); ++it_pgf) {
			if (eState.entails(it_pgf->first) && (it_pgf->second > 0)) {
				h_value += it_pgf->second;
			}
		}

		for (auto it_pgbf = m_bf_score.begin(); it_pgbf != m_bf_score.end(); ++it_pgbf) {
			if (eState.entails(it_pgbf->first)  && (it_pgbf->second > 0)) {
				h_value += it_pgbf->second;
			}
		}
		eState.set_heuristic_value(-h_value);
		break;
	}
	case SUBGOALS:
	{
		eState.set_heuristic_value(satisfied_goals::get_instance().get_unsatisfied_goals(eState));
		break;
	}
	case GNN:
	{
		eState.set_heuristic_value(get_gnn_score(eState));
		break;

	}
	default:
	{
		std::cerr << "\nWrong Heuristic Selection\n";
		exit(1);
	}

	}
}

template <class T>
unsigned short satisfied_goals::get_unsatisfied_goals(const T & eState) const
{
	unsigned short ret = m_goals_number;

	formula_list::const_iterator it_fl;
	for (it_fl = m_goals.begin(); it_fl != m_goals.end(); it_fl++) {
		if (eState.entails(*it_fl)) {
			ret--;
		}
	}
	return ret;

}

template <class T>
unsigned short satisfied_goals::get_satisfied_goals(const T & eState) const
{
	std::string dotFile = "out/state/ex_poss/signal_a_c_signal_c_b_open_a.dot";
    int number = 3;

    std::string command = "./run_python_script.sh " + dotFile + " " + std::to_string(number);
    system(command.c_str());

}


//@TODO fix this
template <class T>
unsigned short get_gnn_score(const T & eState)
{

	std::string graphviz_filename = eState.print_graphviz_ML_dataset(""); // Store the filename

	std::string command = "./lib/RL/run_python_script.sh " + graphviz_filename + " " + std::to_string(eState.get_plan_length());

    int ret = system(command.c_str()); // blocks until script (and Python) finishes

    if (ret != 0) {
        std::cerr << "Using GNN for heuristics failed with exit code: " << ret << std::endl;
    }

    std::ifstream infile("prediction.tmp");
    if (!infile) {
        std::cerr << "Failed to open prediction.tmp" << std::endl;
        return 1;
    }

    std::string line;
    unsigned short valueFromFile = 0;

    while (std::getline(infile, line)) {
        if (line.rfind("VALUE:", 0) == 0) { // line starts with "VALUE:"
            std::istringstream iss(line.substr(6)); // Skip "VALUE:"
            iss >> valueFromFile;
            break;
        }
    }

    infile.close();
	std::cout << "Current Value is " << std::to_string(valueFromFile) << std::endl;

	return valueFromFile;
}