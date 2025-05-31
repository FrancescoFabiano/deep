
#include <map>
//#include "planning_graph_template.ipp"
#include "HeuristicsManager.h"
template <class T>
unsigned short satisfied_goals::get_unsatisfied_goals(const T & eState) const
{
	unsigned short ret = m_goals_number;

	FormulaeList::const_iterator it_fl;
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