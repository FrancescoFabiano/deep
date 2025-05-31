/*
 * \brief Class that implements PlanningGraph.h.
 *
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 31, 2025
 */

#include "PlanningGraph.h"

/*\******************************************************************************************************************/

/*\*****START PLANNING GRAPH TIME MEASURE*******
#include <chrono>
std::chrono::duration<double> t1, t2, t3, t4;

*\******END PLANNING GRAPH TIME MEASURE********/

PlanningGraph::PlanningGraph()
{
	auto goals = Domain::get_instance().get_goal_description();
	StateLevel pg_init;
	pg_init.initialize(goals);
	init(goals, pg_init);
}

PlanningGraph::PlanningGraph(const PlanningGraph & pg)
{
	set_pg(pg);
}

PlanningGraph::PlanningGraph(const FormulaeList & goal)
{
	StateLevel pg_init;
	pg_init.initialize(goal);
	init(goal, pg_init);
}

void PlanningGraph::init(const FormulaeList & goal, const StateLevel & pg_init)
{
	/*\*****START PLANNING GRAPH TIME MEASURE*******
	auto start_pg_build = std::chrono::system_clock::now();
	os << "start" << std::endl;
	*\******END PLANNING GRAPH TIME MEASURE********/


	set_goal(goal);
	m_state_levels.push_back(pg_init);
	auto init_bf_score = pg_init.get_bf_map();
	for (auto it_pgbf = init_bf_score.begin(); it_pgbf != init_bf_score.end(); ++it_pgbf) {
		if (it_pgbf->second < 0) {
			m_belief_formula_false.insert(it_pgbf->first);
		}
	}

	m_never_executed = Domain::get_instance().get_actions();
	set_length(0);
	set_sum(0);

	/*\*****START PLANNING GRAPH TIME MEASURE*******
	auto start_pg_goal_ini = std::chrono::system_clock::now();
	*\******END PLANNING GRAPH TIME MEASURE********/


	bool not_goal = false;
	for (auto it_fl = m_goal.begin(); it_fl != m_goal.end();) {
		if (m_state_levels.back().pg_entailment(*it_fl)) {
			it_fl = m_goal.erase(it_fl);
		} else {
			not_goal = true;
			it_fl++;
		}
	}

	/*\*****START PLANNING GRAPH TIME MEASURE*******
	auto end_pg_goal_ini = std::chrono::system_clock::now();
	std::chrono::duration<double> pg_goal_ini_time = end_pg_goal_ini - start_pg_goal_ini;
	t1 = std::chrono::milliseconds::zero();
	t2 = std::chrono::milliseconds::zero();
	t3 = std::chrono::milliseconds::zero();
	t4 = std::chrono::milliseconds::zero();
	*\******END PLANNING GRAPH TIME MEASURE********/

	if (not_goal) {
		//	if (!is_single) {
		pg_build();
		//	} else {
		//		pg_build_initially(goal);
		//	}
	} else {

		set_satisfiable(true);
		std::cerr << "\nBUILDING: The initial state is goal\n";
		exit(1);
	}

	/*\*****START PLANNING GRAPH TIME MEASURE*******
	auto end_pg_build = std::chrono::system_clock::now();
	std::chrono::duration<double> pg_build_time = end_pg_build - start_pg_build;
	os << "\n\nGenerated Planning Graph of length " << get_length() << " in " << pg_build_time.count() << " seconds of which:";
	os << "\nFirst goal check:      " << pg_goal_ini_time.count();
	os << "\nAction Level creation: " << t1.count();
	os << "\n\nState Level Creation:  " << t2.count() << " of which:";
	os << "\nActions Execution:     " << t4.count();
	os << "\n\nGoals Check:           " << t3.count() << std::endl;
	*\******END PLANNING GRAPH TIME MEASURE********/


}

void PlanningGraph::set_satisfiable(bool sat)
{

	m_satisfiable = sat;
}

bool PlanningGraph::is_satisfiable() const
{

	return m_satisfiable;
}

void PlanningGraph::pg_build()
{
	StateLevel s_level_curr;
	s_level_curr = m_state_levels.back();
	ActionLevel a_level_curr;
	a_level_curr.set_depth(get_length());

	/*\*****START PLANNING GRAPH TIME MEASURE*******
	auto start = std::chrono::system_clock::now();
	*\******END PLANNING GRAPH TIME MEASURE********/

	if (m_action_levels.size() > 0) {
		a_level_curr = m_action_levels.back();
	}
	ActionsSet::iterator it_actset;
	for (it_actset = m_never_executed.begin(); it_actset != m_never_executed.end();) {
		if (s_level_curr.pg_executable(*it_actset)) {
			a_level_curr.add_action(*it_actset);
			it_actset = m_never_executed.erase(it_actset);
		} else {
			it_actset++;
		}
	}
	/*\*****START PLANNING GRAPH TIME MEASURE*******
	auto end = std::chrono::system_clock::now();
	t1 += end - start;
	*\******END PLANNING GRAPH TIME MEASURE********/


	add_action_level(a_level_curr);
	set_length(get_length() + 1);

	//The no-op is done with the copy
	StateLevel s_level_next;
	s_level_next = s_level_curr;
	s_level_next.set_depth(get_length());
	bool new_state_insertion = false;

	/*\*****START PLANNING GRAPH TIME MEASURE*******
	start = std::chrono::system_clock::now();
	*\******END PLANNING GRAPH TIME MEASURE********/

	/*\*****START PLANNING GRAPH TIME MEASURE*******
	std::cerr << "\n\nPlaning Graph Length: " << get_length();
	if (get_length() > 0) {
		std::cerr << "\nAction Level creation: " << t1.count();
		std::cerr << "\nState Level Creation:  " << t2.count() << " of which:";
		std::cerr << "\nActions execution:     " << t4.count();
	}
	*\******END PLANNING GRAPH TIME MEASURE********/


	for (it_actset = a_level_curr.get_actions().begin(); it_actset != a_level_curr.get_actions().end(); it_actset++) {

		/*\*****START PLANNING GRAPH TIME MEASURE*******
		auto startN = std::chrono::system_clock::now();
		*\******END PLANNING GRAPH TIME MEASURE********/

		if (s_level_next.compute_successor(*it_actset, s_level_curr, m_belief_formula_false)) {
			new_state_insertion = true;
		}
		/*\*****START PLANNING GRAPH TIME MEASURE*******
		auto endN = std::chrono::system_clock::now();
		t4 += endN - startN;
		*\******END PLANNING GRAPH TIME MEASURE********/
	}

	/*\*****START PLANNING GRAPH TIME MEASURE*******
	end = std::chrono::system_clock::now();
	t2 += end - start;
	*\******END PLANNING GRAPH TIME MEASURE********/


	add_state_level(s_level_next);

	bool not_goal = false;

	/*\*****START PLANNING GRAPH TIME MEASURE*******
	start = std::chrono::system_clock::now();
	*\******END PLANNING GRAPH TIME MEASURE********/


	//Remove each sub goal already satisfied: it will always be and we just need to check it once
	for (auto it_fl = m_goal.begin(); it_fl != m_goal.end();) {

		if (s_level_next.pg_entailment(*it_fl)) {
			it_fl = m_goal.erase(it_fl);
			m_pg_sum += get_length();

		} else {
			it_fl++;
			not_goal = true;
		}
	}
	/*\*****START PLANNING GRAPH TIME MEASURE*******
	end = std::chrono::system_clock::now();
	t3 += end - start;
	*\******END PLANNING GRAPH TIME MEASURE********/
	if (!not_goal) {
		set_satisfiable(true);
		return;
	} else if (!new_state_insertion) {
		set_satisfiable(false);
		/*\*****START PLANNING GRAPH TIME MEASURE*******
		if (get_length() > 5) {
			print();
		}
		*\******END PLANNING GRAPH TIME MEASURE********/
		return;
	} else {
		pg_build();
	}
}

void PlanningGraph::add_state_level(const StateLevel & s_level)
{
	m_state_levels.push_back(s_level);
}

void PlanningGraph::add_action_level(const ActionLevel & a_level)
{

	m_action_levels.push_back(a_level);
}

void PlanningGraph::set_length(unsigned short length)
{

	m_pg_length = length;
}

void PlanningGraph::set_sum(unsigned short sum)
{

	m_pg_sum = sum;
}

unsigned short PlanningGraph::get_length() const
{
	//if (is_satisfiable()) {

	return m_pg_length;
	//}
	//std::cerr << "\nThe planning BisGraph could not find any solution. Check for the satisfiability before calling \"get_length\"\n";
	//exit(1);
}//construct planning BisGraph and return the level that satisfied the goal.

unsigned short PlanningGraph::get_sum() const
{
	//if (is_satisfiable()) {

	return m_pg_sum;
	//}
	//std::cerr << "\nThe planning BisGraph could not find any solution. Check for the satisfiability before calling \"get_sum\"\n";
	//exit(1);
}//

void PlanningGraph::set_goal(const FormulaeList & goal)
{
	m_goal = goal;
}

const std::vector< StateLevel > & PlanningGraph::get_state_levels() const
{
	return m_state_levels;
}

const std::vector< ActionLevel > & PlanningGraph::get_action_levels() const
{
	return m_action_levels;
}

const PG_FluentsScoreMap & PlanningGraph::get_f_scores() const
{
	return m_state_levels.back().get_f_map();
}

const PG_BeliefFormulaeMap & PlanningGraph::get_bf_scores() const
{
	return m_state_levels.back().get_bf_map();
}

const FormulaeList & PlanningGraph::get_goal() const
{
	return m_goal;
}

const ActionsSet & PlanningGraph::get_never_executed() const
{
	return m_never_executed;
}

const FormulaeSet & PlanningGraph::get_belief_formula_false() const
{
	return m_belief_formula_false;
}

PlanningGraph& PlanningGraph::operator=(const PlanningGraph & to_assign)
{
	set_pg(to_assign);
	return (*this);
}

void PlanningGraph::set_pg(const PlanningGraph & to_assign)
{
	m_state_levels = to_assign.get_state_levels();
	m_action_levels = to_assign.get_action_levels();
	m_pg_length = to_assign.get_length();
	m_pg_sum = to_assign.get_sum();
	m_satisfiable = to_assign.is_satisfiable();
	m_goal = to_assign.get_goal();
	//	m_bfs_score = to_assign.get_bfs_score();
	m_never_executed = to_assign.get_never_executed();
	m_belief_formula_false = to_assign.get_belief_formula_false();
}

void PlanningGraph::print(std::ostream& os) const
{

	os << "\n\n**********ePLANNING-GRAPH PRINT**********\n";
	unsigned short count = 0;
	for (auto it_stlv = m_state_levels.begin(); it_stlv != m_state_levels.end(); ++it_stlv) {
		auto fluents_score = it_stlv->get_f_map();
		auto bf_score = it_stlv->get_bf_map();

		os << "\n\t*******State Level " << count << "*******\n";
		os << "\n\t\t****Fluents****\n\n";
		for (auto it_pgf = fluents_score.begin(); it_pgf != fluents_score.end(); ++it_pgf) {
			os << "\t\t\t" << Domain::get_instance().get_grounder().deground_fluent(it_pgf->first) << " -> " << it_pgf->second << std::endl;
		}
		os << "\n\t\t****Belief Formulae****\n\n";
		for (auto it_pgbf = bf_score.begin(); it_pgbf != bf_score.end(); ++it_pgbf) {
			os << "\t\t\t";
			it_pgbf->first.print(os);
			os << " -> " << it_pgbf->second << std::endl;
		}
		os << "\n\t*******End State Level " << count << "*******\n";


		if (count < m_action_levels.size()) {
			os << "\n\t*******Action Level " << count << "*******\n";
			auto act_set = m_action_levels.at(count).get_actions();
			for (auto it_acts = act_set.begin(); it_acts != act_set.end(); it_acts++) {

				os << "\n\t\t" << it_acts->get_name() << std::endl;
			}
			os << "\n\t*******End Action Level " << count++ << "*******\n";
		}

	}
	os << "\n*********END ePLANNING-GRAPH PRINT**********\n";
}
