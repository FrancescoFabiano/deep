template <class T>
planning_graph::planning_graph(T & eState)


template <class T>
planning_graph::planning_graph(const formula_list & goal, T & eState)
{
	pg_state_level pg_init;
	pg_init.initialize(goal, eState);
	init(goal, pg_init);
}