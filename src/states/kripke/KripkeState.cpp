/*
 * \brief Implementation of \ref KripkeState.h
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date September 16, 2019
 */

#include <iostream>
#include <tuple>
#include <boost/dynamic_bitset.hpp>
#include <boost/functional/hash.hpp>
#include <unordered_map>
#include <set>

#include "KripkeState.h"

void KripkeState::set_worlds(const KripkeWorldPointersSet & to_set)
{
	m_worlds = to_set;
}

void KripkeState::set_pointed(const KripkeWorldPointer & to_set)
{
	m_pointed = to_set;
}

void KripkeState::set_beliefs(const KripkeWorldPointersTransitiveMap & to_set)
{
	m_beliefs = to_set;
}

void KripkeState::set_max_depth(unsigned int to_set)
{
	if (m_max_depth < to_set) m_max_depth = to_set;
}

const KripkeWorldPointersSet & KripkeState::get_worlds() const
{
	return m_worlds;
}

const KripkeWorldPointer & KripkeState::get_pointed() const
{
	return m_pointed;
}

const KripkeWorldPointersTransitiveMap & KripkeState::get_beliefs() const
{
	return m_beliefs;
}

unsigned int KripkeState::get_max_depth() const
{
	return m_max_depth;
}

bool KripkeState::operator=(const KripkeState & to_copy)
{
	set_worlds(to_copy.get_worlds());
	set_beliefs(to_copy.get_beliefs());
	m_max_depth = to_copy.get_max_depth();
	set_pointed(to_copy.get_pointed());
	//std::cerr << "\nDEBUG: eq ";
	return true;
}

bool KripkeState::operator<(const KripkeState & to_compare) const
{

	/*if (m_max_depth < to_compare.get_max_depth()) {
		return true;
	} else if (m_max_depth > to_compare.get_max_depth()) {
		return false;
	}*/

	if (m_pointed < to_compare.get_pointed()) {
		return true;
	} else if (m_pointed > to_compare.get_pointed()) {
		return false;
	}

	if (m_worlds < to_compare.get_worlds()) {
		return true;
	} else if (m_worlds > to_compare.get_worlds()) {
		return false;
	}

	KripkeWorldPointersTransitiveMap::const_iterator it_tramap1;
	KripkeWorldPointersTransitiveMap::const_iterator it_tramap2 = to_compare.get_beliefs().begin();

	KripkeWorldPointersMap tmp_pwmap1, tmp_pwmap2;
	KripkeWorldPointersMap::const_iterator it_pwmap1, it_pwmap2;
	//The same size is assured by the same size of m_worlds
	for (it_tramap1 = m_beliefs.begin(); it_tramap1 != m_beliefs.end(); it_tramap1++) {
		if (it_tramap1->first < it_tramap2->first) {
			return true;
		} else if (it_tramap1->first > it_tramap2->first) {
			return false;
		}

		tmp_pwmap1 = it_tramap1->second;
		tmp_pwmap2 = it_tramap2->second;
		if (tmp_pwmap1.size() < tmp_pwmap2.size()) {
			return true;
		} else if (tmp_pwmap1.size() > tmp_pwmap2.size()) {
			return false;
		}
		it_pwmap2 = tmp_pwmap2.begin();
		for (it_pwmap1 = tmp_pwmap1.begin(); it_pwmap1 != tmp_pwmap1.end(); it_pwmap1++) {
			if (it_pwmap1->first < it_pwmap2->first) {
				return true;
			} else if (it_pwmap1->first > it_pwmap2->first) {
				return false;
			}

			if (it_pwmap1->second < it_pwmap2->second) {
				return true;
			} else if (it_pwmap1->second > it_pwmap2->second) {
				return false;
			}
			it_pwmap2++;
		}
		it_tramap2++;
	}
	return false;
}

bool KripkeState::entails(Fluent f) const
{
	return entails(f, m_pointed);
}

bool KripkeState::entails(Fluent f, const KripkeWorldPointer & world) const
{
	return world.get_ptr()->entails(f);
}

bool KripkeState::entails(const FluentsSet & fl) const
{
	return entails(fl, m_pointed);
}

bool KripkeState::entails(const FluentsSet & fl, const KripkeWorldPointer & world) const
{
	return world.get_ptr()->entails(fl);
}

bool KripkeState::entails(const FluentFormula & ff) const
{
	return entails(ff, m_pointed);
}

bool KripkeState::entails(const FluentFormula & ff, const KripkeWorldPointer & world) const
{
	return world.get_ptr()->entails(ff);
}

bool KripkeState::entails(const BeliefFormula & bf) const
{
	return entails(bf, m_pointed);
}

bool KripkeState::entails(const BeliefFormula & to_check, const KripkeWorldPointersSet & reachable) const
{
	KripkeWorldPointersSet::const_iterator it_pws;
	for (it_pws = reachable.begin(); it_pws != reachable.end(); it_pws++) {
		if (!entails(to_check, *it_pws))
			return false;
	}
	return true;
}

bool KripkeState::entails(const BeliefFormula & bf, const KripkeWorldPointer & world) const
{
	/*
	 The entailment of a \ref BeliefFormula just call recursively the entailment on all the reachable world with that formula.
	 */
	KripkeWorldPointersSet D_reachable;
	switch ( bf.get_formula_type() ) {

	case FLUENT_FORMULA:
	{
		/** \todo Make sure its grounded. Maybe add to \ref BeliefFormula a bool that store if grounded or not or maybe ground
		 * when \ref domain created.
		 * @see BeliefFormula::ground(const grounder &).*/
		return entails(bf.get_fluent_formula(), world);
		break;
	}

	case BELIEF_FORMULA:

		/** \todo what was the at_lest_one of the previous version?*/
		return entails(bf.get_bf1(), get_B_reachable_worlds(bf.get_agent(), world));
		break;

	case PROPOSITIONAL_FORMULA:
		switch ( bf.get_operator() ) {
		case BF_NOT:
			return !entails(bf.get_bf1(), world);
			break;
		case BF_OR:
			return entails(bf.get_bf1(), world) || entails(bf.get_bf2(), world);
			break;
		case BF_AND:
			return entails(bf.get_bf1(), world) && entails(bf.get_bf2(), world);
			break;
		case BF_FAIL:
		default:
			std::cerr << "Something went wrong in checking entailment for Propositional formula";
			exit(1);
		}
		break;

	case E_FORMULA:
		//Check the entails on the E-reachable worlds
		return entails(bf.get_bf1(), get_E_reachable_worlds(bf.get_group_agents(), world));
		break;
		//Check the entails on the D-reachable worlds
	case D_FORMULA:
		D_reachable = get_D_reachable_worlds(bf.get_group_agents(), world);
		if (D_reachable.size() > 0) {
			return entails(bf.get_bf1(), D_reachable);
		} else {
			return false;
		}
		break;
		//Check the entails on the C-reachable worlds
	case C_FORMULA:
		return entails(bf.get_bf1(), get_C_reachable_worlds(bf.get_group_agents(), world));
		break;
	case BF_EMPTY:
	{

		return true;
		break;
	}
	case BF_TYPE_FAIL:
	default:
		std::cerr << "Something went wrong in checking entailment for Belief formula";
		exit(1);
	}

	return false;
}

bool KripkeState::entails(const formula_list & to_check, const KripkeWorldPointer & world) const
{
	//formula_list expresses CNF formula
	formula_list::const_iterator it_fl;
	for (it_fl = to_check.begin(); it_fl != to_check.end(); it_fl++) {
		if (!entails(*it_fl, world)) {
			return false;
		}
	}
	return true;
}

const KripkeWorldPointersSet KripkeState::get_B_reachable_worlds(Agent ag, const KripkeWorldPointer & world) const
{
	KripkeWorldPointersSet ret;
	auto pw_map = m_beliefs.find(world);

	if (pw_map != m_beliefs.end()) {
		auto pw_set = pw_map->second.find(ag);
		if (pw_set != pw_map->second.end()) {
			FormulaHelper_t::sum_set<KripkeWorldPointer>(ret, pw_set->second);
		}
	}
	return ret;
}

bool KripkeState::get_B_reachable_worlds_recoursive(Agent ag, const KripkeWorldPointer & world, KripkeWorldPointersSet& ret) const
{
	/** \todo check: If a--i-->b, b--i-->c then a--i-->c must be there*/
	auto pw_map = m_beliefs.find(world);

	if (pw_map != m_beliefs.end()) {
		auto pw_set = pw_map->second.find(ag);
		if (pw_set != pw_map->second.end()) {
			unsigned long previous_size = ret.size();
			FormulaHelper_t::sum_set<KripkeWorldPointer>(ret, pw_set->second);
			unsigned long current_size = ret.size();

			return previous_size == current_size;
		}
		/**@bug: We don't know why sometimes is outside the two if cases.*/
	}
	return true;
}

const KripkeWorldPointersSet KripkeState::get_E_reachable_worlds(const AgentsSet & ags, const KripkeWorldPointer & world) const
{
	/*The K^0 call of this function*/
	KripkeWorldPointersSet ret;
	AgentsSet::const_iterator it_agset;
	for (it_agset = ags.begin(); it_agset != ags.end(); it_agset++) {
		FormulaHelper_t::sum_set<KripkeWorldPointer>(ret, get_B_reachable_worlds(*it_agset, world));
	}

	return ret;
}

bool KripkeState::get_E_reachable_worlds_recoursive(const AgentsSet & ags, const KripkeWorldPointersSet & worlds, KripkeWorldPointersSet & ret) const
{
	/*The K^i (recoursive) call of this function*/

	bool is_fixed_point = true;
	KripkeWorldPointersSet::const_iterator it_pwptr;
	AgentsSet::const_iterator it_agset;
	for (it_pwptr = worlds.begin(); it_pwptr != worlds.end(); it_pwptr++) {
		for (it_agset = ags.begin(); it_agset != ags.end(); it_agset++) {
			if (!get_B_reachable_worlds_recoursive(*it_agset, *it_pwptr, ret)) {
				is_fixed_point = false;
			}
		}
	}
	return is_fixed_point;

}

const KripkeWorldPointersSet KripkeState::get_C_reachable_worlds(const AgentsSet & ags, const KripkeWorldPointer & world) const
{
	//Use of fixed point to stop.
	bool is_fixed_point = false;
	KripkeWorldPointersSet newly_reached = get_E_reachable_worlds(ags, world);
	KripkeWorldPointersSet already_reached;
	KripkeWorldPointersSet ret;
	while (!is_fixed_point) {
		FormulaHelper_t::sum_set<KripkeWorldPointer>(newly_reached, ret);
		FormulaHelper_t::minus_set<KripkeWorldPointer>(newly_reached, already_reached);
		is_fixed_point = get_E_reachable_worlds_recoursive(ags, newly_reached, ret);
		already_reached = newly_reached;
	}
	return ret;
}

const KripkeWorldPointersSet KripkeState::get_D_reachable_worlds(const AgentsSet & ags, const KripkeWorldPointer & world) const
{
	/**@bug: Notion of D-Reachable is correct (page 24 of Reasoning about Knowledge)*/
	AgentsSet::const_iterator it_agset = ags.begin();
	KripkeWorldPointersSet ret = get_B_reachable_worlds((*it_agset), world);
	it_agset++;

	for (; it_agset != ags.end(); it_agset++) {

		KripkeWorldPointersSet::iterator it_pwset1 = ret.begin();
		KripkeWorldPointersSet to_intersect = get_B_reachable_worlds((*it_agset), world);
		KripkeWorldPointersSet::const_iterator it_pwset2 = to_intersect.begin();
		while ((it_pwset1 != ret.end()) && (it_pwset2 != to_intersect.end())) {

			if ((*it_pwset1 < *it_pwset2) && (((*it_pwset1).get_fluent_based_id()) == ((*it_pwset2).get_fluent_based_id()))) {
				ret.erase(it_pwset1++);
			} else if ((*it_pwset2 < *it_pwset1) && ((((*it_pwset1).get_fluent_based_id()) == ((*it_pwset2).get_fluent_based_id())))) {
				++it_pwset2;
			} else { // *it_pwset1 == *it_pwset2
				++it_pwset1;
				++it_pwset2;
			}
		}

		// Anything left in ret from here on did not appear in to_intersect,
		// so we remove it.
		ret.erase(it_pwset1, ret.end());

	}
	std::cerr << "\nERROR: D_REACHABLILITY not yet Implemented correctly\n";
	exit(1);
	return ret;
}

void KripkeState::add_world(const KripkeWorld & world)
{

	m_worlds.insert(pstore::get_instance().add_world(world));
}

KripkeWorldPointer KripkeState::add_rep_world(const KripkeWorld & world, unsigned short repetition, bool& is_new)
{
	KripkeWorldPointer tmp = pstore::get_instance().add_world(world);
	tmp.set_repetition(repetition);
	is_new = std::get<1>(m_worlds.insert(tmp));

	return tmp;
}

KripkeWorldPointer KripkeState::add_rep_world(const KripkeWorld & world, unsigned short old_repetition)
{
	bool tmp = false;

	return add_rep_world(world, get_max_depth() + old_repetition, tmp);
}

KripkeWorldPointer KripkeState::add_rep_world(const KripkeWorld & world)
{
	bool tmp = false;

	return add_rep_world(world, get_max_depth(), tmp);
}

void KripkeState::add_edge(const KripkeWorldPointer &from, const KripkeWorldPointer &to, Agent ag)
{
	KripkeWorldPointersTransitiveMap::iterator from_beliefs = m_beliefs.find(from);

	if (from_beliefs != m_beliefs.end()) {
		KripkeWorldPointersMap::iterator ag_beliefs = from_beliefs->second.find(ag);

		if (ag_beliefs != from_beliefs->second.end()) {
			ag_beliefs->second.insert(to);
		} else {
			from_beliefs->second.insert(KripkeWorldPointersMap::value_type(ag,{to}));
		}
	} else {

		KripkeWorldPointersMap pwm;
		pwm.insert(KripkeWorldPointersMap::value_type(ag,{to}));
		m_beliefs.insert(KripkeWorldPointersTransitiveMap::value_type(from, pwm));
	}
}

void KripkeState::add_pworld_beliefs(const KripkeWorldPointer & world, const KripkeWorldPointersMap & beliefs)
{

	m_beliefs[world] = beliefs;
}

void KripkeState::build_initial()
{

	/** \todo for now prune building.*/
	std::cout << "\nBuilding initial possibility...\n";
	build_initial_prune();
}

void KripkeState::build_initial_structural()
{

}

void KripkeState::build_initial_prune()
{

	/*Building of all the possible consistent \ref pworld and setting the pointed world.
	 * Creation of all the \ref fluent combinations. All the consistent ones are added to \ref pstore.*/
	FluentsSet permutation;
	InitialStateInformation ini_conditions = domain::get_instance().get_initial_description();
	//	std::cerr << "\nDEBUG: Initially known fluents: ";
	//	printer::get_instance().print_list(domain::get_instance().get_grounder().deground_fluent(ini_conditions.get_initially_known_fluents()));
	generate_initial_pworlds(permutation, 0, ini_conditions.get_initially_known_fluents());


	/*Building of all the consistent edges.*/
	generate_initial_pedges();
}

/*From https://www.geeksforgeeks.org/generate-all-the-binary-strings-of-n-bits/ since is like generating all the binary numbers.*/
void KripkeState::generate_initial_pworlds(FluentsSet& permutation, int index, const FluentsSet & initially_known)
{
	//todo non usare indici ma bitset prima positivo fluent poi negativo
	int fluent_number = domain::get_instance().get_fluent_number();

	int bit_size = (domain::get_instance().get_size_fluent());

	if (index == fluent_number) {
		KripkeWorld to_add(permutation);
		add_initial_pworld(to_add);

		return;
	}

	FluentsSet permutation_2 = permutation;
	//Add the \ref fluent in positive version
	boost::dynamic_bitset<> bitSetToFindPositve(bit_size, index);
	boost::dynamic_bitset<> bitSetToFindNegative(bit_size, index);
	bitSetToFindNegative.set(bitSetToFindPositve.size() - 1, 1);
	bitSetToFindPositve.set(bitSetToFindPositve.size() - 1, 0);


	if (initially_known.find(bitSetToFindNegative) == initially_known.end()) {
		permutation.insert(bitSetToFindPositve);
		generate_initial_pworlds(permutation, index + 1, initially_known);
	}
	if (initially_known.find(bitSetToFindPositve) == initially_known.end()) {
		permutation_2.insert(bitSetToFindNegative);
		generate_initial_pworlds(permutation_2, index + 1, initially_known);
	}

}

void KripkeState::add_initial_pworld(const KripkeWorld & possible_add)
{
	InitialStateInformation ini_conditions = domain::get_instance().get_initial_description();

	switch ( ini_conditions.get_ini_restriction() ) {
	case S5:
	{
		/* Since the common knowledge is on all the agent it means that every possible \ref pworld
		 * in the initial state must entail *phi* where C(*phi*) is an initial condition.*/

		//Already setted in \ref domain::build_initially(bool).
		if (possible_add.entails(ini_conditions.get_ff_forS5())) {
			add_world(possible_add);
			if (possible_add.entails(ini_conditions.get_pointed_world_conditions())) {
				m_pointed = KripkeWorldPointer(possible_add);
				/*std::cout << "pointed world: ";
				printer::get_instance().print_list(possible_add.get_fluent_set());
				std::cout << std::endl;*/
			}
		} else {
			//std::cout << "error non generata una permeutazione uguale ad un altra" << std::endl;
			//Already generated so we save it on pstore
			pstore::get_instance().add_world(possible_add);
		}
		break;
	}
	case K45:
	{
		std::cerr << "\nNot implemented yet\n";
		exit(1);
		break;
	}
	case NONE:
	{
		std::cerr << "\nNot implemented yet\n";
		exit(1);
		break;
	}
	default:
	{
		std::cerr << "\nNot implemented yet\n";
		exit(1);

		break;
	}
	}
}

void KripkeState::generate_initial_pedges()
{
	KripkeWorldPointersSet::const_iterator it_pwps_1, it_pwps_2;

	KripkeWorldPointer pwptr_tmp1, pwptr_tmp2;

	/*This for add to *this* all the possible edges.*/
	for (it_pwps_1 = m_worlds.begin(); it_pwps_1 != m_worlds.end(); it_pwps_1++) {
		for (it_pwps_2 = it_pwps_1; it_pwps_2 != m_worlds.end(); it_pwps_2++) {
			for (auto agent : domain::get_instance().get_agents()) {
				pwptr_tmp1 = *it_pwps_1;
				pwptr_tmp2 = *it_pwps_2;
				add_edge(pwptr_tmp1, pwptr_tmp2, agent);
				add_edge(pwptr_tmp2, pwptr_tmp1, agent);

			}
		}
	}

	//std::cout << "Tot edges: " << m_edges.size() << std::endl;

	InitialStateInformation ini_conditions = domain::get_instance().get_initial_description();

	formula_list::const_iterator it_fl;
	for (it_fl = ini_conditions.get_initial_conditions().begin(); it_fl != ini_conditions.get_initial_conditions().end(); it_fl++) {

		remove_initial_pedge_bf(*it_fl);
	}
	//std::cout << "Removed edges: " << count << std::endl;

	//std::cout << "Final edges: " << m_edges.size() << std::endl;

}

void KripkeState::remove_edge(KripkeWorldPointer &from, const KripkeWorld &to, const Agent ag)
{
	auto from_beliefs = m_beliefs.find(from);

	if (from_beliefs != m_beliefs.end()) {
		auto ag_beliefs = from_beliefs->second.find(ag);

		if (ag_beliefs != from_beliefs->second.end()) {

			ag_beliefs->second.erase(to);
		}
	}
}

void KripkeState::remove_initial_pedge(const FluentFormula &known_ff, Agent ag)
{
	KripkeWorldPointersSet::const_iterator it_pwps_1, it_pwps_2;

	KripkeWorldPointer pwptr_tmp1, pwptr_tmp2;

	/** \todo maybe don't loop twice on the world but exploit using it_pwps_2 = it_pwps_1:
	 * - remove (_1, _2).
	 * - remove (_2, _1).*/
	for (it_pwps_1 = m_worlds.begin(); it_pwps_1 != m_worlds.end(); it_pwps_1++) {
		for (it_pwps_2 = it_pwps_1; it_pwps_2 != m_worlds.end(); it_pwps_2++) {
			/** \todo or entails(-known_ff)?*/
			pwptr_tmp1 = *it_pwps_1;
			pwptr_tmp2 = *it_pwps_2;
			if (pwptr_tmp1.get_ptr()->entails(known_ff) && !pwptr_tmp2.get_ptr()->entails(known_ff)) {
				remove_edge(pwptr_tmp1, *pwptr_tmp2.get_ptr(), ag);
				remove_edge(pwptr_tmp2, *pwptr_tmp1.get_ptr(), ag);
			} else if (pwptr_tmp2.get_ptr()->entails(known_ff) && !pwptr_tmp1.get_ptr()->entails(known_ff)) {

				remove_edge(pwptr_tmp2, *pwptr_tmp1.get_ptr(), ag);
				remove_edge(pwptr_tmp1, *pwptr_tmp2.get_ptr(), ag);
			}
		}
	}
}

void KripkeState::remove_initial_pedge_bf(const BeliefFormula & to_check)
{
	switch ( domain::get_instance().get_initial_description().get_ini_restriction() ) {
	case S5:
	{
		/* Just check whenever is B(--) \/ B(--) and remove that edge*/
		if (to_check.get_formula_type() == C_FORMULA) {

			BeliefFormula tmp = to_check.get_bf1();

			switch ( tmp.get_formula_type() ) {

				//Only one for edges -- expresses that someone is ignorant.
			case PROPOSITIONAL_FORMULA:
			{
				//We remove all the check on the formula since they have already been controlled when ini_conditions has been created
				if (tmp.get_operator() == BF_OR) {

					//fluent_formula known_ff;
					auto known_ff_ptr = std::make_shared<FluentFormula>();
					helper::check_Bff_notBff(tmp.get_bf1(), tmp.get_bf2(), known_ff_ptr);
					if (known_ff_ptr != nullptr) {
						//printer::get_instance().print_list(*known_ff_ptr);
						remove_initial_pedge(*known_ff_ptr, tmp.get_bf2().get_agent());
					}
					return;

				} else if (tmp.get_operator() == BF_AND) {
					//This case doesn't add knowledge.
					return;

				} else {
					std::cerr << "\nError in the type of initial formulae (FIFTH).\n";
					exit(1);
				}
				return;
				break;
			}
			case FLUENT_FORMULA:
			case BELIEF_FORMULA:
			{
				return;
				break;
			}
			case BF_EMPTY:
			{
				return;
				break;
			}

			default:
			{
				std::cerr << "\nError in the type of initial formulae (SIXTH).\n";
				exit(1);
				break;
			}
			}
		} else {
			std::cerr << "\nError in the type of initial formulae (SEVENTH).\n";
			exit(1);
		}

		return;
		break;
	}

	case K45:
	{
		std::cerr << "\nNot implemented yet\n";
		exit(1);
		break;
	}
	case NONE:
	{
		std::cerr << "\nNot implemented yet\n";
		exit(1);
		break;
	}
	default:
	{
		std::cerr << "\nNot implemented yet\n";
		exit(1);

		break;
	}
	}
}

/** \warning executability should be check in \ref state (or \ref planner).*/
KripkeState KripkeState::compute_succ(const Action & act) const
{
	//std::cerr << "\nDEBUG: Executing " << act.get_name();
	switch ( act.get_type() ) {
	case ONTIC:
	{
		return execute_ontic(act);
		break;
	}
	case SENSING:
	{
		return execute_sensing(act);

		break;
	}
	case ANNOUNCEMENT:
	{
		return execute_announcement(act);

		break;
	}
	default:
	{
		std::cerr << "Error in executing an action: " << act.get_name();
		std::cerr << " the type of the action is not defined correctly1";
		std::cerr << std::endl;
		exit(1);

		break;
	}
	}
}

void KripkeState::maintain_oblivious_believed_pworlds(KripkeState &ret, const AgentsSet & oblivious_obs_agents) const
{
	AgentsSet::const_iterator it_agset;
	KripkeWorldPointersSet world_oblivious;
	KripkeWorldPointersSet tmp_world_set;

	KripkeWorldPointersSet::const_iterator it_wo_ob;

	if (!oblivious_obs_agents.empty()) {
		tmp_world_set = get_E_reachable_worlds(oblivious_obs_agents, get_pointed());
		for (it_agset = domain::get_instance().get_agents().begin(); it_agset != domain::get_instance().get_agents().end(); it_agset++) {
			for (it_wo_ob = tmp_world_set.begin(); it_wo_ob != tmp_world_set.end(); it_wo_ob++) {
				FormulaHelper_t::sum_set<KripkeWorldPointer>(world_oblivious, get_B_reachable_worlds(*it_agset, *it_wo_ob));
			}
		}
		FormulaHelper_t::sum_set<KripkeWorldPointer>(world_oblivious, tmp_world_set);
		ret.set_max_depth(get_max_depth() + 1);
		ret.set_worlds(world_oblivious);

		for (it_wo_ob = world_oblivious.begin(); it_wo_ob != world_oblivious.end(); it_wo_ob++) {
			auto it_pwmap = m_beliefs.find(*it_wo_ob);

			if (it_pwmap != m_beliefs.end()) {

				ret.add_pworld_beliefs(*it_wo_ob, it_pwmap->second);
			}
		}
	}
}

KripkeWorldPointer KripkeState::execute_ontic_helper(const Action &act, KripkeState &ret, const KripkeWorldPointer &current_pw, TransitionMap &calculated, AgentsSet & oblivious_obs_agents) const
{
	// Execute the all the effects
	FluentFormula current_pw_effects = helper::get_effects_if_entailed(act.get_effects(), *this);
	FluentsSet world_description = current_pw.get_fluent_set();
	FluentFormula::const_iterator it_eff;

	for (it_eff = current_pw_effects.begin(); it_eff != current_pw_effects.end(); it_eff++) {
		helper::apply_effect(*it_eff, world_description);
		//		if (act.get_name().compare("distract_c_a") == 0) {
		//			std::cerr << "\nDEBUG: Inside the first ONTIC loop " << act.get_name();
		//		}
	}

	//	if (act.get_name().compare("distract_c_a") == 0) {
	//		std::cerr << "\nDEBUG: Out the first ONTIC loop " << act.get_name();
	//	}

	KripkeWorldPointer new_pw = ret.add_rep_world(KripkeWorld(world_description), current_pw.get_repetition()); // We add the corresponding pworld in ret
	calculated.insert(TransitionMap::value_type(current_pw, new_pw)); // And we update the calculated map

	auto it_pwtm = get_beliefs().find(current_pw);

	if (it_pwtm != get_beliefs().end()) {
		KripkeWorldPointersMap::const_iterator it_pwm;
		KripkeWorldPointersSet::const_iterator it_pws;

		for (it_pwm = it_pwtm->second.begin(); it_pwm != it_pwtm->second.end(); it_pwm++) {
			Agent ag = it_pwm->first;

			//			if (act.get_name().compare("distract_c_a") == 0) {
			//				std::cerr << "\nDEBUG: Inside the SECOND ONTIC loop " << act.get_name();
			//			}

			bool is_oblivious_obs = oblivious_obs_agents.find(ag) != oblivious_obs_agents.end();

			for (it_pws = it_pwm->second.begin(); it_pws != it_pwm->second.end(); it_pws++) {

				//				if (act.get_name().compare("distract_c_a") == 0) {
				//					std::cerr << "\nDEBUG: Inside the Third ONTIC loop " << act.get_name();
				//				}

				if (is_oblivious_obs) { // If we are dealing with an OBLIVIOUS agent we maintain its beliefs as they were
					auto maintained_pworld = ret.get_worlds().find(*it_pws);

					if (maintained_pworld != ret.get_worlds().end()) {
						ret.add_edge(new_pw, *it_pws, ag);
					}
				} else { // Otherwise, if we have a FULLY OBS agent
					auto calculated_pworld = calculated.find(*it_pws);

					if (calculated_pworld != calculated.end()) { // If we already calculated the transition function for this pworld
						ret.add_edge(new_pw, calculated_pworld->second, ag); // Then we update agents' beliefs
					} else {

						KripkeWorldPointer believed_pw = execute_ontic_helper(act, ret, *it_pws, calculated, oblivious_obs_agents);
						ret.add_edge(new_pw, believed_pw, ag);

						ret.set_max_depth(ret.get_max_depth() + 1 + current_pw.get_repetition());
					}
				}
			}
			//			if (act.get_name().compare("distract_c_a") == 0) {
			//				std::cerr << "\nDEBUG: Out the Second ONTIC loop " << act.get_name();
			//			}
		}
		//std::cerr << "\nDEBUG: Out the THIRD ONTIC loop " << act.get_name();

	}
	//std::cerr << "\nDEBUG: RETURN TO " << act.get_name();

	return new_pw;
}

KripkeState KripkeState::execute_ontic(const Action & act) const
{
	KripkeState ret;

	//This finds all the worlds that are reachable from the initial state following
	//the edges labeled with fully observant agents.
	AgentsSet agents = domain::get_instance().get_agents();
	AgentsSet fully_obs_agents = helper::get_agents_if_entailed(act.get_fully_observants(), *this);

	AgentsSet oblivious_obs_agents = agents;
    FormulaHelper_t::minus_set<Agent>(oblivious_obs_agents, fully_obs_agents);

	TransitionMap calculated; // A map that links the pworlds of *this* to the corresponding ones of ret
	maintain_oblivious_believed_pworlds(ret, oblivious_obs_agents);

	KripkeWorldPointer new_pointed = execute_ontic_helper(act, ret, get_pointed(), calculated, oblivious_obs_agents);
	ret.set_pointed(new_pointed); // Updating the pointed world
	//std::cerr << "\nDEBUG: Out ONTIC " << act.get_name();

	return ret;
}

KripkeWorldPointer KripkeState::execute_sensing_announcement_helper(const FluentFormula &effects, KripkeState &ret, const KripkeWorldPointer &current_pw, TransitionMap &calculated, AgentsSet &partially_obs_agents, AgentsSet &oblivious_obs_agents, bool previous_entailment) const
{
	KripkeWorldPointer new_pw = ret.add_rep_world(KripkeWorld(current_pw.get_fluent_set()), current_pw.get_repetition()); // We add the corresponding pworld in ret
	calculated.insert(TransitionMap::value_type(current_pw, new_pw)); // And we update the calculated map

	auto it_pwtm = get_beliefs().find(current_pw);

	if (it_pwtm != get_beliefs().end()) {
		KripkeWorldPointersMap::const_iterator it_pwm;
		KripkeWorldPointersSet::const_iterator it_pws;

		for (it_pwm = it_pwtm->second.begin(); it_pwm != it_pwtm->second.end(); it_pwm++) {
			Agent ag = it_pwm->first;

			bool is_oblivious_obs = oblivious_obs_agents.find(ag) != oblivious_obs_agents.end();
			bool is_partially_obs = partially_obs_agents.find(ag) != partially_obs_agents.end();
			bool is_fully_obs = !is_oblivious_obs && !is_partially_obs;

			for (it_pws = it_pwm->second.begin(); it_pws != it_pwm->second.end(); it_pws++) {
				if (is_oblivious_obs) { // If we are dealing with an OBLIVIOUS agent we maintain its beliefs as they were
					auto maintained_pworld = ret.get_worlds().find(*it_pws);

					if (maintained_pworld != ret.get_worlds().end()) {
						ret.add_edge(new_pw, *it_pws, ag);
					}
				} else { // Otherwise, if we have a FULLY/PARTIALLY observant agent
					auto calculated_pworld = calculated.find(*it_pws);
					//fluent_formula effects = get_effects_if_entailed(act.get_effects(), get_pointed());
					//bool ent = act.get_type() == SENSING ? entails(effects, *it_pws) == entails(effects, get_pointed()) : entails(effects, *it_pws);
					bool ent = entails(effects, *it_pws); // == entails(effects, get_pointed());


					bool is_consistent_belief = is_partially_obs || // If "ag" is PARTIALLY OBS, we always add an edge; If "ag" is FULLY OBS, we add an edge if he believes that "calculated" may be true (i.e., when "ent" holds) XOR
						(is_fully_obs && (ent == previous_entailment)); // if a PARTIALLY OBS agent believes that "ag" thinks that "calculated" may be true (i.e., when "previous_entailment" holds)

					if (calculated_pworld != calculated.end()) { // If we already calculated the transition function for this pworld
						if (is_consistent_belief) {
							ret.add_edge(new_pw, calculated_pworld->second, ag);
						}
					} else { // If we did not already calculate the transition function
						if (is_consistent_belief) { // We calculate it if it would result in a consistent belief...

							KripkeWorldPointer believed_pw = execute_sensing_announcement_helper(effects, ret, *it_pws, calculated, partially_obs_agents, oblivious_obs_agents, ent);
							ret.add_edge(new_pw, believed_pw, ag);
						}
						//						else if (is_partially_obs && (ent != previous_entailment)) { // ...and when it does not entail the action effects, but a PARTIALLY OBS agent believes that it may be true
						//							KripkeWorldPointer believed_pw_neg = execute_sensing_announcement_helper(act, ret, *it_pws, calculated, partially_obs_agents, oblivious_obs_agents, ent);
						//							ret.add_edge(new_pw, believed_pw_neg, ag);
						//						}
					}
				}
			}
		}
	}
	return new_pw;
}

KripkeState KripkeState::execute_sensing(const Action & act) const
{
	KripkeState ret;

	//This finds all the worlds that are reachable from the initial state following
	//the edges labeled with fully observant agents.
	AgentsSet agents = domain::get_instance().get_agents();
	AgentsSet fully_obs_agents = helper::get_agents_if_entailed(act.get_fully_observants(), *this);
	AgentsSet partially_obs_agents = helper::get_agents_if_entailed(act.get_partially_observants(), *this);

	AgentsSet oblivious_obs_agents = agents;
	FormulaHelper_t::minus_set<Agent>(oblivious_obs_agents, fully_obs_agents);
	FormulaHelper_t::minus_set<Agent>(oblivious_obs_agents, partially_obs_agents);

	if (!oblivious_obs_agents.empty()) {
		ret.set_max_depth(get_max_depth() + 1);
	}

	TransitionMap calculated; // A map that links the pworlds of *this* to the corresponding ones of ret
	maintain_oblivious_believed_pworlds(ret, oblivious_obs_agents);

	FluentFormula effects = helper::get_effects_if_entailed(act.get_effects(), *this);

	KripkeWorldPointer new_pointed = execute_sensing_announcement_helper(effects, ret, get_pointed(), calculated, partially_obs_agents, oblivious_obs_agents, entails(effects));
	ret.set_pointed(new_pointed); // Updating the pointed world

	/*if (!check_properties(fully_obs_agents, partially_obs_agents, effects, ret)) {
		std::cerr << "\nDEBUG: Some properties are not respected\n\n";
		exit(1);
	}*/

	return ret;
}

KripkeState KripkeState::execute_announcement(const Action & act) const
{
	//	KripkeState ret;
	//
	//	//This finds all the worlds that are reachable from the initial state following
	//	//the edges labeled with fully observant agents.
	//	agent_set agents = domain::get_instance().get_agents();
	//	agent_set fully_obs_agents = get_agents_if_entailed(act.get_fully_observants(), get_pointed());
	//	agent_set partially_obs_agents = get_agents_if_entailed(act.get_partially_observants(), get_pointed());
	//
	//	agent_set oblivious_obs_agents = agents;
	//	helper_t::minus_set(oblivious_obs_agents, fully_obs_agents);
	//	helper_t::minus_set(oblivious_obs_agents, partially_obs_agents);
	//
	//	if (!oblivious_obs_agents.empty()) {
	//		ret.set_max_depth(get_max_depth() + 1);
	//	}
	//
	//	TransitionMap calculated; // A map that links the pworlds of *this* to the corresponding ones of ret
	//	maintain_oblivious_believed_pworlds(ret, oblivious_obs_agents);
	//
	//	fluent_formula effects = get_effects_if_entailed(act.get_effects(), get_pointed());
	//
	//	KripkeWorldPointer new_pointed = execute_sensing_announcement_helper(act, ret, get_pointed(), calculated, partially_obs_agents, oblivious_obs_agents, entails(effects));
	//	ret.set_pointed(new_pointed); // Updating the pointed world
	//
	//	return ret;

	return execute_sensing(act);
}

bool KripkeState::check_properties(const AgentsSet & fully, const AgentsSet & partially, const FluentFormula & effects, const KripkeState & updated) const
{

	if (fully.size() > 0) {
		/************Formulae Building************/
		/******Formulae containing the effects of the sensing******/
		BeliefFormula effects_formula;
		effects_formula.set_formula_type(FLUENT_FORMULA);
		effects_formula.set_fluent_formula(effects);
		effects_formula.set_is_grounded(true);
		effects_formula.deground();


		/******First Property C(Fully, eff)******/
		BeliefFormula property1;
		property1.set_group_agents(fully);
		property1.set_formula_type(C_FORMULA);
		property1.set_bf1(effects_formula);
		property1.set_is_grounded(true);
		property1.deground();



		if (!updated.entails(property1)) {
			std::cerr << "\nDEBUG: First property not respected";
			return false;
		}



		if (partially.size() > 0) {
			/******Second Property C(P, (C(Fully,eff) | C(Fully, -eff)))******/
			BeliefFormula inner_nested2, nested2, disjunction, property2;
			//First nested formula is equal to First Property C(Fully, eff)
			//Second nested formula C(Fully, -f) is comprised of two layer (the NOT and C)
			inner_nested2.set_group_agents(fully);
			inner_nested2.set_formula_type(C_FORMULA);
			inner_nested2.set_bf1(effects_formula);
			inner_nested2.set_is_grounded(true);
			inner_nested2.deground();


			nested2.set_formula_type(PROPOSITIONAL_FORMULA);
			nested2.set_operator(BF_NOT);
			nested2.set_bf1(inner_nested2);
			nested2.set_is_grounded(true);
			nested2.deground();


			//The disjunction (C(Fully,eff) | C(Fully, -eff)) is made with property1 and nested2
			disjunction.set_formula_type(PROPOSITIONAL_FORMULA);
			disjunction.set_operator(BF_OR);
			disjunction.set_bf1(property1);
			disjunction.set_bf2(nested2);
			disjunction.set_is_grounded(true);
			disjunction.deground();


			//Finally we can construct the second property
			property2.set_group_agents(partially);
			property2.set_formula_type(C_FORMULA);
			property2.set_bf1(disjunction);
			property2.set_is_grounded(true);
			property2.deground();




			/******Third Property C(F, C(P, (C(Fully,eff) | C(Fully, -eff))))******/
			//This formula is just C(Fully, property2)
			BeliefFormula property3;
			property3.set_group_agents(fully);
			property3.set_formula_type(C_FORMULA);
			property3.set_bf1(property2);

			property3.set_is_grounded(true);

			property3.deground();


			if (!updated.entails(property2)) {
				std::cerr << "\nDEBUG: Second property not respected in the formula: ";
				property2.print_deground();
				return false;
			}

			if (!updated.entails(property3)) {
				std::cerr << "\nDEBUG: Third property not respected in the formula: ";
				property3.print_deground();
				return false;
			}
		}
	}

	return true;
}

void KripkeState::print() const
{
	int counter = 1;
	std::cout << std::endl;
	std::cout << "The Pointed World has id ";
	printer::get_instance().print_list(get_pointed().get_fluent_set());
	std::cout << "-" << get_pointed().get_repetition();
	std::cout << std::endl;
	std::cout << "*******************************************************************" << std::endl;

	KripkeWorldPointersSet::const_iterator it_pwset;
	std::cout << "World List:" << std::endl;

	for (it_pwset = get_worlds().begin(); it_pwset != get_worlds().end(); it_pwset++) {
		std::cout << "W-" << counter << ": ";
		printer::get_instance().print_list(it_pwset->get_fluent_set());
		std::cout << " rep:" << it_pwset->get_repetition();
		std::cout << std::endl;
		counter++;
	}
	counter = 1;
	std::cout << std::endl;
	std::cout << "*******************************************************************" << std::endl;
	KripkeWorldPointersTransitiveMap::const_iterator it_pwtm;
	KripkeWorldPointersMap::const_iterator it_pwm;
	std::cout << "Edge List:" << std::endl;
	for (it_pwtm = get_beliefs().begin(); it_pwtm != get_beliefs().end(); it_pwtm++) {
		KripkeWorldPointer from = it_pwtm->first;
		KripkeWorldPointersMap from_map = it_pwtm->second;

		for (it_pwm = from_map.begin(); it_pwm != from_map.end(); it_pwm++) {
			Agent ag = it_pwm->first;
			KripkeWorldPointersSet to_set = it_pwm->second;

			for (it_pwset = to_set.begin(); it_pwset != to_set.end(); it_pwset++) {

				KripkeWorldPointer to = *it_pwset;

				std::cout << "E-" << counter << ": (";
				printer::get_instance().print_list(from.get_fluent_set());
				std::cout << "," << from.get_repetition();
				std::cout << ") - (";
				printer::get_instance().print_list(to.get_fluent_set());
				std::cout << "," << to.get_repetition();
				std::cout << ") ag:" << domain::get_instance().get_grounder().deground_agent(ag);
				std::cout << std::endl;
				counter++;
			}
		}
	}
	std::cout << "*******************************************************************" << std::endl;
}

/* void KripkeState::print_ML_dataset_unoptimized() const
{
	//World list; Pointed world; Edges
	// Mapping worlds using fluent set and repetition as unique key
	std::map<std::pair<std::set<fluent>, int>, int> world_map;
	int counter = 1;

	KripkeWorldPointersSet::const_iterator it_pwset;
	for (it_pwset = get_worlds().begin(); it_pwset != get_worlds().end(); it_pwset++) {
		std::pair<std::set<fluent>, int> key = {it_pwset->get_fluent_set(), it_pwset->get_repetition()};
		world_map[key] = counter; // Store W-number
		if (counter > 1){
			std::cout << std::endl;
		}
		std::cout << "W" << counter << ":";
		printer::get_instance().print_list(it_pwset->get_fluent_set());
		std::cout << " rep:" << it_pwset->get_repetition();
		counter++;
	}
	std::cout << ";" << std::endl;

	// Print pointed world using W-number
	KripkeWorldPointer pointed = get_pointed();
	std::pair<std::set<fluent>, int> pointed_key = {pointed.get_fluent_set(), pointed.get_repetition()};
	std::cout << "P:W" << world_map[pointed_key] << ";" << std::endl;

	// Unifying edges by grouping agents
	std::map<std::pair<int, int>, std::set<agent>> edge_map;
	KripkeWorldPointersTransitiveMap::const_iterator it_pwtm;
	KripkeWorldPointersMap::const_iterator it_pwm;
	for (it_pwtm = get_beliefs().begin(); it_pwtm != get_beliefs().end(); it_pwtm++) {
		KripkeWorldPointer from = it_pwtm->first;
		KripkeWorldPointersMap from_map = it_pwtm->second;

		for (it_pwm = from_map.begin(); it_pwm != from_map.end(); it_pwm++) {
			agent ag = it_pwm->first;
			KripkeWorldPointersSet to_set = it_pwm->second;

			for (it_pwset = to_set.begin(); it_pwset != to_set.end(); it_pwset++) {
				KripkeWorldPointer to = *it_pwset;
				std::pair<std::set<fluent>, int> from_key = {from.get_fluent_set(), from.get_repetition()};
				std::pair<std::set<fluent>, int> to_key = {to.get_fluent_set(), to.get_repetition()};

				std::pair<int, int> edge_key = {world_map[from_key], world_map[to_key]};
				edge_map[edge_key].insert(ag);
			}
		}
	}

	// Print unified edges
	counter = 1;
	for (const auto& entry : edge_map) {
		if (counter > 1){
			std::cout << std::endl;
		}
		std::cout << "E" << counter << ":W" << entry.first.first << "->W" << entry.first.second << " {";
		bool first = true;
		for (const auto& ag : entry.second) {
			if (!first) std::cout << ",";
			std::cout << domain::get_instance().get_grounder().deground_agent(ag);
			first = false;
		}
		std::cout << "}";
		counter++;
	}
}
 */

void KripkeState::print_graphviz(std::ostream & graphviz) const
{
	StringsSet::const_iterator it_st_set;
	FluentsSet::const_iterator it_fs;


	graphviz << "//WORLDS List:" << std::endl;
	std::map<FluentsSet, int> map_world_to_index;
	std::map<unsigned short, char> map_rep_to_name;
	char found_rep = (char) ((char) domain::get_instance().get_agents().size() + 'A');
	int found_fs = 0;
	FluentsSet tmp_fs;
	char tmp_unsh;
	StringsSet tmp_stset;
	bool print_first;
	KripkeWorldPointersSet::const_iterator it_pwset;
	for (it_pwset = get_worlds().begin(); it_pwset != get_worlds().end(); it_pwset++) {
		if (*it_pwset == get_pointed())
			graphviz << "	node [shape = doublecircle] ";
		else
			graphviz << "	node [shape = circle] ";

		print_first = false;
		tmp_fs = it_pwset->get_fluent_set();
		if (map_world_to_index.count(tmp_fs) == 0) {
			map_world_to_index[tmp_fs] = found_fs;
			found_fs++;
		}
		tmp_unsh = it_pwset->get_repetition();
		if (map_rep_to_name.count(tmp_unsh) == 0) {
			map_rep_to_name[tmp_unsh] = found_rep;
			found_rep++;
		}
		graphviz << "\"" << map_rep_to_name[tmp_unsh] << "_" << map_world_to_index[tmp_fs] << "\";";
		graphviz << "// (";
		tmp_stset = domain::get_instance().get_grounder().deground_fluent(tmp_fs);
		for (it_st_set = tmp_stset.begin(); it_st_set != tmp_stset.end(); it_st_set++) {
			if (print_first) {
				graphviz << ",";
			}
			print_first = true;
			graphviz << *it_st_set;
		}
		graphviz << ")\n";
	}

	graphviz << "\n\n";
	graphviz << "//RANKS List:" << std::endl;

	std::map<int, KripkeWorldPointersSet> for_rank_print;
	for (it_pwset = get_worlds().begin(); it_pwset != get_worlds().end(); it_pwset++) {
		for_rank_print[it_pwset->get_repetition()].insert(*it_pwset);
	}

	std::map<int, KripkeWorldPointersSet>::const_iterator it_map_rank;
	for (it_map_rank = for_rank_print.begin(); it_map_rank != for_rank_print.end(); it_map_rank++) {
		graphviz << "	{rank = same; ";
		for (it_pwset = it_map_rank->second.begin(); it_pwset != it_map_rank->second.end(); it_pwset++) {
			graphviz << "\"" << map_rep_to_name[it_pwset->get_repetition()] << "_" << map_world_to_index[it_pwset->get_fluent_set()] << "\"; ";
		}
		graphviz << "}\n";
	}


	graphviz << "\n\n";
	graphviz << "//EDGES List:" << std::endl;

	std::map < std::tuple<std::string, std::string>, std::set<std::string> > edges;

	KripkeWorldPointersTransitiveMap::const_iterator it_pwtm;
	KripkeWorldPointersMap::const_iterator it_pwm;
	std::tuple<std::string, std::string> tmp_tuple;
	std::string tmp_string = "";

	for (it_pwtm = get_beliefs().begin(); it_pwtm != get_beliefs().end(); it_pwtm++) {
		KripkeWorldPointer from = it_pwtm->first;
		KripkeWorldPointersMap from_map = it_pwtm->second;

		for (it_pwm = from_map.begin(); it_pwm != from_map.end(); it_pwm++) {
			Agent ag = it_pwm->first;
			KripkeWorldPointersSet to_set = it_pwm->second;

			for (it_pwset = to_set.begin(); it_pwset != to_set.end(); it_pwset++) {
				KripkeWorldPointer to = *it_pwset;

				tmp_string = "_" + std::to_string(map_world_to_index[from.get_fluent_set()]);
				tmp_string.insert(0, 1, map_rep_to_name[from.get_repetition()]);
				std::get<0>(tmp_tuple) = tmp_string;

				tmp_string = "_" + std::to_string(map_world_to_index[to.get_fluent_set()]);
				tmp_string.insert(0, 1, map_rep_to_name[to.get_repetition()]);
				std::get<1>(tmp_tuple) = tmp_string;

				edges[tmp_tuple].insert(domain::get_instance().get_grounder().deground_agent(ag));

			}
		}
	}

	std::map < std::tuple<std::string, std::string>, std::set < std::string>>::iterator it_map;
	std::map < std::tuple<std::string, std::string>, std::set < std::string>>::const_iterator it_map_2;

	std::map < std::tuple<std::string, std::string>, std::set < std::string>> to_print_double;
	for (it_map = edges.begin(); it_map != edges.end(); it_map++) {
		for (it_map_2 = it_map; it_map_2 != edges.end(); it_map_2++) {
			if (std::get<0>(it_map->first).compare(std::get<1>(it_map_2->first)) == 0) {
				if (std::get<1>(it_map->first).compare(std::get<0>(it_map_2->first)) == 0) {
					if (it_map->second == it_map_2->second) {
						if (std::get<0>(it_map->first).compare(std::get<1>(it_map->first)) != 0) {
							to_print_double[it_map->first] = it_map->second;
							//std::cerr << std::get<0>(it_map->first) << " " << std::get<0>(it_map_2->first) << "\n";
							it_map_2 = edges.erase(it_map_2);
							it_map = edges.erase(it_map);
						}
					}
				}
			}
		}
	}

	std::set<std::string>::const_iterator it_stset;
	for (it_map = edges.begin(); it_map != edges.end(); it_map++) {
		graphviz << "	\"";
		graphviz << std::get<0>(it_map->first);
		graphviz << "\" -> \"";
		graphviz << std::get<1>(it_map->first);
		graphviz << "\" ";
		graphviz << "[ label = \"";
		tmp_string = "";
		for (it_stset = it_map->second.begin(); it_stset != it_map->second.end(); it_stset++) {
			tmp_string += *it_stset;
			tmp_string += ",";
		}
		tmp_string.pop_back();
		graphviz << tmp_string;
		graphviz << "\" ];\n";
	}

	for (it_map = to_print_double.begin(); it_map != to_print_double.end(); it_map++) {
		graphviz << "	\"";
		graphviz << std::get<0>(it_map->first);
		graphviz << "\" -> \"";
		graphviz << std::get<1>(it_map->first);
		graphviz << "\" ";
		graphviz << "[ dir=both label = \"";
		tmp_string = "";
		for (it_stset = it_map->second.begin(); it_stset != it_map->second.end(); it_stset++) {

			tmp_string += *it_stset;
			tmp_string += ",";
		}
		tmp_string.pop_back();
		graphviz << tmp_string;
		graphviz << "\" ];\n";
	}

	std::string color = "<font color=\"#ffffff\">";
	graphviz << "\n\n//WORLDS description Table:" << std::endl;
	graphviz << "	node [shape = plain]\n\n";
	graphviz << "	description[label=<\n";
	graphviz << "	<table border = \"0\" cellborder = \"1\" cellspacing = \"0\" >\n";
	for (it_pwset = get_worlds().begin(); it_pwset != get_worlds().end(); it_pwset++) {
		tmp_fs = it_pwset->get_fluent_set();
		print_first = false;
		graphviz << "		<tr><td>" << map_rep_to_name[it_pwset->get_repetition()] << "_" << map_world_to_index[tmp_fs] << "</td> <td>";
		for (it_fs = tmp_fs.begin(); it_fs != tmp_fs.end(); it_fs++) {
			if (print_first) {

				graphviz << ", ";
			}
			print_first = true;
			if (helper::is_negate(*it_fs)) color = "<font color=\"#0000ff\"> ";
			else color = "<font color=\"#ff1020\">";
			graphviz << color << domain::get_instance().get_grounder().deground_fluent(*it_fs) << "</font>";
		}
		graphviz << "</td></tr>\n";
	}
	graphviz << "	</table>>]\n";
	graphviz << "	{rank = max; description};\n";

}


struct WorldHash {
    std::size_t operator()(const std::pair<std::set<Fluent>, int>& p) const {
        std::size_t seed = 0;
        for (const auto& f : p.first) {
            boost::hash_combine(seed, boost::hash_value(f));
        }
        boost::hash_combine(seed, std::hash<int>{}(p.second));
        return seed;
    }
};


std::string to_base36(int num) {
    /*const char* digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    if (num == 0) return "0";
    std::string result;
    while (num > 0) {
        result = digits[num % 36] + result;
        num /= 36;
    } For now we keep the information in integer for better compatibility with GNNs*/
    return std::to_string(num);
}

void KripkeState::print_ML_dataset(std::ostream & graphviz) const
{
    graphviz << "digraph G {" << std::endl;

    std::unordered_map<std::size_t, int> world_map;
    int world_counter = 1;

    // Assign IDs
    for (const auto& pw : get_worlds()) {
        std::pair<std::set<Fluent>, int> world_key = {pw.get_fluent_set(), pw.get_repetition()};
        std::size_t hash_value = WorldHash{}(world_key);

        if (world_map.find(hash_value) == world_map.end()) {
            world_map[hash_value] = world_counter++;
        }
    }

    // Print nodes
    for (const auto& [hash, id] : world_map) {
        graphviz << to_base36(id) << ";" << std::endl;
    }

    // (Optional) Pointed world node decoration
    {
        std::pair<std::set<Fluent>, int> pointed_key = {get_pointed().get_fluent_set(), get_pointed().get_repetition()};
        std::size_t pointed_hash = WorldHash{}(pointed_key);
        graphviz <<  to_base36(world_map[pointed_hash]) << " [shape=doublecircle];" << std::endl;
    }

    // Belief edges
    std::map<std::pair<int, int>, std::set<Agent>> edge_map;
    for (const auto& [from_pw, from_map] : get_beliefs()) {
        for (const auto& [ag, to_set] : from_map) {
            for (const auto& to_pw : to_set) {
                std::pair<std::set<Fluent>, int> from_key = {from_pw.get_fluent_set(), from_pw.get_repetition()};
                std::pair<std::set<Fluent>, int> to_key = {to_pw.get_fluent_set(), to_pw.get_repetition()};

                std::size_t from_hash = WorldHash{}(from_key);
                std::size_t to_hash = WorldHash{}(to_key);

                int from_id = world_map[from_hash];
                int to_id = world_map[to_hash];
                edge_map[{from_id, to_id}].insert(ag);
            }
        }
    }

    // Print edges
    for (const auto& [edge, agents] : edge_map) {
        graphviz <<  to_base36(edge.first)
                  << " ->" << to_base36(edge.second)
                  << " [label=\"";

        bool first_ag = true;
        for (const auto& ag : agents) {
            if (!first_ag) graphviz << ",";
            graphviz << domain::get_instance().get_grounder().deground_agent(ag);
            first_ag = false;
        }

        graphviz << "\"];" << std::endl;
    }

    graphviz << "}" << std::endl;
}


void KripkeState::print_graphviz_explicit(std::ostream & graphviz) const
{
	StringsSet::const_iterator it_st_set;
	FluentsSet::const_iterator it_fs;


	graphviz << "//WORLDS List:" << std::endl;
	std::map<FluentsSet, std::string> map_world_to_index;
	std::map<unsigned short, int> map_rep_to_name;
	int found_rep = 0;
	int found_fs = 0;
	FluentsSet tmp_fs;
	char tmp_unsh;
	StringsSet tmp_stset;
	bool print_first;
	KripkeWorldPointersSet::const_iterator it_pwset;
	for (it_pwset = get_worlds().begin(); it_pwset != get_worlds().end(); it_pwset++) {
		if (*it_pwset == get_pointed())
			graphviz << "	node [shape = doublecircle] ";
		else
			graphviz << "	node [shape = circle] ";

		print_first = false;
		tmp_fs = it_pwset->get_fluent_set();
		std::string world_value = "";
		if (map_world_to_index.count(tmp_fs) == 0) {

			tmp_stset = domain::get_instance().get_grounder().deground_fluent(tmp_fs);
			for (it_st_set = tmp_stset.begin(); it_st_set != tmp_stset.end(); it_st_set++) {
				if (print_first) {
					world_value += ",";
				}
				print_first = true;
				world_value += *it_st_set;
			}

			map_world_to_index[tmp_fs] = world_value;
			found_fs++;
		}
		tmp_unsh = it_pwset->get_repetition();
		if (map_rep_to_name.count(tmp_unsh) == 0) {
			map_rep_to_name[tmp_unsh] = found_rep;
			found_rep++;
		}
		graphviz << "\"" << map_rep_to_name[tmp_unsh] << "_" << map_world_to_index[tmp_fs] << "\";";
		graphviz << "// (";
		tmp_stset = domain::get_instance().get_grounder().deground_fluent(tmp_fs);
		for (it_st_set = tmp_stset.begin(); it_st_set != tmp_stset.end(); it_st_set++) {
			if (print_first) {
				graphviz << ",";
			}
			print_first = true;
			graphviz << *it_st_set;
		}
		graphviz << ")\n";
	}

	graphviz << "\n\n";
	graphviz << "//RANKS List:" << std::endl;

	std::map<int, KripkeWorldPointersSet> for_rank_print;
	for (it_pwset = get_worlds().begin(); it_pwset != get_worlds().end(); it_pwset++) {
		for_rank_print[it_pwset->get_repetition()].insert(*it_pwset);
	}

	std::map<int, KripkeWorldPointersSet>::const_iterator it_map_rank;
	for (it_map_rank = for_rank_print.begin(); it_map_rank != for_rank_print.end(); it_map_rank++) {
		graphviz << "	{rank = same; ";
		for (it_pwset = it_map_rank->second.begin(); it_pwset != it_map_rank->second.end(); it_pwset++) {
			graphviz << "\"" << map_rep_to_name[it_pwset->get_repetition()] << "_" << map_world_to_index[it_pwset->get_fluent_set()] << "\"; ";
		}
		graphviz << "}\n";
	}


	graphviz << "\n\n";
	graphviz << "//EDGES List:" << std::endl;

	std::map < std::tuple<std::string, std::string>, std::set<std::string> > edges;

	KripkeWorldPointersTransitiveMap::const_iterator it_pwtm;
	KripkeWorldPointersMap::const_iterator it_pwm;
	std::tuple<std::string, std::string> tmp_tuple;
	std::string tmp_string = "";

	for (it_pwtm = get_beliefs().begin(); it_pwtm != get_beliefs().end(); it_pwtm++) {
		KripkeWorldPointer from = it_pwtm->first;
		KripkeWorldPointersMap from_map = it_pwtm->second;

		for (it_pwm = from_map.begin(); it_pwm != from_map.end(); it_pwm++) {
			Agent ag = it_pwm->first;
			KripkeWorldPointersSet to_set = it_pwm->second;

			for (it_pwset = to_set.begin(); it_pwset != to_set.end(); it_pwset++) {
				KripkeWorldPointer to = *it_pwset;

				tmp_string = std::to_string(map_rep_to_name[from.get_repetition()]);
				tmp_string += "_" + map_world_to_index[from.get_fluent_set()];
				//tmp_string.insert(0, 1, std::to_string(map_rep_to_name[from.get_repetition()]));
				std::get<0>(tmp_tuple) = tmp_string;

				tmp_string = std::to_string(map_rep_to_name[to.get_repetition()]);
				tmp_string += "_" + map_world_to_index[to.get_fluent_set()];
				//tmp_string.insert(0, 1, map_rep_to_name[to.get_repetition()]);
				std::get<1>(tmp_tuple) = tmp_string;

				edges[tmp_tuple].insert(domain::get_instance().get_grounder().deground_agent(ag));

			}
		}
	}

	std::map < std::tuple<std::string, std::string>, std::set < std::string>>::iterator it_map;
	std::map < std::tuple<std::string, std::string>, std::set < std::string>>::const_iterator it_map_2;

	std::map < std::tuple<std::string, std::string>, std::set < std::string>> to_print_double;
	for (it_map = edges.begin(); it_map != edges.end(); it_map++) {
		for (it_map_2 = it_map; it_map_2 != edges.end(); it_map_2++) {
			if (std::get<0>(it_map->first).compare(std::get<1>(it_map_2->first)) == 0) {
				if (std::get<1>(it_map->first).compare(std::get<0>(it_map_2->first)) == 0) {
					if (it_map->second == it_map_2->second) {
						if (std::get<0>(it_map->first).compare(std::get<1>(it_map->first)) != 0) {
							to_print_double[it_map->first] = it_map->second;
							//std::cerr << std::get<0>(it_map->first) << " " << std::get<0>(it_map_2->first) << "\n";
							it_map_2 = edges.erase(it_map_2);
							it_map = edges.erase(it_map);
						}
					}
				}
			}
		}
	}

	std::set<std::string>::const_iterator it_stset;
	for (it_map = edges.begin(); it_map != edges.end(); it_map++) {
		graphviz << "	\"";
		graphviz << std::get<0>(it_map->first);
		graphviz << "\" -> \"";
		graphviz << std::get<1>(it_map->first);
		graphviz << "\" ";
		graphviz << "[ label = \"";
		tmp_string = "";
		for (it_stset = it_map->second.begin(); it_stset != it_map->second.end(); it_stset++) {
			tmp_string += *it_stset;
			tmp_string += ",";
		}
		tmp_string.pop_back();
		graphviz << tmp_string;
		graphviz << "\" ];\n";
	}

	for (it_map = to_print_double.begin(); it_map != to_print_double.end(); it_map++) {
		graphviz << "	\"";
		graphviz << std::get<0>(it_map->first);
		graphviz << "\" -> \"";
		graphviz << std::get<1>(it_map->first);
		graphviz << "\" ";
		graphviz << "[ dir=both label = \"";
		tmp_string = "";
		for (it_stset = it_map->second.begin(); it_stset != it_map->second.end(); it_stset++) {

			tmp_string += *it_stset;
			tmp_string += ",";
		}
		tmp_string.pop_back();
		graphviz << tmp_string;
		graphviz << "\" ];\n";
	}

}

/****************BISIMULATION********************/
void KripkeState::get_all_reachable_worlds(const KripkeWorldPointer & pw, KripkeWorldPointersSet & reached_worlds, KripkeWorldPointersTransitiveMap & reached_edges) const
{
	KripkeWorldPointersSet::const_iterator it_pwps;
	KripkeWorldPointersSet pw_list;

	auto ag_set = domain::get_instance().get_agents();
	auto ag_it = ag_set.begin();
	for (; ag_it != ag_set.end(); ag_it++) {

		try {
			pw_list = m_beliefs.at(pw).at(*ag_it);
		} catch (const std::out_of_range& e) {
			//std::cerr << "[ERROR] Key not found in m_beliefs or inner map for agent " << *ag_it << ": " << e.what() << std::endl;
			pw_list.clear();
		}

		for (it_pwps = pw_list.begin(); it_pwps != pw_list.end(); it_pwps++) {
			if (reached_worlds.insert(*it_pwps).second) {
				get_all_reachable_worlds(*it_pwps, reached_worlds, reached_edges);
				try {
					reached_edges.insert(std::make_pair(*it_pwps, m_beliefs.at(*it_pwps)));
				} catch (const std::out_of_range& e) {
					//std::cerr << "[ERROR] Key not found in m_beliefs during reachable worlds exploration: " << e.what() << std::endl;
					// Nothing else needed here
				}
			}
		}
	}
}

void KripkeState::clean_unreachable_pworlds()
{
	//std::cerr << "\nDEBUG: INIZIO CLEAN EXTRA KripkeState\n" << std::flush;

	KripkeWorldPointersSet reached_worlds;
	KripkeWorldPointersTransitiveMap reached_edges;

	reached_worlds.insert(get_pointed());
	try {
		reached_edges.insert(std::make_pair(get_pointed(), m_beliefs.at(get_pointed())));
	} catch (const std::out_of_range& e) {
		//std::cerr << "[ERROR] Key not found in m_beliefs for pointed world: " << e.what() << std::endl;
		//exit(1);
	}	//std::cerr << "\nDEBUG: CLEAN 1 EXTRA KripkeState\n" << std::flush;

	get_all_reachable_worlds(get_pointed(), reached_worlds, reached_edges);

	//std::cerr << "\nDEBUG: CLEAN 2 EXTRA KripkeState\n" << std::flush;

	set_worlds(reached_worlds);
	set_beliefs(reached_edges);

	//std::cerr << "\nDEBUG: FINE CLEAN EXTRA KripkeState\n" << std::flush;
}

const automa KripkeState::pstate_to_automaton(std::vector<KripkeWorldPointer> & pworld_vec, const std::map<Agent, bis_label> & agent_to_label) const
{

	std::map<int, int> compact_indices;
	std::map<KripkeWorldPointer, int> index_map;
	pbislabel_map label_map; // Map: from -> (to -> ag_set)

	automa *a;
	int Nvertex = get_worlds().size();
	int ag_set_size = domain::get_instance().get_agents().size();
	//BIS_ADAPTATION For the loop that identifies the id (We add one edge for each node)
	v_elem *Vertex;

	Vertex = (v_elem *) malloc(sizeof(v_elem) * Nvertex);

	// Initializating vertices
	KripkeWorldPointersSet::const_iterator it_pwps;
	KripkeWorldPointersTransitiveMap::const_iterator it_peps;
	pbislabel_map::const_iterator it_plm;
	bis_label_set::const_iterator it_bislab;
	std::map<KripkeWorldPointer, bis_label_set>::const_iterator it_pw_bislab;

	//std::cerr << "\nDEBUG: Inizializzazione Edges\n";

	// The pointed world is set to the index 0. This ensures that, when deleting the bisimilar nodes, the pointed pworld
	// is always chosen as the first of its block. Therefore, we do not need to update it when converting back to a kstate
	index_map[get_pointed()] = 0;
	pworld_vec.push_back(get_pointed());
	compact_indices[get_pointed().get_internal_world_id()] = 0;

	//For the loop that identifies the id
	//BIS_ADAPTATION For the loop that identifies the id (+1)
	///@bug: If the pointed has no self-loop to add
	//KripkeWorldPointersSet pointed_adj = adj_list.at(get_pointed());

	Vertex[0].ne = 0; // pointed_adj.size(); // edge_counter[get_pointed()];
	//	if (pointed_adj.find(get_pointed()) == pointed_adj.end()) {
	//		Vertex[0].ne++;
	//	}
	//	Vertex[0].e = (e_elem *) malloc(sizeof(e_elem) * Vertex[0].ne);

	int i = 1, c = 1;

	//std::cerr << "\nDEBUG: Inizializzazione Vertex\n";

	for (it_pwps = m_worlds.begin(); it_pwps != m_worlds.end(); it_pwps++) {
		if (!(*it_pwps == get_pointed())) {
			index_map[*it_pwps] = i;
			pworld_vec.push_back(*it_pwps);

			// if (compact_indices.find(it_pwps->get_internal_world_id()) == compact_indices.end()) {
			if (compact_indices.insert({it_pwps->get_internal_world_id(), c}).second) {
				// compact_indices[it_pwps->get_internal_world_id()] = c;
				c++;
				//std::cerr << "\nDEBUG: Added:" << it_pwps->get_id() << "\n";
			}
			Vertex[i].ne = 0;
			i++;
		}
		//BIS_ADAPTATION (Added self-loop)
		label_map[*it_pwps][*it_pwps].insert(compact_indices[it_pwps->get_internal_world_id()] + ag_set_size);
		//std::cerr << "\nDEBUG: Added to " << it_pwps->get_internal_world_id() << " the label " << compact_indices[it_pwps->get_internal_world_id()] + ag_set_size << std::endl;
	}


	//BIS_ADAPTATION For the loop that identifies the id (We add one potential label for each node)
	int bhtabSize = ag_set_size + c;

	//std::cerr << "\nDEBUG: Inizializzazione Behavs\n";

	//BIS_ADAPTATION (Moved down here)

	for (it_peps = m_beliefs.begin(); it_peps != m_beliefs.end(); it_peps++) {

		for (auto it_mid_bel = it_peps->second.begin(); it_mid_bel != it_peps->second.end(); it_mid_bel++) {
			for (auto it_int_ed = it_mid_bel->second.begin(); it_int_ed != it_mid_bel->second.end(); it_int_ed++) {
				label_map[it_peps->first][*it_int_ed].insert(agent_to_label.at(it_mid_bel->first));
				Vertex[index_map[it_peps->first]].ne++;
			}
		}
	}

	i = 0;
	for (it_pwps = m_worlds.begin(); it_pwps != m_worlds.end(); it_pwps++) {
		Vertex[i].ne++; //Self loop bisimulation
		Vertex[i].e = (e_elem *) malloc(sizeof(e_elem) * Vertex[i].ne);
		i++;
	}

	//std::cerr << "\nDEBUG: Fine Inizializzazione Vertex\n";


	int from, to, j = 0; //, k = 0, nbh;

	//std::cerr << "\nDEBUG: Inizializzazione Mappa\n";
	for (it_plm = label_map.begin(); it_plm != label_map.end(); it_plm++) {
		from = index_map[it_plm->first]; // For each pworld 'from'

		//std::cerr << "\nDEBUG: Inizializzazione K\n";

		for (it_pw_bislab = it_plm->second.begin(); it_pw_bislab != it_plm->second.end(); it_pw_bislab++) { // For each edge that reaches the pworld 'to'
			to = index_map[it_pw_bislab->first];
			//nbh = it_pw_bislab->second.size();

			for (it_bislab = it_pw_bislab->second.begin(); it_bislab != it_pw_bislab->second.end(); it_bislab++) { // For each agent 'ag' in the label of the kedge
				//std::cerr << "\nDEBUG: j is: " << j << " and k is: " << k << "\n";
				//nbh = 1;
				Vertex[from].e[j].nbh = 1; // Let j be the index of the adjacency list of from that stores the kedge (from, to)
				Vertex[from].e[j].bh = (int *) malloc(sizeof(int)); // Let nbh be the number of agents in such kedge
				Vertex[from].e[j].tv = to; // Update the value of the reache pworld
				Vertex[from].e[j].bh[0] = *it_bislab; // Update the value of the label at index k to 'ag'
				//std::cerr << "\nDEBUG: j is: " << j << " and k is: " << k << "\n";

				j++; // Update the value of the index j
			}


		}

		j = 0; // Reset j
	}

	int Nbehavs = bhtabSize;
	a = (automa *) malloc(sizeof(automa));
	a->Nvertex = Nvertex;
	a->Nbehavs = Nbehavs;
	a->Vertex = Vertex;

	//	std::cerr << "\nDEBUG: \n\tNvertex = " << Nvertex << std::endl;
	//	std::cerr << "\tNbehavs = " << Nbehavs << std::endl;

	return *a;
}

void KripkeState::automaton_to_pstate(const automa & a, const std::vector<KripkeWorldPointer> & pworld_vec, const std::map<bis_label, Agent> & label_to_agent)
{
	//std::cerr << "\nDEBUG: INIZIO BISIMULATION TO KripkeState\n" << std::flush;

	KripkeWorldPointersSet worlds;
	m_beliefs.clear();
	// The pointed world does not change when we calculate the minimum bisimilar state
	// Hence we do not need to update it

	int i, j, k, label, agents_size = domain::get_instance().get_agents().size();

	for (i = 0; i < a.Nvertex; i++) {
		if (a.Vertex[i].ne > 0) {
			worlds.insert(pworld_vec[i]);
			for (j = 0; j < a.Vertex[i].ne; j++) {
				for (k = 0; k < a.Vertex[i].e[j].nbh; k++) {
					label = a.Vertex[i].e[j].bh[k];
					if (label < agents_size) {
						add_edge(pworld_vec[i], pworld_vec[a.Vertex[i].e[j].tv], label_to_agent.at(label));
					}
				}
			}
		}

	}


	set_worlds(worlds);
	//std::cerr << "\nDEBUG: FINE BISIMULATION TO KripkeState\n" << std::flush;

	//set_edges(edges);
}

void KripkeState::calc_min_bisimilar()
{
	//std::cerr << "\nDEBUG: INIZIO BISIMULATION IN KripkeState\n" << std::flush;

	// ************* Cleaning unreachable pworlds *************

	clean_unreachable_pworlds();

	std::vector<KripkeWorldPointer> pworld_vec; // Vector of all KripkeWorldPointer
	//std::cerr << "\nDEBUG: PRE-ALLOCAZIONE AUTOMA\n" << std::flush;


	//	std::cerr << "\nDEBUG: \n\tNvertex_before = " << m_worlds.size() << std::endl;
	//	std::cerr << "\tNbehavs_before = " << m_edges.size() << std::endl;

	automa a;
	pworld_vec.reserve(get_worlds().size());

	std::map<bis_label, Agent> label_to_agent;
	std::map<Agent, bis_label> agent_to_label;


	auto agents = domain::get_instance().get_agents();
	auto it_ag = agents.begin();
	bis_label ag_label = 0;
	Agent lab_agent;
	for (; it_ag != agents.end(); it_ag++) {
		lab_agent = *it_ag;
		label_to_agent.insert(std::make_pair(ag_label, lab_agent));
		agent_to_label.insert(std::make_pair(lab_agent, ag_label));
		ag_label++;

	}

	a = pstate_to_automaton(pworld_vec, agent_to_label);

	bisimulation b;
	//std::cout << "\nDEBUG: Printing automa pre-Bisimulation\n";
	//b.VisAutoma(&a);

	if (domain::get_instance().get_bisimulation() == PaigeTarjan) {
		if (b.MinimizeAutomaPT(&a)) {
			//VisAutoma(a);

			//std::cout << "\nDEBUG: Printing automa post-Bisimulation\n";
			//b.VisAutoma(&a);
			//std::cout << "Done\n";
			automaton_to_pstate(a, pworld_vec, label_to_agent);

			//b.DisposeAutoma(&a);
		}
	} else {
		if (b.MinimizeAutomaFB(&a)) {

			//std::cerr << "\nDEBUG: Printing automa post-Bisimulation\n";
			//b.VisAutoma(&a);
			//std::cerr << "Done\n";
			automaton_to_pstate(a, pworld_vec, label_to_agent);
			//b.DisposeAutoma(&a);
		}
	}

	//std::cerr << "\nDEBUG: PRe Clean" << std::endl;

}
/*********************END BISIMILUATION***********************/


//int KripkeState::get_edges() const
//{
//	std::cerr << "\nYou are playing with debug only options-1.\n";
//	exit(1);
//}
//
//void KripkeState::debug_print(KripkeState tmp)
//{
//	std::cerr << "\nYou are playing with debug only options-2.\n";
//	exit(1);
//}
