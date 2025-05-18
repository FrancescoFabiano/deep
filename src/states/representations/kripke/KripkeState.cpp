/*
 * \brief Implementation of \ref KripkeState.h
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 17, 2025
 */

#include <iostream>
#include <tuple>
#include <boost/dynamic_bitset.hpp>
#include <boost/functional/hash.hpp>
#include <unordered_map>
#include <set>

#include "KripkeState.h"
#include "ArgumentParser.h"
#include "FormulaHelper.h"
#include "HelperPrint.h"
#include "KripkeStorage.h"
#include "SetHelper.h"
#include "utilities/ExitHandler.h"

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

void KripkeState::set_max_depth(unsigned int to_set) noexcept
{
	if (m_max_depth < to_set) m_max_depth = to_set;
}

const KripkeWorldPointersSet & KripkeState::get_worlds() const noexcept
{
	return m_worlds;
}

const KripkeWorldPointer & KripkeState::get_pointed() const noexcept
{
	return m_pointed;
}

const KripkeWorldPointersTransitiveMap & KripkeState::get_beliefs() const noexcept
{
	return m_beliefs;
}

unsigned int KripkeState::get_max_depth() const noexcept
{
	return m_max_depth;
}

KripkeState & KripkeState::operator=(const KripkeState & to_copy)
{
	set_worlds(to_copy.get_worlds());
	set_beliefs(to_copy.get_beliefs());
	m_max_depth = to_copy.get_max_depth();
	set_pointed(to_copy.get_pointed());
	return *this;
}

bool KripkeState::operator<(const KripkeState & to_compare) const
{
    if (m_pointed != to_compare.get_pointed())
        return m_pointed < to_compare.get_pointed();

    if (m_worlds != to_compare.get_worlds())
        return m_worlds < to_compare.get_worlds();

    const auto& beliefs1 = m_beliefs;
    const auto& beliefs2 = to_compare.get_beliefs();

    auto it1 = beliefs1.begin();
    auto it2 = beliefs2.begin();

    while (it1 != beliefs1.end() && it2 != beliefs2.end()) {
        if (it1->first != it2->first)
            return it1->first < it2->first;

        const auto& map1 = it1->second;
        const auto& map2 = it2->second;

        auto m1 = map1.begin();
        auto m2 = map2.begin();

        while (m1 != map1.end() && m2 != map2.end()) {
            if (m1->first != m2->first)
                return m1->first < m2->first;
            if (m1->second != m2->second)
                return m1->second < m2->second;
            ++m1;
            ++m2;
        }
        if (m1 != map1.end())
			return false;
        if (m2 != map2.end())
			return true;

        ++it1;
        ++it2;
		}
    return (it1 == beliefs1.end()) && (it2 != beliefs2.end());
}

[[nodiscard]] bool KripkeState::entails(const Fluent &f) const
{
	return entails(f, m_pointed);
}


[[nodiscard]] bool KripkeState::entails(const FluentsSet & fl) const
{
	return EntailmentHelper::entails(fl, m_pointed);
}

[[nodiscard]] bool KripkeState::entails(const FluentFormula & ff) const
{
	return EntailmentHelper::entails(ff, m_pointed);
}

[[nodiscard]] bool KripkeState::entails(const BeliefFormula & bf) const
{
	return EntailmentHelper::entails(bf, m_pointed);
}


void KripkeState::add_world(const KripkeWorld & world)
{
	m_worlds.insert(KripkeStorage::get_instance().add_world(world));
}

KripkeWorldPointer KripkeState::add_rep_world(const KripkeWorld & world, unsigned short repetition, bool& is_new)
{
	KripkeWorldPointer tmp = KripkeStorage::get_instance().add_world(world);
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
	auto from_beliefs = m_beliefs.find(from);
	if (from_beliefs != m_beliefs.end()) {
		auto ag_beliefs = from_beliefs->second.find(ag);
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
	std::cout << "\nBuilding initial possibility...\n";
	build_initial_prune();
}

void KripkeState::build_initial_prune()
{
	FluentsSet permutation;
	InitialStateInformation ini_conditions = Domain::get_instance().get_initial_description();
	generate_initial_pworlds(permutation, 0, ini_conditions.get_initially_known_fluents());
	generate_initial_pedges();
}

void KripkeState::generate_initial_pworlds(FluentsSet& permutation, int index, const FluentsSet & initially_known)
{
	int fluent_number = Domain::get_instance().get_fluent_number();
	int bit_size = (Domain::get_instance().get_size_fluent());

	if (index == fluent_number) {
		KripkeWorld to_add(permutation);
		add_initial_pworld(to_add);
		return;
	}

	FluentsSet permutation_2 = permutation;
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
	InitialStateInformation ini_conditions = Domain::get_instance().get_initial_description();
	if (possible_add.entails(ini_conditions.get_ff_forS5())) {
		add_world(possible_add);
		if (possible_add.entails(ini_conditions.get_pointed_world_conditions())) {
			m_pointed = KripkeWorldPointer(possible_add);
		}
	} else {
		KripkeStorage::get_instance().add_world(possible_add);
	}
}

void KripkeState::generate_initial_pedges()
{
	KripkeWorldPointersSet::const_iterator it_pwps_1, it_pwps_2;
	KripkeWorldPointer pwptr_tmp1, pwptr_tmp2;

	for (it_pwps_1 = m_worlds.begin(); it_pwps_1 != m_worlds.end(); it_pwps_1++) {
		for (it_pwps_2 = it_pwps_1; it_pwps_2 != m_worlds.end(); it_pwps_2++) {
			for (auto agent : Domain::get_instance().get_agents()) {
				pwptr_tmp1 = *it_pwps_1;
				pwptr_tmp2 = *it_pwps_2;
				add_edge(pwptr_tmp1, pwptr_tmp2, agent);
				add_edge(pwptr_tmp2, pwptr_tmp1, agent);
			}
		}
	}

	InitialStateInformation ini_conditions = Domain::get_instance().get_initial_description();
	for (auto it_fl = ini_conditions.get_initial_conditions().begin(); it_fl != ini_conditions.get_initial_conditions().end(); ++it_fl) {
		remove_initial_pedge_bf(*it_fl);
	}
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
	for (auto it_pwps_1 = m_worlds.begin(); it_pwps_1 != m_worlds.end(); it_pwps_1++) {
		for (auto it_pwps_2 = it_pwps_1; it_pwps_2 != m_worlds.end(); it_pwps_2++) {
			auto pwptr_tmp1 = *it_pwps_1;
			auto pwptr_tmp2 = *it_pwps_2;
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
	if (to_check.get_formula_type() == BeliefFormulaType::C_FORMULA) {
		BeliefFormula tmp = to_check.get_bf1();
		switch (tmp.get_formula_type()) {
		case BeliefFormulaType::PROPOSITIONAL_FORMULA:
			if (tmp.get_operator() == BeliefFormulaOperator::BF_OR) {
				auto known_ff_ptr = std::make_shared<FluentFormula>();
				FormulaHelper::check_Bff_notBff(tmp.get_bf1(), tmp.get_bf2(), known_ff_ptr);
				if (known_ff_ptr != nullptr) {
					remove_initial_pedge(*known_ff_ptr, tmp.get_bf2().get_agent());
				}
				return;
			} else if (tmp.get_operator() == BeliefFormulaOperator::BF_AND) {
				return;
			} else {
				ExitHandler::exit_with_message(
					ExitHandler::ExitCode::FormulaBadDeclaration,
					"Error: Invalid type of initial formula (FIFTH) in remove_initial_pedge_bf."
				);
			}
			break;
		case BeliefFormulaType::FLUENT_FORMULA:
		case BeliefFormulaType::BELIEF_FORMULA:
		case BeliefFormulaType::BF_EMPTY:
			return;
		default:
			ExitHandler::exit_with_message(
				ExitHandler::ExitCode::FormulaBadDeclaration,
				"Error: Invalid type of initial formula (SIXTH) in remove_initial_pedge_bf."
			);
		}
	} else {
		ExitHandler::exit_with_message(
			ExitHandler::ExitCode::FormulaBadDeclaration,
			"Error: Invalid type of initial formula (SEVENTH) in remove_initial_pedge_bf."
		);
	}
	return;
}

KripkeState KripkeState::compute_succ(const Action & act) const
{
	switch ( act.get_type() ) {
	case PropositionType::ONTIC:
		return execute_ontic(act);
	case PropositionType::SENSING:
		return execute_sensing(act);
	case PropositionType::ANNOUNCEMENT:
		return execute_announcement(act);
	default:
		ExitHandler::exit_with_message(
			ExitHandler::ExitCode::ActionTypeConflict,
			"Error: Executing an action with undefined type: " + act.get_name()
		);
	}
}

void KripkeState::maintain_oblivious_believed_pworlds(KripkeState &ret, const AgentsSet & oblivious_obs_agents) const
{
	if (!oblivious_obs_agents.empty()) {
		auto tmp_world_set = get_E_reachable_worlds(oblivious_obs_agents, get_pointed());
		KripkeWorldPointersSet world_oblivious;
		for (auto it_agset = Domain::get_instance().get_agents().begin(); it_agset != Domain::get_instance().get_agents().end(); it_agset++) {
			for (auto it_wo_ob = tmp_world_set.begin(); it_wo_ob != tmp_world_set.end(); it_wo_ob++) {
				SetHelper::sum_set<KripkeWorldPointer>(world_oblivious, get_B_reachable_worlds(*it_agset, *it_wo_ob));
			}
		}
		SetHelper::sum_set<KripkeWorldPointer>(world_oblivious, tmp_world_set);
		ret.set_max_depth(get_max_depth() + 1);
		ret.set_worlds(world_oblivious);

		for (auto it_wo_ob = world_oblivious.begin(); it_wo_ob != world_oblivious.end(); it_wo_ob++) {
			auto it_pwmap = m_beliefs.find(*it_wo_ob);
			if (it_pwmap != m_beliefs.end()) {
				ret.add_pworld_beliefs(*it_wo_ob, it_pwmap->second);
			}
		}
	}
}

KripkeWorldPointer KripkeState::execute_ontic_helper(const Action &act, KripkeState &ret, const KripkeWorldPointer &current_pw, TransitionMap &calculated, AgentsSet & oblivious_obs_agents) const
{
	FluentFormula current_pw_effects = FormulaHelper::get_effects_if_entailed(act.get_effects(), *this);
	FluentsSet world_description = current_pw.get_fluent_set();
	for (const auto & effect : current_pw_effects) {
		FormulaHelper::apply_effect(effect, world_description);
	}

	KripkeWorldPointer new_pw = ret.add_rep_world(KripkeWorld(world_description), current_pw.get_repetition());
	calculated.insert(TransitionMap::value_type(current_pw, new_pw));

	auto it_pwtm = get_beliefs().find(current_pw);

	if (it_pwtm != get_beliefs().end()) {
		for (const auto & [ag, beliefs] : it_pwtm->second) {
			bool is_oblivious_obs = oblivious_obs_agents.find(ag) != oblivious_obs_agents.end();

			for (const auto & belief : beliefs) {
				if (is_oblivious_obs) {
					auto maintained_pworld = ret.get_worlds().find(belief);
					if (maintained_pworld != ret.get_worlds().end()) {
						ret.add_edge(new_pw, belief, ag);
					}
				} else {
					auto calculated_pworld = calculated.find(belief);
					if (calculated_pworld != calculated.end()) {
						ret.add_edge(new_pw, calculated_pworld->second, ag);
					} else {
						KripkeWorldPointer believed_pw = execute_ontic_helper(act, ret, belief, calculated, oblivious_obs_agents);
						ret.add_edge(new_pw, believed_pw, ag);
						ret.set_max_depth(ret.get_max_depth() + 1 + current_pw.get_repetition());
					}
				}
			}
		}
	}

	return new_pw;
}

KripkeState KripkeState::execute_ontic(const Action & act) const
{
	KripkeState ret;

	AgentsSet agents = Domain::get_instance().get_agents();
	AgentsSet fully_obs_agents = FormulaHelper::get_agents_if_entailed(act.get_fully_observants(), *this);

	AgentsSet oblivious_obs_agents = agents;
	SetHelper::minus_set<Agent>(oblivious_obs_agents, fully_obs_agents);

	TransitionMap calculated;
	maintain_oblivious_believed_pworlds(ret, oblivious_obs_agents);

	KripkeWorldPointer new_pointed = execute_ontic_helper(act, ret, get_pointed(), calculated, oblivious_obs_agents);
	ret.set_pointed(new_pointed);

	return ret;
}

KripkeWorldPointer KripkeState::execute_sensing_announcement_helper(const FluentFormula &effects, KripkeState &ret, const KripkeWorldPointer &current_pw, TransitionMap &calculated, AgentsSet &partially_obs_agents, AgentsSet &oblivious_obs_agents, bool previous_entailment) const
{
	KripkeWorldPointer new_pw = ret.add_rep_world(KripkeWorld(current_pw.get_fluent_set()), current_pw.get_repetition());
	calculated.insert(TransitionMap::value_type(current_pw, new_pw));

	auto it_pwtm = get_beliefs().find(current_pw);

	if (it_pwtm != get_beliefs().end()) {
		for (const auto & [ag, beliefs] : it_pwtm->second) {
			bool is_oblivious_obs = oblivious_obs_agents.find(ag) != oblivious_obs_agents.end();
			bool is_partially_obs = partially_obs_agents.find(ag) != partially_obs_agents.end();
			bool is_fully_obs = !is_oblivious_obs && !is_partially_obs;

			for (const auto & belief : beliefs) {
				if (is_oblivious_obs) {
					auto maintained_pworld = ret.get_worlds().find(belief);
					if (maintained_pworld != ret.get_worlds().end()) {
						ret.add_edge(new_pw, belief, ag);
					}
				} else {
					auto calculated_pworld = calculated.find(belief);
					bool ent = entails(effects, belief);

					bool is_consistent_belief = is_partially_obs || (is_fully_obs && (ent == previous_entailment));

					if (calculated_pworld != calculated.end()) {
						if (is_consistent_belief) {
							ret.add_edge(new_pw, calculated_pworld->second, ag);
						}
					} else {
						if (is_consistent_belief) {
							KripkeWorldPointer believed_pw = execute_sensing_announcement_helper(effects, ret, belief, calculated, partially_obs_agents, oblivious_obs_agents, ent);
							ret.add_edge(new_pw, believed_pw, ag);
						}
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

	AgentsSet agents = Domain::get_instance().get_agents();
	AgentsSet fully_obs_agents = FormulaHelper::get_agents_if_entailed(act.get_fully_observants(), *this);
	AgentsSet partially_obs_agents = FormulaHelper::get_agents_if_entailed(act.get_partially_observants(), *this);

	AgentsSet oblivious_obs_agents = agents;
	SetHelper::minus_set<Agent>(oblivious_obs_agents, fully_obs_agents);
	SetHelper::minus_set<Agent>(oblivious_obs_agents, partially_obs_agents);

	if (!oblivious_obs_agents.empty()) {
		ret.set_max_depth(get_max_depth() + 1);
	}

	TransitionMap calculated;
	maintain_oblivious_believed_pworlds(ret, oblivious_obs_agents);

	FluentFormula effects = FormulaHelper::get_effects_if_entailed(act.get_effects(), *this);

	KripkeWorldPointer new_pointed = execute_sensing_announcement_helper(effects, ret, get_pointed(), calculated, partially_obs_agents, oblivious_obs_agents, entails(effects));
	ret.set_pointed(new_pointed);

	return ret;
}

KripkeState KripkeState::execute_announcement(const Action & act) const
{
	return execute_sensing(act);
}

[[nodiscard]] bool KripkeState::check_properties(const AgentsSet & fully, const AgentsSet & partially, const FluentFormula & effects, const KripkeState & updated) const
{
	if (!fully.empty()) {
		BeliefFormula effects_formula;
		effects_formula.set_formula_type(BeliefFormulaType::FLUENT_FORMULA);
		effects_formula.set_fluent_formula(effects);

		BeliefFormula property1;
		property1.set_group_agents(fully);
		property1.set_formula_type(BeliefFormulaType::C_FORMULA);
		property1.set_bf1(effects_formula);

		if (!updated.entails(property1)) {
			std::cerr << "\nDEBUG: First property not respected";
			return false;
		}

		if (!partially.empty()) {
			BeliefFormula inner_nested2, nested2, disjunction, property2;
			inner_nested2.set_group_agents(fully);
			inner_nested2.set_formula_type(BeliefFormulaType::C_FORMULA);
			inner_nested2.set_bf1(effects_formula);

			nested2.set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);
			nested2.set_operator(BeliefFormulaOperator::BF_NOT);
			nested2.set_bf1(inner_nested2);

			disjunction.set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);
			disjunction.set_operator(BeliefFormulaOperator::BF_OR);
			disjunction.set_bf1(property1);
			disjunction.set_bf2(nested2);

			property2.set_group_agents(partially);
			property2.set_formula_type(BeliefFormulaType::C_FORMULA);
			property2.set_bf1(disjunction);

			BeliefFormula property3;
			property3.set_group_agents(fully);
			property3.set_formula_type(BeliefFormulaType::C_FORMULA);
			property3.set_bf1(property2);

			if (!updated.entails(property2)) {
				std::cerr << "\nDEBUG: Second property not respected in the formula: ";
				return false;
			}
			if (!updated.entails(property3)) {
				std::cerr << "\nDEBUG: Third property not respected in the formula: ";
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
	HelperPrint::get_instance().print_list(get_pointed().get_fluent_set());
	std::cout << "-" << get_pointed().get_repetition();
	std::cout << std::endl;
	std::cout << "*******************************************************************" << std::endl;

	KripkeWorldPointersSet::const_iterator it_pwset;
	std::cout << "World List:" << std::endl;

	for (it_pwset = get_worlds().begin(); it_pwset != get_worlds().end(); it_pwset++) {
		std::cout << "W-" << counter << ": ";
		HelperPrint::get_instance().print_list(it_pwset->get_fluent_set());
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
				HelperPrint::get_instance().print_list(from.get_fluent_set());
				std::cout << "," << from.get_repetition();
				std::cout << ") - (";
				HelperPrint::get_instance().print_list(to.get_fluent_set());
				std::cout << "," << to.get_repetition();
				std::cout << ") ag:" << Domain::get_instance().get_grounder().deground_agent(ag);
				std::cout << std::endl;
				counter++;
			}
		}
	}
	std::cout << "*******************************************************************" << std::endl;
}

void KripkeState::print_ML_dataset(std::ostream & graphviz) const
{
	graphviz << "digraph G {" << std::endl;

	std::unordered_map<std::size_t, int> world_map;
	int world_counter = 1;

	for (const auto& pw : get_worlds()) {
		std::pair<std::set<Fluent>, int> world_key = {pw.get_fluent_set(), pw.get_repetition()};
		std::size_t hash_value = WorldHash{}(world_key);

		if (world_map.find(hash_value) == world_map.end()) {
			world_map[hash_value] = world_counter++;
		}
	}

	for (const auto& [hash, id] : world_map) {
		graphviz << to_base36(id) << ";" << std::endl;
	}

	{
		std::pair<std::set<Fluent>, int> pointed_key = {get_pointed().get_fluent_set(), get_pointed().get_repetition()};
		std::size_t pointed_hash = WorldHash{}(pointed_key);
		graphviz <<  to_base36(world_map[pointed_hash]) << " [shape=doublecircle];" << std::endl;
	}

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

	for (const auto& [edge, agents] : edge_map) {
		graphviz <<  to_base36(edge.first)
				  << " ->" << to_base36(edge.second)
				  << " [label=\"";

		bool first_ag = true;
		for (const auto& ag : agents) {
			if (!first_ag) graphviz << ",";
			graphviz << Domain::get_instance().get_grounder().deground_agent(ag);
			first_ag = false;
		}

		graphviz << "\"];" << std::endl;
	}

	graphviz << "}" << std::endl;
}

void KripkeState::print_graphviz(std::ostream & graphviz) const
{
	StringsSet::const_iterator it_st_set;
	FluentsSet::const_iterator it_fs;

	graphviz << "//WORLDS List:" << std::endl;
	std::map<FluentsSet, int> map_world_to_index;
	std::map<unsigned short, char> map_rep_to_name;
	char found_rep = (char) ((char) Domain::get_instance().get_agents().size() + 'A');
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
		tmp_stset = Domain::get_instance().get_grounder().deground_fluent(tmp_fs);
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

				edges[tmp_tuple].insert(Domain::get_instance().get_grounder().deground_agent(ag));
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
			if (FormulaHelper::is_negated(*it_fs)) color = "<font color=\"#0000ff\"> ";
			else color = "<font color=\"#ff1020\">";
			graphviz << color << Domain::get_instance().get_grounder().deground_fluent(*it_fs) << "</font>";
		}
		graphviz << "</td></tr>\n";
	}
	graphviz << "	</table>>]\n";
	graphviz << "	{rank = max; description};\n";
}

/****************BISIMULATION********************/
void KripkeState::get_all_reachable_worlds(const KripkeWorldPointer & pw, KripkeWorldPointersSet & reached_worlds, KripkeWorldPointersTransitiveMap & reached_edges) const
{
	KripkeWorldPointersSet::const_iterator it_pwps;
	KripkeWorldPointersSet pw_list;

	auto ag_set = Domain::get_instance().get_agents();
	auto ag_it = ag_set.begin();
	for (; ag_it != ag_set.end(); ag_it++) {
		try {
			pw_list = m_beliefs.at(pw).at(*ag_it);
		} catch (const std::out_of_range& e) {
			pw_list.clear();
		}

		for (it_pwps = pw_list.begin(); it_pwps != pw_list.end(); it_pwps++) {
			if (reached_worlds.insert(*it_pwps).second) {
				get_all_reachable_worlds(*it_pwps, reached_worlds, reached_edges);
				try {
					reached_edges.insert(std::make_pair(*it_pwps, m_beliefs.at(*it_pwps)));
				} catch (const std::out_of_range& e) {
				}
			}
		}
	}
}

void KripkeState::clean_unreachable_pworlds()
{
	KripkeWorldPointersSet reached_worlds;
	KripkeWorldPointersTransitiveMap reached_edges;

	reached_worlds.insert(get_pointed());
	try {
		reached_edges.insert(std::make_pair(get_pointed(), m_beliefs.at(get_pointed())));
	} catch (const std::out_of_range& e) {
	}

	get_all_reachable_worlds(get_pointed(), reached_worlds, reached_edges);

	set_worlds(reached_worlds);
	set_beliefs(reached_edges);
}

automa KripkeState::pstate_to_automaton(std::vector<KripkeWorldPointer> & pworld_vec, const std::map<Agent, bis_label> & agent_to_label) const
{
	std::map<int, int> compact_indices;
	std::map<KripkeWorldPointer, int> index_map;
	pbislabel_map label_map;

	automa *a;
	int Nvertex = get_worlds().size();
	int ag_set_size = Domain::get_instance().get_agents().size();
	v_elem *Vertex;

	Vertex = (v_elem *) malloc(sizeof(v_elem) * Nvertex);

	KripkeWorldPointersSet::const_iterator it_pwps;
	KripkeWorldPointersTransitiveMap::const_iterator it_peps;
	pbislabel_map::const_iterator it_plm;
	bis_label_set::const_iterator it_bislab;
	std::map<KripkeWorldPointer, bis_label_set>::const_iterator it_pw_bislab;

	index_map[get_pointed()] = 0;
	pworld_vec.push_back(get_pointed());
	compact_indices[get_pointed().get_internal_world_id()] = 0;

	Vertex[0].ne = 0;

	int i = 1, c = 1;

	for (it_pwps = m_worlds.begin(); it_pwps != m_worlds.end(); it_pwps++) {
		if (!(*it_pwps == get_pointed())) {
			index_map[*it_pwps] = i;
			pworld_vec.push_back(*it_pwps);

			if (compact_indices.insert({it_pwps->get_internal_world_id(), c}).second) {
				c++;
			}
			Vertex[i].ne = 0;
			i++;
		}
		label_map[*it_pwps][*it_pwps].insert(compact_indices[it_pwps->get_internal_world_id()] + ag_set_size);
	}

	int bhtabSize = ag_set_size + c;

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
		Vertex[i].ne++;
		Vertex[i].e = (e_elem *) malloc(sizeof(e_elem) * Vertex[i].ne);
		i++;
	}

	int from, to, j = 0;

	for (it_plm = label_map.begin(); it_plm != label_map.end(); it_plm++) {
		from = index_map[it_plm->first];

		for (it_pw_bislab = it_plm->second.begin(); it_pw_bislab != it_plm->second.end(); it_pw_bislab++) {
			to = index_map[it_pw_bislab->first];

			for (it_bislab = it_pw_bislab->second.begin(); it_bislab != it_pw_bislab->second.end(); it_bislab++) {
				Vertex[from].e[j].nbh = 1;
				Vertex[from].e[j].bh = (int *) malloc(sizeof(int));
				Vertex[from].e[j].tv = to;
				Vertex[from].e[j].bh[0] = *it_bislab;

				j++;
			}
		}

		j = 0;
	}

	int Nbehavs = bhtabSize;
	a = (automa *) malloc(sizeof(automa));
	a->Nvertex = Nvertex;
	a->Nbehavs = Nbehavs;
	a->Vertex = Vertex;

	return *a;
}

void KripkeState::automaton_to_pstate(const automa & a, const std::vector<KripkeWorldPointer> & pworld_vec, const std::map<bis_label, Agent> & label_to_agent)
{
	KripkeWorldPointersSet worlds;
	m_beliefs.clear();

	int i, j, k, label, agents_size = Domain::get_instance().get_agents().size();

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
}

void KripkeState::calc_min_bisimilar()
{
	clean_unreachable_pworlds();

	std::vector<KripkeWorldPointer> pworld_vec;
	pworld_vec.reserve(get_worlds().size());

	std::map<bis_label, Agent> label_to_agent;
	std::map<Agent, bis_label> agent_to_label;

	auto agents = Domain::get_instance().get_agents();
	auto it_ag = agents.begin();
	bis_label ag_label = 0;
	Agent lab_agent;
	for (; it_ag != agents.end(); it_ag++) {
		lab_agent = *it_ag;
		label_to_agent.insert(std::make_pair(ag_label, lab_agent));
		agent_to_label.insert(std::make_pair(lab_agent, ag_label));
		ag_label++;
	}

	automa a = pstate_to_automaton(pworld_vec, agent_to_label);

	bisimulation b;

	if (ArgumentParser::get_instance().get_bisimulation()) {
		if (b.MinimizeAutomaPT(&a)) {
			automaton_to_pstate(a, pworld_vec, label_to_agent);
		}
	}
}
