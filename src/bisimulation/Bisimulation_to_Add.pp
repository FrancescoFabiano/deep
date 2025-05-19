//
// Created by franc on 5/17/2025.
//

#include "Bisimulation.h"


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
