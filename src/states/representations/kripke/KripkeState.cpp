/*
 * \brief Implementation of \ref KripkeState.h
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 17, 2025
 */

#include <boost/dynamic_bitset.hpp>
#include <iostream>
#include <set>
#include <tuple>

#include "ArgumentParser.h"
#include "Domain.h"
#include "FormulaHelper.h"
#include "HelperPrint.h"
#include "InitialStateInformation.h"
#include "KripkeEntailmentHelper.h"
#include "KripkeReachabilityHelper.h"
#include "KripkeState.h"

#include <ranges>

#include "KripkeStorage.h"
#include "SetHelper.h"
#include "utilities/ExitHandler.h"

// --- Setters ---

void KripkeState::set_worlds(const KripkeWorldPointersSet &to_set) {
  m_worlds = to_set;
}

void KripkeState::set_pointed(const KripkeWorldPointer &to_set) {
  m_pointed = to_set;
}

void KripkeState::set_beliefs(const KripkeWorldPointersTransitiveMap &to_set) {
  m_beliefs = to_set;
}

void KripkeState::clear_beliefs() { m_beliefs.clear(); }

void KripkeState::set_max_depth(unsigned int to_set) noexcept {
  if (m_max_depth < to_set)
    m_max_depth = to_set;
}

// --- Getters ---

[[nodiscard]] const KripkeWorldPointersSet &
KripkeState::get_worlds() const noexcept {
  return m_worlds;
}

[[nodiscard]] const KripkeWorldPointer &
KripkeState::get_pointed() const noexcept {
  return m_pointed;
}

[[nodiscard]] const KripkeWorldPointersTransitiveMap &
KripkeState::get_beliefs() const noexcept {
  return m_beliefs;
}

[[nodiscard]] unsigned int KripkeState::get_max_depth() const noexcept {
  return m_max_depth;
}

// --- Operators ---

KripkeState &KripkeState::operator=(const KripkeState &to_copy) {
  set_worlds(to_copy.get_worlds());
  set_beliefs(to_copy.get_beliefs());
  m_max_depth = to_copy.get_max_depth();
  set_pointed(to_copy.get_pointed());
  return *this;
}

[[nodiscard]] bool KripkeState::operator<(const KripkeState &to_compare) const {
  if (m_pointed != to_compare.get_pointed())
    return m_pointed < to_compare.get_pointed();

  if (m_worlds != to_compare.get_worlds())
    return m_worlds < to_compare.get_worlds();

  const auto &beliefs1 = m_beliefs;
  const auto &beliefs2 = to_compare.get_beliefs();

  auto it1 = beliefs1.begin();
  auto it2 = beliefs2.begin();

  while (it1 != beliefs1.end() && it2 != beliefs2.end()) {
    if (it1->first != it2->first)
      return it1->first < it2->first;

    const auto &map1 = it1->second;
    const auto &map2 = it2->second;

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

void KripkeState::print() const {
  HelperPrint::get_instance().print_state(*this);
}

void KripkeState::print_dot_format(std::ofstream &ofs) const {
  HelperPrint::get_instance().print_dot_format(*this, ofs);
}

void KripkeState::print_dataset_format(std::ofstream &ofs) const {
  HelperPrint::print_dataset_format(*this, ofs);
}

// --- Structure Building ---

void KripkeState::add_world(const KripkeWorld &to_add) {
  m_worlds.insert(KripkeStorage::get_instance().add_world(to_add));
}

KripkeWorldPointer KripkeState::add_rep_world(const KripkeWorld &to_add,
                                              const unsigned short repetition,
                                              bool &is_new) {
  KripkeWorldPointer tmp = KripkeStorage::get_instance().add_world(to_add);
  tmp.set_repetition(repetition);
  is_new = std::get<1>(m_worlds.insert(tmp));
  return tmp;
}

KripkeWorldPointer KripkeState::add_rep_world(const KripkeWorld &to_add,
                                              unsigned short old_repetition) {
  bool tmp = false;
  return add_rep_world(to_add, get_max_depth() + old_repetition, tmp);
}

KripkeWorldPointer KripkeState::add_rep_world(const KripkeWorld &to_add) {
  bool tmp = false;
  return add_rep_world(to_add, get_max_depth(), tmp);
}

void KripkeState::add_edge(const KripkeWorldPointer &from,
                           const KripkeWorldPointer &to, const Agent &ag) {

  auto from_beliefs = m_beliefs.find(from);
  if (from_beliefs != m_beliefs.end()) {
    auto &beliefs_map = from_beliefs->second;
    auto ag_beliefs = beliefs_map.find(ag);
    if (ag_beliefs != beliefs_map.end()) {
      ag_beliefs->second.insert(to);
    } else {
      beliefs_map.emplace(ag, KripkeWorldPointersSet{to});
    }
  } else {
    KripkeWorldPointersMap pwm;
    pwm.emplace(ag, KripkeWorldPointersSet{to});
    m_beliefs.emplace(from, std::move(pwm));
  }
}

void KripkeState::add_world_beliefs(const KripkeWorldPointer &world,
                                    const KripkeWorldPointersMap &beliefs) {
  m_beliefs[world] = beliefs;
  /**TEMPORARY PATCH**/
  for (const auto &to_add : beliefs | std::views::values) {
    for (const auto &pw : to_add) {
      bool is_new = false;
      add_rep_world(KripkeWorld(pw.get_fluent_set()), pw.get_repetition(),is_new);
    }
  }
  /**END TEMPORARY PATCH**/

}

void KripkeState::build_initial() {
  FluentsSet permutation;
  const InitialStateInformation ini_conditions =
      Domain::get_instance().get_initial_description();
  generate_initial_worlds(permutation, 0,
                          ini_conditions.get_initially_known_fluents());
  generate_initial_edges();
}

void KripkeState::generate_initial_worlds(FluentsSet &permutation,
                                          const unsigned int index,
                                          const FluentsSet &initially_known) {
  auto const fluent_number = Domain::get_instance().get_fluent_number();
  auto const bit_size = Domain::get_instance().get_size_fluent();

  if (index == fluent_number) {
    const KripkeWorld to_add(permutation);
    add_initial_world(to_add);
    return;
  }

  FluentsSet permutation_2 = permutation;
  boost::dynamic_bitset<> bitSetToFindPositive(bit_size, index);
  boost::dynamic_bitset<> bitSetToFindNegative(bit_size, index);
  bitSetToFindNegative.set(bitSetToFindPositive.size() - 1, true);
  bitSetToFindPositive.set(bitSetToFindPositive.size() - 1, false);

  if (!initially_known.contains(bitSetToFindNegative)) {
    permutation.insert(bitSetToFindPositive);
    generate_initial_worlds(permutation, index + 1, initially_known);
  }
  if (!initially_known.contains(bitSetToFindPositive)) {
    permutation_2.insert(bitSetToFindNegative);
    generate_initial_worlds(permutation_2, index + 1, initially_known);
  }
}

void KripkeState::add_initial_world(const KripkeWorld &possible_add) {
  const InitialStateInformation ini_conditions =
      Domain::get_instance().get_initial_description();
  const auto &ff_forS5 = ini_conditions.get_ff_forS5();
  FluentFormula ff_forS5_nonempty;
  for (const auto &s : ff_forS5) {
    if (!s.empty()) {
      ff_forS5_nonempty.insert(s);
    }
  }
  if (ff_forS5_nonempty.empty() ||
      KripkeEntailmentHelper::entails(ff_forS5_nonempty, possible_add)) {
    add_world(possible_add);
    if (KripkeEntailmentHelper::entails(
            ini_conditions.get_pointed_world_conditions(), possible_add)) {
      m_pointed = KripkeWorldPointer(possible_add);
    }
  } else {
    KripkeStorage::get_instance().add_world(possible_add);
  }
}

void KripkeState::generate_initial_edges() {
  for (auto it_pwps_1 = m_worlds.begin(); it_pwps_1 != m_worlds.end();
       ++it_pwps_1) {
    for (auto it_pwps_2 = it_pwps_1; it_pwps_2 != m_worlds.end(); ++it_pwps_2) {
      for (const auto &agent : Domain::get_instance().get_agents()) {
        add_edge(*it_pwps_1, *it_pwps_2, agent);
        add_edge(*it_pwps_2, *it_pwps_1, agent);
      }
    }
  }

  const auto &ini_conditions = Domain::get_instance().get_initial_description();
  for (const auto &bf : ini_conditions.get_initial_conditions()) {
    remove_initial_edge_bf(bf);
  }
}

void KripkeState::remove_edge(const KripkeWorldPointer &from,
                              const KripkeWorldPointer &to, const Agent &ag) {
  auto from_beliefs = m_beliefs.find(from);
  if (from_beliefs != m_beliefs.end()) {
    auto ag_beliefs = from_beliefs->second.find(ag);
    if (ag_beliefs != from_beliefs->second.end()) {
      ag_beliefs->second.erase(to);
    }
  }
}

void KripkeState::remove_initial_edge(const FluentFormula &known_ff,
                                      const Agent &ag) {
  for (const auto &pwptr_tmp1 : m_worlds) {
    for (const auto &pwptr_tmp2 : m_worlds) {
      if (pwptr_tmp1 == pwptr_tmp2)
        continue;
      const bool entails1 =
          KripkeEntailmentHelper::entails(known_ff, pwptr_tmp1);
      const bool entails2 =
          KripkeEntailmentHelper::entails(known_ff, pwptr_tmp2);
      if (entails1 && !entails2) {
        remove_edge(pwptr_tmp1, pwptr_tmp2, ag);
        remove_edge(pwptr_tmp2, pwptr_tmp1, ag);
      } else if (entails2 && !entails1) {
        remove_edge(pwptr_tmp2, pwptr_tmp1, ag);
        remove_edge(pwptr_tmp1, pwptr_tmp2, ag);
      }
    }
  }
}

void KripkeState::remove_initial_edge_bf(const BeliefFormula &to_check) {
  if (to_check.get_formula_type() == BeliefFormulaType::C_FORMULA) {
    const BeliefFormula &tmp = to_check.get_bf1();
    switch (tmp.get_formula_type()) {
    case BeliefFormulaType::PROPOSITIONAL_FORMULA:
      if (tmp.get_operator() == BeliefFormulaOperator::BF_OR) {
        auto known_ff_ptr = FluentFormula();
        FormulaHelper::check_Bff_notBff(tmp.get_bf1(), tmp.get_bf2(),
                                        known_ff_ptr);
        if (!known_ff_ptr.empty()) {
          remove_initial_edge(known_ff_ptr, tmp.get_bf2().get_agent());
        }
      } else if (tmp.get_operator() != BeliefFormulaOperator::BF_AND) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::FormulaBadDeclaration,
            "Error: Invalid type of initial formula (FIFTH) in "
            "remove_initial_edge_bf.");
      }
      break;
    case BeliefFormulaType::FLUENT_FORMULA:
    case BeliefFormulaType::BELIEF_FORMULA:
    case BeliefFormulaType::BF_EMPTY:
      return;
    default:
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::FormulaBadDeclaration,
          "Error: Invalid type of initial formula (SIXTH) in "
          "remove_initial_edge_bf.");
    }
  } else {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::FormulaBadDeclaration,
                                   "Error: Invalid type of initial formula "
                                   "(SEVENTH) in remove_initial_edge_bf.");
  }
}

// --- Transition/Execution ---

KripkeState KripkeState::compute_successor(const Action &act) const {
  switch (act.get_type()) {
  case PropositionType::ONTIC:
    return execute_ontic(act);
  case PropositionType::SENSING:
    return execute_sensing(act);
  case PropositionType::ANNOUNCEMENT:
    return execute_announcement(act);
  default:
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::ActionTypeConflict,
        "Error: Executing an action with undefined type: " + act.get_name());
  }
  // Jut To please the compiler
  exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));
}

void KripkeState::maintain_oblivious_believed_worlds(
    KripkeState &ret, const AgentsSet &oblivious_obs_agents) const {
  if (!oblivious_obs_agents.empty()) {
    const auto tmp_world_set = KripkeReachabilityHelper::get_E_reachable_worlds(
        oblivious_obs_agents, get_pointed(), *this);
    KripkeWorldPointersSet world_oblivious;
    /*for (const auto &agent : Domain::get_instance().get_agents()) {
      for (const auto &wo_ob : tmp_world_set) {
        SetHelper::sum_set<KripkeWorldPointer>(
            world_oblivious, KripkeReachabilityHelper::get_B_reachable_worlds(
                                 agent, wo_ob, *this));
      }
    }*/
    KripkeReachabilityHelper::get_E_reachable_worlds_recursive(
                                 Domain::get_instance().get_agents(), tmp_world_set, world_oblivious, *this);



    SetHelper::sum_set<KripkeWorldPointer>(world_oblivious, tmp_world_set);
    //ret.set_max_depth(get_max_depth() + 1);
    ret.set_worlds(world_oblivious);

    for (const auto &wo_ob : world_oblivious) {
      auto it_pwmap = m_beliefs.find(wo_ob);
      if (it_pwmap != m_beliefs.end()) {
        ret.add_world_beliefs(wo_ob, it_pwmap->second);
      }
    }
  }
}

KripkeWorldPointer KripkeState::execute_ontic_helper(
    const Action &act, KripkeState &ret, const KripkeWorldPointer &current_pw,
    TransitionMap &calculated, AgentsSet &oblivious_obs_agents) const {
  FluentFormula current_pw_effects =
      FormulaHelper::get_effects_if_entailed(act.get_effects(), *this);
  FluentsSet world_description = current_pw.get_fluent_set();
  for (const auto &effect : current_pw_effects) {
    FormulaHelper::apply_effect(effect, world_description);
  }

  KripkeWorldPointer new_pw = ret.add_rep_world(KripkeWorld(world_description),
                                                current_pw.get_repetition());
  calculated.insert(TransitionMap::value_type(current_pw, new_pw));

  auto it_pwtm = get_beliefs().find(current_pw);

  if (it_pwtm != get_beliefs().end()) {
    for (const auto &[ag, beliefs] : it_pwtm->second) {
      bool is_oblivious_obs = oblivious_obs_agents.contains(ag);

      for (const auto &belief : beliefs) {
        if (is_oblivious_obs) {
          auto maintained_world = ret.get_worlds().find(belief);
          if (maintained_world != ret.get_worlds().end()) {
            ret.add_edge(new_pw, belief, ag);
          }
        } else {
          auto calculated_world = calculated.find(belief);
          if (calculated_world != calculated.end()) {
            ret.add_edge(new_pw, calculated_world->second, ag);
          } else {
            KripkeWorldPointer believed_pw = execute_ontic_helper(
                act, ret, belief, calculated, oblivious_obs_agents);
            ret.add_edge(new_pw, believed_pw, ag);
            ret.set_max_depth(ret.get_max_depth() + 1 +
                              current_pw.get_repetition());
          }
        }
      }
    }
  }

  return new_pw;
}

KripkeState KripkeState::execute_ontic(const Action &act) const {
  KripkeState ret;

  AgentsSet agents = Domain::get_instance().get_agents();
  AgentsSet fully_obs_agents =
      FormulaHelper::get_agents_if_entailed(act.get_fully_observants(), *this);

  AgentsSet oblivious_obs_agents = agents;
  SetHelper::minus_set<Agent>(oblivious_obs_agents, fully_obs_agents);

  TransitionMap calculated;
  maintain_oblivious_believed_worlds(ret, oblivious_obs_agents);

  KripkeWorldPointer new_pointed = execute_ontic_helper(
      act, ret, get_pointed(), calculated, oblivious_obs_agents);
  ret.set_pointed(new_pointed);

  return ret;
}

KripkeWorldPointer KripkeState::execute_sensing_announcement_helper(
    const FluentFormula &effects, KripkeState &ret,
    const KripkeWorldPointer &current_pw, TransitionMap &calculated,
    AgentsSet &partially_obs_agents, AgentsSet &oblivious_obs_agents,
    bool previous_entailment) const {
  KripkeWorldPointer new_pw = ret.add_rep_world(
      KripkeWorld(current_pw.get_fluent_set()), current_pw.get_repetition());
  calculated.insert(TransitionMap::value_type(current_pw, new_pw));

  auto it_pwtm = get_beliefs().find(current_pw);

  if (it_pwtm != get_beliefs().end()) {
    for (const auto &[ag, beliefs] : it_pwtm->second) {
      bool is_oblivious_obs = oblivious_obs_agents.contains(ag);
      bool is_partially_obs = partially_obs_agents.contains(ag);
      bool is_fully_obs = !is_oblivious_obs && !is_partially_obs;

      for (const auto &belief : beliefs) {
        if (is_oblivious_obs) {
          auto maintained_world = ret.get_worlds().find(belief);
          if (maintained_world != ret.get_worlds().end()) {
            ret.add_edge(new_pw, belief, ag);
          }
        } else {
          auto calculated_world = calculated.find(belief);
          bool ent = KripkeEntailmentHelper::entails(effects, belief);

          bool is_consistent_belief =
              is_partially_obs ||
              (is_fully_obs && (ent == previous_entailment));

          if (calculated_world != calculated.end()) {
            if (is_consistent_belief) {
              ret.add_edge(new_pw, calculated_world->second, ag);
            }
          } else {
            if (is_consistent_belief) {
              KripkeWorldPointer believed_pw =
                  execute_sensing_announcement_helper(
                      effects, ret, belief, calculated, partially_obs_agents,
                      oblivious_obs_agents, ent);
              ret.add_edge(new_pw, believed_pw, ag);
            }
          }
        }
      }
    }
  }
  return new_pw;
}

KripkeState KripkeState::execute_sensing(const Action &act) const {
  KripkeState ret;

  AgentsSet agents = Domain::get_instance().get_agents();
  AgentsSet fully_obs_agents =
      FormulaHelper::get_agents_if_entailed(act.get_fully_observants(), *this);
  AgentsSet partially_obs_agents = FormulaHelper::get_agents_if_entailed(
      act.get_partially_observants(), *this);

  AgentsSet oblivious_obs_agents = agents;
  SetHelper::minus_set<Agent>(oblivious_obs_agents, fully_obs_agents);
  SetHelper::minus_set<Agent>(oblivious_obs_agents, partially_obs_agents);

  if (!oblivious_obs_agents.empty()) {
    ret.set_max_depth(get_max_depth() + 1);
  }

  TransitionMap calculated;
  maintain_oblivious_believed_worlds(ret, oblivious_obs_agents);

  FluentFormula effects =
      FormulaHelper::get_effects_if_entailed(act.get_effects(), *this);

  KripkeWorldPointer new_pointed = execute_sensing_announcement_helper(
      effects, ret, get_pointed(), calculated, partially_obs_agents,
      oblivious_obs_agents,
      KripkeEntailmentHelper::entails(effects, get_pointed()));
  ret.set_pointed(new_pointed);

  return ret;
}

KripkeState KripkeState::execute_announcement(const Action &act) const {
  return execute_sensing(act);
}

bool KripkeState::entails(const Fluent &to_check) const {
  return KripkeEntailmentHelper::entails(to_check, get_pointed());
}

bool KripkeState::entails(const FluentsSet &to_check) const {
  return KripkeEntailmentHelper::entails(to_check, get_pointed());
}

bool KripkeState::entails(const FluentFormula &to_check) const {
  return KripkeEntailmentHelper::entails(to_check, get_pointed());
}

bool KripkeState::entails(const BeliefFormula &to_check) const {
  return KripkeEntailmentHelper::entails(to_check, *this);
}

bool KripkeState::entails(const FormulaeList &to_check) const {
  return KripkeEntailmentHelper::entails(to_check, *this);
}

void KripkeState::contract_with_bisimulation() {
  KripkeReachabilityHelper::clean_unreachable_worlds(*this);
  Bisimulation b;
  b.calc_min_bisimilar(*this);
}

// --- Constructors ---

KripkeState::KripkeState(const KripkeState &other)
    : m_max_depth(other.m_max_depth), m_worlds(other.m_worlds),
      m_pointed(other.m_pointed), m_beliefs(other.m_beliefs) {}
