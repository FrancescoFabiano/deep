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
#include <unordered_set>

#include "KripkeEqualityHelper.h"
#include "KripkeStorage.h"
#include "SetHelper.h"
#include "utilities/ExitHandler.h"

#ifdef USE_NEURALNETS
#include "neuralnets/GraphNN.h"
#endif

// --- Setters ---

void KripkeState::set_worlds(const KripkeWorldPointersSet &to_set) {
  m_worlds = to_set;
  set_worlds_vec();
}

void KripkeState::set_worlds_vec() {
  m_worlds_vec = KripkeEqualityHelper::canonicalize_worlds(m_worlds);
}

void KripkeState::set_designated_worlds(
    const KripkeWorldPointersSet &to_set) {
  m_designated_worlds = to_set;
}

void KripkeState::add_designated_world(
    const KripkeWorldPointer &to_add) {
  m_designated_worlds.insert(to_add);
}

void KripkeState::set_beliefs(
    const KripkeWorldPointersTransitiveMap &to_set) {

  m_beliefs = to_set;
  set_beliefs_vec();

}

void KripkeState::set_beliefs_vec() {
  m_beliefs_vec = KripkeEqualityHelper::canonicalize_transitive_map(m_beliefs);
}

void KripkeState::clear_beliefs() {
  m_beliefs.clear();
}

// --- Getters ---

[[nodiscard]] const KripkeWorldPointersSet &
KripkeState::get_worlds() const noexcept {
  return m_worlds;
}

const KripkeWorldPointersVec &KripkeState::get_worlds_vec() const noexcept {
  return m_worlds_vec;
}

[[nodiscard]] const KripkeWorldPointersSet &
KripkeState::get_designated_worlds() const noexcept {
  return m_designated_worlds;
}

[[nodiscard]] bool KripkeState::is_designated(
    const KripkeWorldPointer &world) const noexcept {
  return m_designated_worlds.contains(world);
}

[[nodiscard]] const KripkeWorldPointersTransitiveMap &
KripkeState::get_beliefs() const noexcept {
  return m_beliefs;
}

const KripkeWorldPointersTransitiveMapVec &
KripkeState::get_beliefs_vec() const noexcept {
  return m_beliefs_vec;
}

// --- Operators ---

KripkeState &KripkeState::operator=(const KripkeState &to_copy) {
  if (this != &to_copy) {
    m_worlds = to_copy.m_worlds;
    m_designated_worlds = to_copy.m_designated_worlds;
    m_beliefs = to_copy.m_beliefs;
    m_worlds_vec = to_copy.m_worlds_vec;
    m_beliefs_vec = to_copy.m_beliefs_vec;
    m_hash = to_copy.m_hash;
  }

  return *this;
}

bool KripkeState::operator==(
    const KripkeState &to_compare) const {
  return !(*this < to_compare) &&
         !(to_compare < *this);
}

bool KripkeState::operator<(
    const KripkeState &to_compare) const {
  return KripkeEqualityHelper::less_operator(
      *this,
      to_compare);
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

void KripkeState::add_world(
    const KripkeWorld &to_add) {

  m_worlds.insert(
          KripkeStorage::get_instance().add_world(to_add));
}

KripkeWorldPointer KripkeState::add_rep_world(
    const KripkeWorld &to_add,
    const unsigned short repetition) {

  KripkeWorldPointer tmp =
      KripkeStorage::get_instance().add_world(to_add);

  tmp.set_repetition(repetition);
  m_worlds.insert(tmp);

  return tmp;
}

void KripkeState::add_edge(const KripkeWorldPointer &from,
                           const KripkeWorldPointer &to, const Agent &ag) {
  auto from_beliefs = m_beliefs.find(from);

  if (from_beliefs != m_beliefs.end()) {
    auto &beliefs_map = from_beliefs->second;
    auto ag_beliefs = beliefs_map.find(ag);

    if (ag_beliefs != beliefs_map.end()) {
      if (ag_beliefs->second.insert(to).second) {
      }
    } else {
      beliefs_map.emplace(ag, KripkeWorldPointersSet{to});
    }
  } else {
    KripkeWorldPointersMap pwm;
    pwm.emplace(ag, KripkeWorldPointersSet{to});
    m_beliefs.emplace(from, std::move(pwm));
  }
}


void KripkeState::build_initial() {
  FluentsSet permutation;
  const InitialStateInformation ini_conditions =
      Domain::get_instance().get_initial_description();
  generate_initial_worlds(permutation, 0,
                          ini_conditions.get_initially_known_fluents());
  generate_initial_edges();
  recompute_hash();
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
      add_designated_world(
          KripkeWorldPointer(possible_add));
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
  if (from_beliefs == m_beliefs.end()) {
    return;
  }

  auto ag_beliefs = from_beliefs->second.find(ag);
  if (ag_beliefs == from_beliefs->second.end()) {
    return;
  }

  ag_beliefs->second.erase(to);
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


void KripkeState::recompute_hash() {
  m_hash = FormulaHelper::hash_kripke_state(*this);
}

uint64_t KripkeState::get_hash() const noexcept {
  return m_hash;
}


// --- Transition ---

KripkeState KripkeState::compute_successor(
    const Action &action) const {


#ifdef DEBUG

  if (m_designated_worlds.empty() ||
      action.get_designated_events().empty()) {

    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::StateActionNotExecutableError,
        "Action '" + action.get_name() +
            "' cannot be applied: missing designated worlds "
            "or designated events.");
      }

  for (const auto &world : m_designated_worlds) {

    bool has_applicable_event = false;

    for (const auto event_id :
         action.get_designated_events()) {

      const Event &event =
          action.get_event(event_id);

      if (KripkeEntailmentHelper::entails(
              event.get_precondition(),
              world,
              *this)) {

        has_applicable_event = true;
        break;
              }
         }

    if (!has_applicable_event) {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::StateActionNotExecutableError,
          "Action '" + action.get_name() +
              "' is not executable in one of the designated worlds.");
    }
  }

#endif


  using ProductWorld =
      std::pair<KripkeWorldPointer, EventId>;

  KripkeState successor;

  /*
   * Exact semantic identity of a product world:
   *
   *     (source world, event)
   *
   * This map, rather than the repetition number, determines
   * whether a product world has already been created.
   */
  std::map<ProductWorld, KripkeWorldPointer> product_worlds;

  /*
   * Product worlds whose outgoing accessibility edges still
   * need to be expanded.
   */
  std::queue<ProductWorld> pending;

  /*
   * Repetition is only implementation metadata used by
   * KripkeWorldPointer. It no longer encodes the source-world
   * depth as in the old transition function.
   */
  unsigned short next_repetition = 0;


  // --------------------------------------------------------------------------
  // Helpers
  // --------------------------------------------------------------------------

  const auto is_applicable =
      [&](const KripkeWorldPointer &world,
          const Event &event) {

        return KripkeEntailmentHelper::entails(
            event.get_precondition(),
            world,
            *this);
      };


  const auto apply_postconditions =
      [&](const KripkeWorldPointer &world,
          const Event &event) {

        FluentsSet description =
            world.get_fluent_set();

        /*
         * DEL postconditions have the form:
         *
         *     p -> phi
         *
         * where phi is evaluated in the source world.
         */
        for (const auto &[fluent, postcondition] :
             event.get_postconditions()) {

          const bool value =
              KripkeEntailmentHelper::entails(
                  postcondition,
                  world,
                  *this);

          /*
           * Deep stores fluents with polarity.
           * Normalize the key to the positive fluent first.
           */
          const Fluent positive_fluent =
              FormulaHelper::is_negated(fluent)
                  ? FormulaHelper::negate_fluent(fluent)
                  : fluent;

          description.erase(positive_fluent);
          description.erase(
              FormulaHelper::negate_fluent(
                  positive_fluent));

          description.insert(
              value
                  ? positive_fluent
                  : FormulaHelper::negate_fluent(
                        positive_fluent));
        }

        return description;
      };


  const auto get_or_create_product_world =
      [&](const KripkeWorldPointer &source_world,
          const Event &event) {

        const ProductWorld product{
            source_world,
            event.get_id()
        };

        /*
         * Exact check: this is the actual product-node identity.
         */
        const auto existing =
            product_worlds.find(product);

        if (existing != product_worlds.end()) {
          return existing->second;
        }

        /*
         * Create the valuation of (source_world, event).
         */
        const FluentsSet description =
            apply_postconditions(
                source_world,
                event);

        /*
         * Give the new KripkeWorld a fresh repetition.
         */
        const unsigned short repetition =
            next_repetition++;

        const KripkeWorldPointer product_world =
            successor.add_rep_world(
                KripkeWorld(description),
                repetition);

        /*
         * Store the exact correspondence between:
         *
         *     (source world, event)
         *
         * and
         *
         *     successor world
         */
        product_worlds.emplace(
            product,
            product_world);

        /*
         * Its outgoing product edges still need to be generated.
         */
        pending.push(product);

        return product_world;
      };


  // --------------------------------------------------------------------------
  // 1. Create designated successor worlds
  // --------------------------------------------------------------------------

  /*
   * The designated points of the successor are:
   *
   *     (w, e)
   *
   * where w is designated in the source state,
   * e is designated in the action,
   * and e is applicable in w.
   */
  for (const auto &source_world :
       m_designated_worlds) {

    for (const EventId event_id :
         action.get_designated_events()) {

      const Event &event =
          action.get_event(event_id);

      if (!is_applicable(
              source_world,
              event)) {
        continue;
      }

      const KripkeWorldPointer product_world =
          get_or_create_product_world(
              source_world,
              event);

      successor.add_designated_world(
          product_world);
    }
  }


  // --------------------------------------------------------------------------
  // 2. Expand the DEL product relation
  // --------------------------------------------------------------------------

  /*
   * For every product world (w,e), construct:
   *
   *     (w,e) R'_a (v,f)
   *
   * exactly when:
   *
   *     w R_a v
   *
   * and
   *
   *     e R^A_a f
   *
   * and f is applicable in v.
   */
  while (!pending.empty()) {

    const ProductWorld current =
        pending.front();

    pending.pop();

    const KripkeWorldPointer &source_world =
        current.first;

    const Event &source_event =
        action.get_event(
            current.second);

    const KripkeWorldPointer& product_source =
        product_worlds.at(current);


    /*
     * Get the original Kripke successors of w.
     */
    const auto source_beliefs =
        m_beliefs.find(source_world);

    if (source_beliefs == m_beliefs.end()) {
      continue;
    }


    for (const auto &[agent, world_targets] :
         source_beliefs->second) {

      /*
       * Get the event accessibility relation for agent a.
       */
      const auto &event_relation =
          action.get_event_relation(agent);


      for (const auto &target_world :
           world_targets) {

        /*
         * Pair the current event e with every event f that
         * is accessible from e for this agent.
         */
        for (const auto &[from_event, to_event] :
             event_relation) {

          if (from_event !=
              source_event.get_id()) {
            continue;
          }

          const Event &target_event =
              action.get_event(to_event);

          /*
           * (v,f) exists only when f's precondition holds in v.
           */
          if (!is_applicable(
                  target_world,
                  target_event)) {
            continue;
          }

          const KripkeWorldPointer product_target =
              get_or_create_product_world(
                  target_world,
                  target_event);

          successor.add_edge(
              product_source,
              product_target,
              agent);
        }
      }
    }
  }

  successor.recompute_hash();
  return successor;
}

bool KripkeState::entails(const Fluent &to_check) const {
  if (m_designated_worlds.empty()) {
    return false;
  }

  for (const auto &world : m_designated_worlds) {
    if (!KripkeEntailmentHelper::entails(to_check, world)) {
      return false;
    }
  }

  return true;
}

bool KripkeState::entails(const FluentsSet &to_check) const {
  if (m_designated_worlds.empty()) {
    return false;
  }

  for (const auto &world : m_designated_worlds) {
    if (!KripkeEntailmentHelper::entails(to_check, world)) {
      return false;
    }
  }

  return true;
}

bool KripkeState::entails(const FluentFormula &to_check) const {
  if (m_designated_worlds.empty()) {
    return false;
  }

  for (const auto &world : m_designated_worlds) {
    if (!KripkeEntailmentHelper::entails(to_check, world)) {
      return false;
    }
  }

  return true;
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
  recompute_hash();
}

const GraphTensor &KripkeState::get_tensor_representation() {
#ifdef USE_NEURALNETS
  if (!m_computed_tensor_representation) {
    m_tensor_representation =
        GraphNN<KripkeState>::get_instance().state_to_tensor_minimal(*this);
    m_computed_tensor_representation = true;
  }
  return m_tensor_representation;
#else
  ExitHandler::exit_with_message(
      ExitHandler::ExitCode::HeuristicsBadDeclaration,
      "Trying to create a tensor of a state but neural network support (onnx "
      "handler) is "
      "not "
      "enabled or linked. Please recompile with the nn option.");
  // This line will never be reached, but added to avoid compiler warning.
  std::exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));
#endif
}

// --- Constructors ---

KripkeState::KripkeState(
    const KripkeState &other)
    : m_worlds(other.m_worlds),
      m_designated_worlds(other.m_designated_worlds),
      m_beliefs(other.m_beliefs),
      m_worlds_vec(other.m_worlds_vec),
      m_beliefs_vec(other.m_beliefs_vec),
      m_hash(other.m_hash) {}