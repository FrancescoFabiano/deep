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
#include "KripkeEntailmentHelper.h"
#include "KripkeReachabilityHelper.h"
#include "KripkeState.h"

#include <ranges>
#include <unordered_set>
#include <utility>

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
}

void KripkeState::set_designated_worlds(const KripkeWorldPointersSet &to_set) {
  m_designated_worlds = to_set;
}

void KripkeState::add_designated_world(const KripkeWorldPointer &to_add) {
  m_designated_worlds.insert(to_add);
}

void KripkeState::set_beliefs(const KripkeWorldPointersTransitiveMap &to_set) {
  m_beliefs = to_set;
}

void KripkeState::clear_beliefs() { m_beliefs.clear(); }

// --- Getters ---

[[nodiscard]] const KripkeWorldPointersSet &
KripkeState::get_worlds() const noexcept {
  return m_worlds;
}

[[nodiscard]] const KripkeWorldPointersSet &
KripkeState::get_designated_worlds() const noexcept {
  return m_designated_worlds;
}

[[nodiscard]] bool
KripkeState::is_designated(const KripkeWorldPointer &world) const noexcept {
  return m_designated_worlds.contains(world);
}

[[nodiscard]] const KripkeWorldPointersTransitiveMap &
KripkeState::get_beliefs() const noexcept {
  return m_beliefs;
}
// --- Operators ---

KripkeState &KripkeState::operator=(const KripkeState &to_copy) {
  if (this != &to_copy) {
    m_worlds = to_copy.m_worlds;
    m_designated_worlds = to_copy.m_designated_worlds;
    m_beliefs = to_copy.m_beliefs;

    m_hash = to_copy.m_hash;
  }

  return *this;
}

bool KripkeState::operator==(const KripkeState &to_compare) const {
  return !(*this < to_compare) && !(to_compare < *this);
}

bool KripkeState::operator<(const KripkeState &to_compare) const {
  return KripkeEqualityHelper::less_operator(*this, to_compare);
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

void KripkeState::add_world(KripkeWorld &&to_add) {

  m_worlds.insert(KripkeStorage::get_instance().add_world(std::move(to_add)));
}

KripkeWorldPointer KripkeState::add_rep_world(const KripkeWorld &to_add,
                                              const unsigned short repetition) {

  KripkeWorldPointer tmp = KripkeStorage::get_instance().add_world(to_add);

  tmp.set_repetition(repetition);
  m_worlds.insert(tmp);

  return tmp;
}

KripkeWorldPointer KripkeState::add_rep_world(KripkeWorld &&to_add,
                                              const unsigned short repetition) {

  KripkeWorldPointer tmp =
      KripkeStorage::get_instance().add_world(std::move(to_add));

  tmp.set_repetition(repetition);

  m_worlds.insert(tmp);

  return tmp;
}

void KripkeState::add_edge(const KripkeWorldPointer &from,
                           const KripkeWorldPointer &to, const Agent &ag) {

  m_beliefs[from][ag].insert(to);
}

void KripkeState::build_initial() {
  const auto &domain = Domain::get_instance();
  const auto &plank_state = domain.get_initial_state();

  if (!plank_state) {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::DomainBuildError,
                                   "Cannot build initial Kripke state: "
                                   "plank initial state is null.");
  }

  const auto &positive_fluents = domain.get_positive_fluents();

  const auto &agents = domain.get_agents();

  const auto worlds_number = plank_state->get_worlds_number();

  // Temporary mapping only:
  // plank world id -> canonical DEEP world pointer.
  std::vector<KripkeWorldPointer> world_map;
  world_map.reserve(worlds_number);

  // ---------------------------------------------------------
  // Worlds, labels and designated worlds.
  // ---------------------------------------------------------

  for (plank::del::world_id world_id = 0; world_id < worlds_number;
       ++world_id) {

    const auto &label = plank_state->get_label(world_id);

    FluentsSet description;

    for (std::size_t atom_id = 0; atom_id < positive_fluents.size();
         ++atom_id) {

      Fluent fluent = positive_fluents[atom_id];

      if (!label[static_cast<plank::del::atom>(atom_id)]) {

        // Last bit distinguishes positive/negative literals
        // in DEEP.
        fluent.set(fluent.size() - 1, false);
      }

      description.insert(std::move(fluent));
    }

    KripkeWorld world(std::move(description));

    const auto world_ptr =
        KripkeStorage::get_instance().add_world(std::move(world));

    m_worlds.insert(world_ptr);

    world_map.push_back(world_ptr);

    if (plank_state->is_designated(world_id)) {
      m_designated_worlds.insert(world_ptr);
    }
  }

  // ---------------------------------------------------------
  // Accessibility relations.
  // ---------------------------------------------------------

  std::size_t agent_id = 0;

  for (const auto &agent : agents) {
    for (plank::del::world_id from_id = 0; from_id < worlds_number; ++from_id) {

      const auto &possible_worlds = plank_state->get_agent_possible_worlds(
          static_cast<plank::del::agent>(agent_id), from_id);

      for (const plank::del::world_id to_id : possible_worlds) {
        add_edge(world_map[from_id], world_map[to_id], agent);
      }
    }

    ++agent_id;
  }

  if (m_designated_worlds.empty()) {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::DomainBuildError,
                                   "Cannot build initial Kripke state: "
                                   "plank produced no designated worlds.");
  }

  recompute_hash();

#ifdef DEBUG
  if (ArgumentParser::get_instance().get_verbose()) {
    auto &os = ArgumentParser::get_instance().get_output_stream();

    os << "\n[EPDDL] Initial Kripke state imported:" << "\n  Worlds: "
       << m_worlds.size()
       << "\n  Designated worlds: " << m_designated_worlds.size()
       << "\n  Agents: " << Domain::get_instance().get_agents().size()
       << std::endl;

    std::size_t edges = 0;

    for (const auto &[source, agent_map] : m_beliefs) {
      for (const auto &[agent, targets] : agent_map) {
        edges += targets.size();
      }
    }

    os << "  Accessibility edges: " << edges << std::endl;
  }
#endif
}

bool KripkeState::is_executable(const Action &action) const {

  const auto &designated_events = action.get_designated_events();

  if (designated_events.empty()) {
    return false;
  }

  for (const auto &world : m_designated_worlds) {

    bool executable_in_world = false;

    for (const EventId event_id : designated_events) {

      const Event &event = action.get_event(event_id);

      if (is_event_applicable(event, world)) {

        executable_in_world = true;
        break;
      }
    }

    if (!executable_in_world) {
      return false;
    }
  }

  return true;
}

void KripkeState::recompute_hash() {
  m_hash = FormulaHelper::hash_kripke_state(*this);
}

uint64_t KripkeState::get_hash() const noexcept { return m_hash; }

// --- Transition ---

bool KripkeState::is_event_applicable(const Event &event,
                                      const KripkeWorldPointer &world) const {

  return KripkeEntailmentHelper::entails(event.get_precondition(), world,
                                         *this);
}

bool KripkeState::is_event_applicable_cached(const Event &event,
                                             const KripkeWorldPointer &world,
                                             ApplicabilityCache &cache) const {

  const ProductWorld key{world, event.get_id()};

  const auto it = cache.find(key);

  if (it != cache.end()) {
    return it->second;
  }

  const bool applicable = is_event_applicable(event, world);

  cache.emplace(key, applicable);

  return applicable;
}

FluentsSet
KripkeState::apply_event_postconditions(const Event &event,
                                        const KripkeWorldPointer &world) const {

  FluentsSet description = world.get_fluent_set();

  /*
   * DEL postconditions:
   *
   *     p -> phi
   *
   * phi is evaluated in the source epistemic model/world.
   */
  for (const auto &[fluent, postcondition] : event.get_postconditions()) {

    const bool value =
        KripkeEntailmentHelper::entails(postcondition, world, *this);

    /*
     * Normalize the postcondition key to its positive fluent.
     */
    const Fluent positive_fluent = FormulaHelper::is_negated(fluent)
                                       ? FormulaHelper::negate_fluent(fluent)
                                       : fluent;

    /*
     * Remove both possible truth assignments before inserting
     * the new value.
     */
    description.erase(positive_fluent);

    description.erase(FormulaHelper::negate_fluent(positive_fluent));

    description.insert(value ? positive_fluent
                             : FormulaHelper::negate_fluent(positive_fluent));
  }

  return description;
}

KripkeState::ResolvedObservability
KripkeState::resolve_observability_types(const Action &action) const {

  ResolvedObservability result;

  /*
   * Action already contains:
   *
   *   agent -> (observability type -> condition)
   *
   * Therefore there is no need to ask Domain for an agent list.
   */
  for (const auto &[agent, conditions] :
       action.get_observability_conditions()) {

    bool found = false;
    ObservabilityType selected_type{};

    /*
     * Match plank's state-level observability resolution.
     *
     * Conditions are evaluated against the complete source
     * epistemic state, once before product construction.
     */
    for (const auto &[obs_type, condition] : conditions) {

      if (!KripkeEntailmentHelper::entails(condition, *this)) {
        continue;
      }

      /*
       * Preserve plank's behaviour if multiple conditions
       * happen to hold: the last satisfied type is retained.
       */
      selected_type = obs_type;
      found = true;
    }

    if (!found) {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::DomainBuildError,
          "No observability condition holds for an agent "
          "while executing action '" +
              action.get_name() + "'.");
    }

    result.emplace(agent, selected_type);
  }

  return result;
}

KripkeWorldPointer KripkeState::get_or_create_product_world(
    const KripkeWorldPointer &source_world, const Event &event,
    KripkeState &successor, ProductWorldMap &product_worlds,
    ProductWorldQueue &pending, unsigned short &next_repetition) const {

  const ProductWorld product{source_world, event.get_id()};

  const auto existing = product_worlds.find(product);

  if (existing != product_worlds.end()) {
    return existing->second;
  }

  FluentsSet description = apply_event_postconditions(event, source_world);

  const KripkeWorldPointer product_world = successor.add_rep_world(
      KripkeWorld(std::move(description)), next_repetition++);

  product_worlds.emplace(product, product_world);

  pending.push(product);

  return product_world;
}

void KripkeState::create_designated_product_worlds(
    const Action &action, KripkeState &successor,
    ProductWorldMap &product_worlds, ProductWorldQueue &pending,
    ApplicabilityCache &applicability_cache,
    unsigned short &next_repetition) const {

  for (const auto &source_world : m_designated_worlds) {

    for (const EventId event_id : action.get_designated_events()) {

      const Event &event = action.get_event(event_id);

      if (!is_event_applicable_cached(event, source_world,
                                      applicability_cache)) {
        continue;
      }

      const KripkeWorldPointer product_world =
          get_or_create_product_world(source_world, event, successor,
                                      product_worlds, pending, next_repetition);

      successor.m_designated_worlds.insert(product_world);
    }
  }

  if (successor.get_designated_worlds().empty()) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::StateActionNotExecutableError,
        "Action '" + action.get_name() +
            "' produced no designated successor worlds.");
  }
}

void KripkeState::expand_product_relations(
    const Action &action, const ResolvedObservability &observability,
    KripkeState &successor, ProductWorldMap &product_worlds,
    ProductWorldQueue &pending, ApplicabilityCache &applicability_cache,
    unsigned short &next_repetition) const {

  while (!pending.empty()) {

    const ProductWorld current = pending.front();

    pending.pop();

    const KripkeWorldPointer &source_world = current.first;

    const Event &source_event = action.get_event(current.second);

    const KripkeWorldPointer &product_source = product_worlds.at(current);

    /*
     * Find the outgoing epistemic relations of the original
     * source world.
     */
    const auto source_beliefs = m_beliefs.find(source_world);

    if (source_beliefs == m_beliefs.end()) {
      continue;
    }

    /*
     * For every agent having outgoing accessibility edges
     * from source_world.
     */
    for (const auto &[agent, world_targets] : source_beliefs->second) {

      /*
       * Observability was resolved once for the complete
       * source epistemic state.
       */
      const auto obs_it = observability.find(agent);

      if (obs_it == observability.end()) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::DomainBuildError,
            "Missing resolved observability type for an agent "
            "while executing action '" +
                action.get_name() + "'.");
      }

      /*
       * Select the event relation associated with the
       * observability type resolved for this agent.
       *
       * EventRelation:
       *
       *   source event -> target events
       */
      const EventRelation &event_relation =
          action.get_observability_relation(obs_it->second);

      /*
       * We only need the event successors of the current
       * source event.
       */
      const auto event_targets_it = event_relation.find(source_event.get_id());

      if (event_targets_it == event_relation.end()) {
        continue;
      }

      const EventTargets &target_events = event_targets_it->second;

      /*
       * Product relation:
       *
       *   (w,e) R'_a (v,f)
       *
       * iff:
       *
       *   w R_a v
       *   e R^A_a f
       *   M,v |= pre(f)
       */
      for (const auto &target_world : world_targets) {

        for (const EventId target_event_id : target_events) {

          const Event &target_event = action.get_event(target_event_id);

          /*
           * Product world (v,f) exists only if the
           * target event is applicable in v.
           */
          if (!is_event_applicable_cached(target_event, target_world,
                                          applicability_cache)) {
            continue;
          }

          const KripkeWorldPointer product_target = get_or_create_product_world(
              target_world, target_event, successor, product_worlds, pending,
              next_repetition);

          successor.add_edge(product_source, product_target, agent);
        }
      }
    }
  }
}

KripkeState KripkeState::compute_successor(const Action &action) const {

#ifdef DEBUG
  if (!is_executable(action)) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::StateActionNotExecutableError,
        "Action '" + action.get_name() +
            "' is not executable in the current Kripke state.");
  }
#endif

  KripkeState successor;

  ProductWorldMap product_worlds;
  ProductWorldQueue pending;
  ApplicabilityCache applicability_cache;

  unsigned short next_repetition = 0;

  /*
   * Plank resolves one observability type per agent against
   * the complete source epistemic state.
   */
  const ResolvedObservability observability =
      resolve_observability_types(action);

  /*
   * Seed the reachable product model with designated
   * (world,event) pairs.
   */
  create_designated_product_worlds(action, successor, product_worlds, pending,
                                   applicability_cache, next_repetition);

  /*
   * Expand all reachable product worlds and accessibility
   * relations.
   */
  expand_product_relations(action, observability, successor, product_worlds,
                           pending, applicability_cache, next_repetition);

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

#if defined(DEBUG) || defined(DEEP_VERIFY)
  const KripkeState before_bisimulation = *this;
#endif

  KripkeReachabilityHelper::clean_unreachable_worlds(*this);

  Bisimulation b;
  b.calc_min_bisimilar(*this);

  // Keep this here: the Kripke structure may have changed.
  recompute_hash();

#if defined(DEBUG) || defined(DEEP_VERIFY)

  /*
   * Bisimulation is allowed to change the structure, so do NOT
   * require structural equality. Check semantic equivalence with
   * 500 deterministic random epistemic formulae.
   */
  if (!KripkeEqualityHelper::verify_equivalence(before_bisimulation, *this,
                                                false, 500, 5)) {

    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::SearchMethodError,
        "DEBUG: bisimulation equivalence verification failed.");
  }

#endif
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

KripkeState::KripkeState(const KripkeState &other)
    : m_worlds(other.m_worlds), m_designated_worlds(other.m_designated_worlds),
      m_beliefs(other.m_beliefs), m_hash(other.m_hash) {}