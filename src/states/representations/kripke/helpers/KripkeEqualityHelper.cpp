#include "KripkeEqualityHelper.h"

#include <random>

#include "ArgumentParser.h"
#include "Define.h"
#include "Domain.h"
#include "FormulaHelper.h"

#include "KripkeState.h"
#include "KripkeWorld.h"

bool KripkeEqualityHelper::world_ptr_equal(const KripkeWorldPointer &a,
                                           const KripkeWorldPointer &b) {
  return a.internal_equal(b);
}

bool KripkeEqualityHelper::world_ptr_smaller(const KripkeWorldPointer &a,
                                             const KripkeWorldPointer &b) {
  if (a.internal_equal(b)) {
    return false;
  }
  return a.internal_smaller(b);
}

KripkeWorldPointersVec KripkeEqualityHelper::canonicalize_worlds(
    const KripkeWorldPointersSet &worlds) {
  KripkeWorldPointersVec result;
  result.reserve(worlds.size());

  for (const auto &w : worlds) {
    result.push_back(w);
  }

  std::ranges::sort(result,
                    [](const KripkeWorldPointer &a, const KripkeWorldPointer &b) {
                      const auto ida = a.get_internal_world_id();
                      const auto idb = b.get_internal_world_id();

                      if (ida != idb) {
                        return ida < idb;
                      }

                      if (a.internal_equal(b)) {
                        return false;
                      }

                      return a.internal_smaller(b);
                    });

  return result;
}

KripkeWorldPointersMapVec KripkeEqualityHelper::canonicalize_agent_map(
    const KripkeWorldPointersMap &beliefs) {
  KripkeWorldPointersMapVec result;
  result.reserve(beliefs.size());

  for (const auto &[agent, worlds] : beliefs) {
    result.emplace_back(agent, canonicalize_worlds(worlds));
  }

  return result;
}

KripkeWorldPointersTransitiveMapVec
KripkeEqualityHelper::canonicalize_transitive_map(
    const KripkeWorldPointersTransitiveMap &beliefs) {
  KripkeWorldPointersTransitiveMapVec result;
  result.reserve(beliefs.size());

  for (const auto &[world, agent_map] : beliefs) {
    result.emplace_back(world, canonicalize_agent_map(agent_map));
  }

  std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
    const auto ida = a.first.get_internal_world_id();
    const auto idb = b.first.get_internal_world_id();

    if (ida != idb) {
      return ida < idb;
    }

    if (a.first.internal_equal(b.first)) {
      return false;
    }

    return a.first.internal_smaller(b.first);
  });

  return result;
}

bool KripkeEqualityHelper::internal_equal(const KripkeWorldPointersVec &lhs,
                                          const KripkeWorldPointersVec &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (!world_ptr_equal(lhs[i], rhs[i])) {
      return false;
    }
  }

  return true;
}

bool KripkeEqualityHelper::internal_smaller(const KripkeWorldPointersVec &lhs,
                                            const KripkeWorldPointersVec &rhs) {
  return std::ranges::lexicographical_compare(
      lhs, rhs, [](const KripkeWorldPointer &a, const KripkeWorldPointer &b) {
        return world_ptr_smaller(a, b);
      });
}

bool KripkeEqualityHelper::internal_equal(
    const KripkeWorldPointersMapVec &lhs,
    const KripkeWorldPointersMapVec &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i].first != rhs[i].first) {
      return false;
    }

    if (!internal_equal(lhs[i].second, rhs[i].second)) {
      return false;
    }
  }

  return true;
}

bool KripkeEqualityHelper::internal_smaller(
    const KripkeWorldPointersMapVec &lhs,
    const KripkeWorldPointersMapVec &rhs) {
  return std::ranges::lexicographical_compare(
      lhs, rhs, [](const auto &a, const auto &b) {
        if (a.first != b.first) {
          return a.first < b.first;
        }

        if (internal_equal(a.second, b.second)) {
          return false;
        }

        return internal_smaller(a.second, b.second);
      });
}

bool KripkeEqualityHelper::internal_equal(
    const KripkeWorldPointersTransitiveMapVec &lhs,
    const KripkeWorldPointersTransitiveMapVec &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (!world_ptr_equal(lhs[i].first, rhs[i].first)) {
      return false;
    }

    if (!internal_equal(lhs[i].second, rhs[i].second)) {
      return false;
    }
  }

  return true;
}

bool KripkeEqualityHelper::internal_smaller(
    const KripkeWorldPointersTransitiveMapVec &lhs,
    const KripkeWorldPointersTransitiveMapVec &rhs) {
  return std::ranges::lexicographical_compare(
      lhs, rhs, [](const auto &a, const auto &b) {
        if (!world_ptr_equal(a.first, b.first)) {
          return world_ptr_smaller(a.first, b.first);
        }

        if (internal_equal(a.second, b.second)) {
          return false;
        }

        return internal_smaller(a.second, b.second);
      });
}

bool KripkeEqualityHelper::verify_exact_equality(
    const KripkeState &lhs,
    const KripkeState &rhs) {

  /*
   * IMPORTANT:
   *
   * This function deliberately ignores:
   *   - state hashes;
   *   - s_fast_comparison;
   *   - KripkeState::operator==.
   *
   * It independently verifies strong structural equality using the
   * repetition-independent canonical representation.
   */


  // ------------------------------------------------------------------------
  // Worlds
  // ------------------------------------------------------------------------

  if (lhs.get_worlds().size() !=
      rhs.get_worlds().size()) {

    return false;
  }

  const auto lhs_worlds =
      canonicalize_worlds(
          lhs.get_worlds());

  const auto rhs_worlds =
      canonicalize_worlds(
          rhs.get_worlds());

  if (!internal_equal(
          lhs_worlds,
          rhs_worlds)) {

    return false;
  }


  // ------------------------------------------------------------------------
  // Designated worlds
  // ------------------------------------------------------------------------

  if (lhs.get_designated_worlds().size() !=
      rhs.get_designated_worlds().size()) {

    return false;
  }

  const auto lhs_designated =
      canonicalize_worlds(
          lhs.get_designated_worlds());

  const auto rhs_designated =
      canonicalize_worlds(
          rhs.get_designated_worlds());

  if (!internal_equal(
          lhs_designated,
          rhs_designated)) {

    return false;
  }


  // ------------------------------------------------------------------------
  // Belief relation
  // ------------------------------------------------------------------------

  if (lhs.get_beliefs().size() !=
      rhs.get_beliefs().size()) {

    return false;
  }

  const auto lhs_beliefs =
      canonicalize_transitive_map(
          lhs.get_beliefs());

  const auto rhs_beliefs =
      canonicalize_transitive_map(
          rhs.get_beliefs());

  if (!internal_equal(
          lhs_beliefs,
          rhs_beliefs)) {

    return false;
  }

  return true;
}


bool KripkeEqualityHelper::less_operator(
    const KripkeState &reference,
    const KripkeState &to_compare) {

  const auto reference_hash =
      reference.get_hash();

  const auto to_compare_hash =
      to_compare.get_hash();

  /*
   * Different hashes immediately establish ordering.
   *
   * No canonical structures need to be constructed.
   */
  if (reference_hash != to_compare_hash) {
    return reference_hash < to_compare_hash;
  }

  /*
   * Fast mode deliberately accepts equal state hashes as
   * state equality.
   *
   * No structural comparison and therefore no temporary
   * canonical structures are required.
   */
  if (s_fast_comparison) {
    return false;
  }

  /*
   * Equal hashes in exact mode.
   *
   * Canonical structures are generated lazily and exist only
   * for the duration of this comparison.
   */


  // ------------------------------------------------------------------------
  // Designated worlds
  // ------------------------------------------------------------------------

  const auto reference_designated =
      canonicalize_worlds(
          reference.get_designated_worlds());

  const auto to_compare_designated =
      canonicalize_worlds(
          to_compare.get_designated_worlds());

  if (!internal_equal(
          reference_designated,
          to_compare_designated)) {

    return internal_smaller(
        reference_designated,
        to_compare_designated);
  }


  // ------------------------------------------------------------------------
  // Worlds
  // ------------------------------------------------------------------------

  const auto reference_worlds =
      canonicalize_worlds(
          reference.get_worlds());

  const auto to_compare_worlds =
      canonicalize_worlds(
          to_compare.get_worlds());

  if (!internal_equal(
          reference_worlds,
          to_compare_worlds)) {

    return internal_smaller(
        reference_worlds,
        to_compare_worlds);
  }


  // ------------------------------------------------------------------------
  // Belief relation
  // ------------------------------------------------------------------------

  const auto reference_beliefs =
      canonicalize_transitive_map(
          reference.get_beliefs());

  const auto to_compare_beliefs =
      canonicalize_transitive_map(
          to_compare.get_beliefs());

  return internal_smaller(
      reference_beliefs,
      to_compare_beliefs);
}


void KripkeEqualityHelper::set_fast_comparison(
    const bool value) noexcept {
  s_fast_comparison = value;
}


namespace {

BeliefFormula make_random_formula(
    std::mt19937_64 &rng,
    const std::vector<Fluent> &fluents,
    const AgentsList &agents,
    const unsigned int depth) {

  /*
   * Leaf:
   *
   * Generate a grounded fluent formula.
   */
  if (depth == 0 || agents.empty()) {

    std::uniform_int_distribution<std::size_t>
        fluent_dist(
            0,
            fluents.size() - 1);

    BeliefFormula result;

    result.set_formula_type(
        BeliefFormulaType::FLUENT_FORMULA);

    result.set_fluent_formula_from_fluent(
        fluents[
            fluent_dist(rng)]);

    return result;
  }


  /*
   * Formula kinds:
   *
   *   0  fluent
   *   1  NOT
   *   2  AND
   *   3  OR
   *   4  B_a
   *   5  E_G
   *   6  C_G
   */
  std::uniform_int_distribution<int>
      kind_dist(0, 6);

  const int kind =
      kind_dist(rng);


  // ----------------------------------------------------------------------
  // Fluent
  // ----------------------------------------------------------------------

  if (kind == 0) {

    std::uniform_int_distribution<std::size_t>
        fluent_dist(
            0,
            fluents.size() - 1);

    BeliefFormula result;

    result.set_formula_type(
        BeliefFormulaType::FLUENT_FORMULA);

    result.set_fluent_formula_from_fluent(
        fluents[
            fluent_dist(rng)]);

    return result;
  }


  // ----------------------------------------------------------------------
  // NOT
  // ----------------------------------------------------------------------

  if (kind == 1) {

    BeliefFormula child =
        make_random_formula(
            rng,
            fluents,
            agents,
            depth - 1);

    BeliefFormula result;

    result.set_formula_type(
        BeliefFormulaType::PROPOSITIONAL_FORMULA);

    result.set_operator(
        BeliefFormulaOperator::BF_NOT);

    result.set_bf1(
        child);

    return result;
  }


  // ----------------------------------------------------------------------
  // AND / OR
  // ----------------------------------------------------------------------

  if (kind == 2 ||
      kind == 3) {

    BeliefFormula lhs =
        make_random_formula(
            rng,
            fluents,
            agents,
            depth - 1);

    BeliefFormula rhs =
        make_random_formula(
            rng,
            fluents,
            agents,
            depth - 1);

    BeliefFormula result;

    result.set_formula_type(
        BeliefFormulaType::PROPOSITIONAL_FORMULA);

    result.set_operator(
        kind == 2
            ? BeliefFormulaOperator::BF_AND
            : BeliefFormulaOperator::BF_OR);

    result.set_bf1(lhs);
    result.set_bf2(rhs);

    return result;
  }


  // ----------------------------------------------------------------------
  // Individual belief B_a(phi)
  // ----------------------------------------------------------------------

  if (kind == 4) {

    std::uniform_int_distribution<std::size_t>
        agent_dist(
            0,
            agents.size() - 1);

    BeliefFormula child =
        make_random_formula(
            rng,
            fluents,
            agents,
            depth - 1);

    BeliefFormula result;

    result.set_formula_type(
        BeliefFormulaType::BELIEF_FORMULA);

    result.set_agent(
        agents[
            agent_dist(rng)]);

    result.set_bf1(
        child);

    return result;
  }


  // ----------------------------------------------------------------------
  // Everyone / Common belief
  // ----------------------------------------------------------------------

  BeliefFormula child =
      make_random_formula(
          rng,
          fluents,
          agents,
          depth - 1);

  /*
   * Generate a non-empty random agent group.
   */
  AgentsSet group;

  std::bernoulli_distribution
      include_agent(0.5);

  for (const Agent &agent : agents) {

    if (include_agent(rng)) {
      group.insert(agent);
    }
  }

  /*
   * set_group_agents() rejects an empty group.
   */
  if (group.empty()) {

    std::uniform_int_distribution<std::size_t>
        agent_dist(
            0,
            agents.size() - 1);

    group.insert(
        agents[
            agent_dist(rng)]);
  }

  BeliefFormula result;

  result.set_formula_type(
      kind == 5
          ? BeliefFormulaType::E_FORMULA
          : BeliefFormulaType::C_FORMULA);

  result.set_group_agents(
      group);

  result.set_bf1(
      child);

  return result;
}

} // namespace

bool KripkeEqualityHelper::verify_equivalence(
    const KripkeState &lhs,
    const KripkeState &rhs,
    const bool require_structural_equality,
    const std::size_t formula_count,
    const unsigned int max_depth) {

    /*
     * Strong structural equality is required for visited-state
     * pruning, but deliberately NOT for bisimulation.
     */
    if (require_structural_equality &&
        !verify_exact_equality(lhs, rhs)) {

#ifdef DEBUG
        if (ArgumentParser::get_instance().get_verbose()) {
            ArgumentParser::get_instance()
                .get_output_stream()
                << "[DEBUG] Structural equivalence FAILED."
                << std::endl;
        }
#endif

        return false;
        }

    /*
     * Semantic equivalence is checked independently by evaluating
     * deterministic random epistemic formulae in both states.
     */
    if (formula_count > 0) {

        FormulaHelper::verify_semantic_equivalence(
            lhs,
            rhs,
            max_depth,
            static_cast<unsigned int>(formula_count));
    }

#ifdef DEBUG
    if (ArgumentParser::get_instance().get_verbose()) {

        auto &os =
            ArgumentParser::get_instance()
                .get_output_stream();

        os << "[DEBUG] State equivalence verified";

        if (require_structural_equality) {
            os << " structurally";
        }

        if (formula_count > 0) {
            os << " and with "
               << formula_count
               << " epistemic formulae";
        }

        os << "."
           << std::endl;
    }
#endif

    return true;
}