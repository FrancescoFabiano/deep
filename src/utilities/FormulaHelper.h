/**
 * \brief Utility class for manipulating grounded fluent and belief formulas.
 *
 *  The class implements static methods to facilitate
 *  the modification of the formulae and other.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 16, 2025
 */
#pragma once

#include <random>

#include "Define.h"
#include "State.h"
#include "formulae/BeliefFormula.h"

class FormulaHelper {
public:
  /** \brief Function that checks if two FluentsSet are consistent.
   *
   * Two FluentsSet are consistent if they not contains a Fluent and
   * its negation together.*/
  static bool is_consistent(const FluentsSet &, const FluentsSet &);

  /** \brief Function that returns the negation of a given Fluent.
   *
   * @param[in] to_negate: the Fluent to negate
   *
   * @return the negation of \p to_negate.*/
  static Fluent negate_fluent(const Fluent &to_negate);

  /** \brief Function that returns the negation of a given FluentFormula.
   *
   * @param[in] to_negate: the FluentFormula to negate
   *
   * @return the negation of \p to_negate.*/
  static FluentFormula negate_fluent_formula(const FluentFormula &to_negate);

  /** \brief Function that returns the positive version of a given Fluent.
   *
   * @param[in] to_normalize: the Fluent to normalize
   *
   * @return the normalized of fluent.*/
  static Fluent normalize_fluent(const Fluent &to_normalize);

  static bool is_negated(const Fluent &f);

  /** \brief Function to set the truth value of a fluent in a world description.
   *
   * @param[in] effect: the fluent to set.
   * @param[out] world_description: the FluentsSet contained inside a
   * single world to modify.*/
  static void apply_effect(const Fluent &effect, FluentsSet &world_description);

  /** \brief Merge a conjunctive postcondition into a world valuation.
   *
   * @param[in] effect: part of an event postcondition written as a conjunctive
   * fluent set.
   * @param[out] world_description: the FluentsSet contained inside a
   * single world.
   *
   * @return the description of the world after \p effect has been applied to \p
   * world_description.*/
  static void apply_effect(const FluentsSet &effect,
                           FluentsSet &world_description);

  /** \brief Function that merges two conjunctive set of Fluent into one.
   *
   * @param[in]  to_merge_1: the first conjunctive set of Fluent to merge.
   * @param[in]  to_merge_2: the second conjunctive set of Fluent to merge.
   *
   * @return the union of all the Fluent in \p to_merge_1\2 if is
   * consistent (exit otherwise).*/
  static FluentsSet and_ff(const FluentsSet &to_merge_1,
                           const FluentsSet &to_merge_2);

  /** \brief Function that merges two FluentFormula into one.
   *
   * @param[in] to_merge_1: the first FluentFormula to merge.
   * @param[in] to_merge_2: the second FluentFormula to merge.
   *
   * @return the union of all the FluentsSet in \p to_merge_1\2 if is
   * consistent (exit otherwise).*/
  static FluentFormula and_ff(const FluentFormula &to_merge_1,
                              const FluentFormula &to_merge_2);

  /** \brief Function that checks if two BeliefFormula are of the form
   * B(i, *phi*) -- B(i, -*phi*) where *phi* is a FluentFormula.
   *
   * This function is useful to identify when an agent knows the truth value of
   * *phi* by checking both the positive and negative knowledge forms.
   *
   * @param[in] to_check_1: the first BeliefFormula to check.
   * @param[in] to_check_2: the second BeliefFormula to check.
   * @param[out] ret: the FluentFormula containing *phi*.
   *
   * @return true: if the two BeliefFormula contain FluentFormula
   * that are the negation of each other.
   * @return false: otherwise.*/
  static bool check_Bff_notBff(const BeliefFormula &to_check_1,
                               const BeliefFormula &to_check_2,
                               FluentFormula &ret);

  /** \brief Apply a fluent formula as an event postcondition to a world.
   *
   * The formula must denote a deterministic conjunctive assignment. If it
   * contains disjunction, the helper reports an error.
   *
   * Then it calls apply_effect(const FluentsSet &, FluentsSet &).
   *
   * @param[in] effect: the grounded postcondition formula to apply.
   * @param[out] world_description: the description of the world after \p effect
   * has been applied to \p world_description.
   *
   * @return the description of the world after \p effect has been applied to \p
   * world_description.*/
  static void apply_effect(const FluentFormula &effect,
                           FluentsSet &world_description);
  /**
   * \brief Computes 2 raised to the power of the given length.
   * \param[in] length The exponent value.
   * \return 2 to the power of \p length.
   */
  static int length_to_power_two(int length);

  /**
   * \brief Checks if two sets of fluents have an empty intersection.
   * \param[in] set1 The first set of fluents.
   * \param[in] set2 The second set of fluents.
   * \return true if the intersection is empty, false otherwise.
   */
  static bool fluentset_empty_intersection(const FluentsSet &set1,
                                           const FluentsSet &set2);

  /**
   * \brief Checks if the intersection of set1 and the negation of set2 is
   * empty. \param[in] set1 The first set of fluents. \param[in] set2 The second
   * set of fluents. \return true if there is no fluent in set1 that is the
   * negation of a fluent in set2.
   */
  static bool fluentset_negated_empty_intersection(const FluentsSet &set1,
                                                   const FluentsSet &set2);
  /** \brief Return the agents whose observability condition is entailed.
   *
   * @param[in] map: the map that contains the tuples to check for entailment.
   * @param[in] state: the state in which to check the entailment.
   * @return The set of grounded agents whose condition holds in \p state.
   */
  static AgentsSet get_agents_if_entailed(const ObservabilitiesMap &map,
                                          const KripkeState &state);

  /** \brief Return the first grounded effect whose enabling condition is entailed.
   *
   * @param[in] map: the map that contains the tuples to check for entailment.
   * @param[in] state: the state in which to check the entailment.
   * @return The grounded fluent formula selected in \p state.
   */
  static FluentFormula get_effects_if_entailed(const EffectsMap &map,
                                               const KripkeState &state);

  /**
   * \brief Concatenate two dynamic_bitsets as strings.
   * \param[in] bs1 The first bitset.
   * \param[in] bs2 The second bitset.
   * \return The concatenated bitset.
   */
  static boost::dynamic_bitset<>
  concatStringDyn(const boost::dynamic_bitset<> &bs1,
                  const boost::dynamic_bitset<> &bs2);

  /**
   * \brief Concatenate two dynamic_bitsets using bitwise operators.
   * \param[in] bs1 The first bitset.
   * \param[in] bs2 The second bitset.
   * \return The concatenated bitset.
   */
  static boost::dynamic_bitset<>
  concatOperatorsDyn(const boost::dynamic_bitset<> &bs1,
                     const boost::dynamic_bitset<> &bs2);

  /**
   * \brief Concatenate two dynamic_bitsets using a loop.
   * \param[in] bs1 The first bitset.
   * \param[in] bs2 The second bitset.
   * \return The concatenated bitset.
   */
  static boost::dynamic_bitset<>
  concatLoopDyn(const boost::dynamic_bitset<> &bs1,
                const boost::dynamic_bitset<> &bs2);

  /**
   * \brief Hash a set of fluents into a unique id.
   * \param[in] fl The set of fluents.
   * \return The unique id.
   */
  static KripkeWorldId hash_fluents_into_id(const FluentsSet &fl);

  /**
   * \brief Hash a string into a unique id.
   * \param[in] string The string (that represents the ideal overflowed number
   * of KripkeWorldPointer) to hash. \return The unique id.
   */
  static KripkeWorldId hash_string_into_id(const std::string &string);

 static uint64_t hash_kripke_state(const KripkeState &state);

  /**
   * \brief Check if a set of fluents is consistent.
   * \param[in] to_check The set to check.
   * \details Exits with an error if the set is not consistent.
   * \return true if consistent (never returns false, will exit on failure).
   *
   *  \todo Integrate static_law for consistency checking.
   */
  static bool consistent(const FluentsSet &to_check);

  /**
   * \brief Check if two Kripke states are bisimilar (entail the same set of
   * formulae). Fails if they are not. \param[in] first The first state to
   * check. \param[in] second The second state to check. \param modal_depth \param formula_count \param seed
   */
 static void verify_semantic_equivalence(
     const KripkeState &first,
     const KripkeState &second,
     unsigned int modal_depth = 3,
     unsigned int formula_count = 100,
     std::uint32_t seed = 3760);


 static BeliefFormula make_random_atom(
    const std::vector<Fluent> &fluents,
    std::mt19937 &rng);

 static BeliefFormula make_random_propositional_formula(
     const std::vector<Fluent> &fluents,
     std::mt19937 &rng);

 static BeliefFormula make_random_modal_formula(
     BeliefFormula inner,
     const std::vector<Agent> &agents,
     unsigned int depth,
     std::mt19937 &rng);

 static BeliefFormula make_random_formula(
     const std::vector<Fluent> &fluents,
     const std::vector<Agent> &agents,
     unsigned int modal_depth,
     std::mt19937 &rng);

};
