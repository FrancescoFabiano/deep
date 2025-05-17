/**
 * \brief Class used to check properties of formulae and modify them.
 * 
 *  The class implements static methods to facilitate
 *  the modification of the formulae and other.
 *
 * \see fluent_formula, belief_formula.
 * 
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 16, 2025
 */
#pragma once

#include "Define.h"

class FormulaHelper
{
public:

    /** \brief Function that checks if two \ref fluent_set are consistent.
     * 
     * Two \ref fluent_set are consistent if they not contains a \ref fluent and its negation together.*/
    static bool is_consistent(const FluentsSet &, const FluentsSet &);

    /** \brief Function that returns the negation of a given \ref fluent.
     * 
     * @param[in] to_negate: the \ref fluent to negate
     * 
     * @return the negation of \p to_negate.*/
    static Fluent negate_fluent(const Fluent & to_negate);

    /** \brief Function that returns the negation of a given \ref fluent_formula.
     * 
     * @param[in] to_negate: the \ref fluent_formula to negate
     * 
     * @return the negation of \p to_negate.*/
    static FluentFormula negate_fluent_formula(const FluentFormula & to_negate);

    /** \brief Function that returns the positive version of a given \ref fluent.
     * 
     * @param[in] to_normalize: the \ref fluent to normalize
     * 
     * @return the normalized of fluent.*/
    static Fluent normalize_fluent(const Fluent & to_normalize);

    static bool is_negated(const Fluent & f);

    /** \brief Function to set the truth value of a fluent in a world description.
     *   
     * @param[in] effect: the fluent to set.
     * @param[out] world_description: the \ref fluent_set contained inside a single world to modify.*/
    static void apply_effect(const Fluent& effect, FluentsSet& world_description);

    /** \brief Function to merge the results of an \ref ONTIC \ref action with a world description.
     *   
     * @param[in] effect: part of the effect of an \ref ONTIC \ref action in CNF form.
     * @param[out] world_description: the \ref fluent_set contained inside a single world.
     * 
     * @return the description of the world after \p effect has been applied to \p world_description.*/
    static void apply_effect(const FluentsSet& effect, FluentsSet& world_description);

    /** \brief Function that merges two conjunctive set of \ref fluent into one.
     *   
     * @param[in]  to_merge_1: the first conjunctive set of \ref fluent to merge.
     * @param[in]  to_merge_2: the second conjunctive set of \ref fluent to merge.
     * 
     * @return the union of all the \ref fluent in \p to_merge_1\2 if is consistent (exit otherwise).*/
    static FluentsSet and_ff(const FluentsSet& to_merge_1, const FluentsSet& to_merge_2);

    /** \brief Function that merges two \ref fluent_formula into one.
     * 
     * @param[in] to_merge_1: the first \ref fluent_formula to merge.
     * @param[in] to_merge_2: the second \ref fluent_formula to merge.
     * 
     * @return the union of all the \ref fluent_set in \p to_merge_1\2 if is consistent (exit otherwise).*/
    static FluentFormula and_ff(const FluentFormula& to_merge_1, const FluentFormula& to_merge_2);

    /** \brief Function that checks if two \ref belief_formula are of the form B(i, *phi*) -- B(i, -*phi*) where *phi* is a \ref fluent_formula.
     * 
     * This function is useful to identify when an agent \p knows the true value of *phi*.
     * Is one of the accepted formulae in \ref S5.
     * 
     * @param[in] to_check_1: the first \ref belief_formula to check.
     * @param[in] to_check_2: the second \ref belief_formula to check.
     * @param[out] ret: the \ref fluent_formula containing *phi*.
     * 
     * @return true: if the two \ref belief_formula contain \ref fluent_formula that are the negation of each other.
     * @return false: otherwise.*/
    static bool check_Bff_notBff(const BeliefFormula& to_check_1, const BeliefFormula& to_check_2, const std::shared_ptr<FluentFormula>& ret);

    /** \brief Function that check that the \ref ONTIC effect doesn't have uncertainty (OR).
     * 
     * Then it calls apply_effect(const fluent_set&, const fluent_set&);
     *   
     * @param[in] effect: the effect of an \ref ONTIC \ref action.
     * @param[out] world_description: the description of the world after \p effect has been applied to \p world_description.
     * 
     * @return the description of the world after \p effect has been applied to \p world_description.*/
    static void apply_effect(const FluentFormula& effect, FluentsSet& world_description);

    static int length_to_power_two(int length);

    static bool fluentset_empty_intersection(const FluentsSet & set1, const FluentsSet & set2);
    static bool fluentset_negated_empty_intersection(const FluentsSet & set1, const FluentsSet & set2);

    /** \brief Function that return the set of \ref agent that entails the obs condition.
     *
     * @param[in] map: the map that contains the tuples to check for entailment.
     * @param[in] state: the state in which to check the entailment.
     * @return the effects that are feasible in *this* with \p start as pointed world*.*/
    static AgentsSet get_agents_if_entailed(const observability_map & map, const KripkeState & state);

    /** \brief Function that return the \ref fluent_formula (effect) that entails the exe condition.
     *
     * @param[in] map: the map that contains the tuples to check for entailment.
     * @param[in] state: the state in which to check the entailment.
     * @return the effects that are feasible in \p state.*/
    static FluentFormula get_effects_if_entailed(const effects_map & map, const KripkeState & state);

    /**
     * \brief Concatenate two dynamic_bitsets as strings.
     * \param[in] bs1 The first bitset.
     * \param[in] bs2 The second bitset.
     * \return The concatenated bitset.
     */
    static boost::dynamic_bitset<> concatStringDyn(const boost::dynamic_bitset<>& bs1, const boost::dynamic_bitset<>& bs2);

    /**
     * \brief Concatenate two dynamic_bitsets using bitwise operators.
     * \param[in] bs1 The first bitset.
     * \param[in] bs2 The second bitset.
     * \return The concatenated bitset.
     */
    static boost::dynamic_bitset<> concatOperatorsDyn(const boost::dynamic_bitset<>& bs1, const boost::dynamic_bitset<>& bs2);

    /**
     * \brief Concatenate two dynamic_bitsets using a loop.
     * \param[in] bs1 The first bitset.
     * \param[in] bs2 The second bitset.
     * \return The concatenated bitset.
     */
    static boost::dynamic_bitset<> concatLoopDyn(const boost::dynamic_bitset<>& bs1, const boost::dynamic_bitset<>& bs2);

    /**
     * \brief Hash a set of fluents into a unique id.
     * \param[in] fl The set of fluents.
     * \return The unique id.
     */
    static KripkeWorldId hash_fluents_into_id(const FluentsSet& fl);

    /**
     * \brief Check if a set of fluents is consistent.
     * \param[in] to_check The set to check.
     * \details Exits with an error if the set is not consistent.
     * \return true if consistent (never returns false, will exit on failure).
     *
     *  \todo Integrate static_law for consistency checking.
     */
    static bool consistent(const FluentsSet& to_check);
};
