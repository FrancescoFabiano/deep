/**
 * \brief Class used to check properties of formulae and modify them.
 * 
 *  The class implements static methods to facilitate
 *  the modification of the formulae and other.
 *
 * \see fluent_formula, belief_formula.
 * 
 * \todo public fields into private and getter and setter.
 * 
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date April 7, 2019
 */
#pragma once

#include "define.h"
#include "../formulae/belief_formula.h"

class helper
{
public:

    /** \brief Function that checks if two \ref fluent_set are consistent.
     * 
     * Two \ref fluent_set are consistent if they not contains a \ref fluent and its negation together.*/
    static bool is_consistent(const fluent_set &, const fluent_set &);

    /** \brief Function that returns the negation of a given \ref fluent.
     * 
     * @param[in] to_negate: the \ref fluent to negate
     * 
     * @return the negation of \p to_negate.*/
    static fluent negate_fluent(const fluent & to_negate);

    /** \brief Function that returns the negation of a given \ref fluent_formula.
     * 
     * @param[in] to_negate: the \ref fluent_formula to negate
     * 
     * @return the negation of \p to_negate.*/
    static fluent_formula negate_fluent_formula(const fluent_formula & to_negate);

    /** \brief Function that returns the positive version of a given \ref fluent.
     * 
     * @param[in] to_normalize: the \ref fluent to normalize
     * 
     * @return the normalized of fluent.*/
    static fluent normalize_fluent(const fluent & to_normalize);

    static bool is_negate(const fluent & f);




    /** \brief Function to set the truth value of a fluent in a world description.
     *   
     * @param[in] effect: the fluent to set.
     * @param[out] world_description: the \ref fluent_set contained inside a single world to modify.*/
    static void apply_effect(fluent effect, fluent_set& world_description);

    /** \brief Function to merge the results of an \ref ONTIC \ref action with a world description.
     *   
     * @param[in] effect: part of the effect of an \ref ONTIC \ref action in CNF form.
     * @param[out] world_description: the \ref fluent_set contained inside a single world.
     * 
     * @return the description of the world after \p effect has been applied to \p world_description.*/
    static void apply_effect(const fluent_set& effect, fluent_set& world_description);

    /* Set has == operator
     * \brief Function that checks if two \ref fluent_set are the same.
     * 
     * @param[in]  to_check_1: the first \ref fluent to check.
     * @param[in]  to_check_2: the second \ref fluent to check.
     * 
     * @return true: if each \ref fluent in \p to_check_1 exists in \p to_check_2 and vice-versa.
     * @return false: otherwise.*/
    /*  static bool is_the_same_ff(const fluent_set& to_check_1, const fluent_set& to_check_2);*/
    /* \brief Function that checks if two \ref fluent_formula are syntacticly the same.
     * 
     * This function uses recursively \ref is_the_same_ff(const fluent_set&, const fluent_set&).
     * 
     * @param[in]  to_check_1: the first \ref fluent_formula to check.
     * @param[in]  to_check_2: the second \ref fluent_formula to check.
     * 
     * @return true: if each \ref fluent_set in \p to_check_1 exists in \p to_check_2 and vice-versa.
     * @return false: otherwise.*/
    /* static bool is_the_same_ff(const fluent_formula& to_check_1, const fluent_formula& to_check_2);*/

    /** \brief Function that merges two conjunctive set of \ref fluent into one.
     *   
     * @param[in]  to_merge_1: the first conjunctive set of \ref fluent to merge.
     * @param[in]  to_merge_2: the second conjunctive set of \ref fluent to merge.
     * 
     * @return the union of all the \ref fluent in \p to_merge_1\2 if is consistent (exit otherwise).*/
    static fluent_set and_ff(const fluent_set& to_merge_1, const fluent_set& to_merge_2);

    /** \brief Function that merges two \ref fluent_formula into one.
     * 
     * @param[in] to_merge_1: the first \ref fluent_formula to merge.
     * @param[in] to_merge_2: the second \ref fluent_formula to merge.
     * 
     * @return the union of all the \ref fluent_set in \p to_merge_1\2 if is consistent (exit otherwise).*/
    static fluent_formula and_ff(const fluent_formula& to_merge_1, const fluent_formula& to_merge_2);

    /** \brief Function that checks if two \ref belief_formula are of the form B(i, *phi*) -- B(i, -*phi*) where *phi* is a \ref fluent_formula.
     * 
     * This function is useful to identify when an agent \p i knows the true value of *phi*. 
     * Is one of the accepted formulae in \ref S5.
     * 
     * @param[in] to_check_1: the first \ref belief_formula to check.
     * @param[in] to_check_2: the second \ref belief_formula to check.
     * @param[out] ret: the \ref fluent_formula containing *phi*.
     * 
     * @return true: if the two \ref belief_formula contain \ref fluent_formula that are the negation of each other.
     * @return false: otherwise.*/
    static bool check_Bff_notBff(const belief_formula& to_check_1, const belief_formula& to_check_2, std::shared_ptr<fluent_formula> ret);


    /** \brief Function that check that the \ref ONTIC effect doesn't have uncertainty (OR).
     * 
     * Then it calls apply_effect(const fluent_set&, const fluent_set&);
     *   
     * @param[in] effect: the effect of an \ref ONTIC \ref action.
     * @param[out] world_description: the description of the world after \p effect has been applied to \p world_description.
     * 
     * @return the description of the world after \p effect has been applied to \p world_description.*/
    static void apply_effect(const fluent_formula& effect, fluent_set& world_description);

    static int lenght_to_power_two(int length);

    static bool fluentset_empty_intersection(const fluent_set & set1, const fluent_set & set2);
    static bool fluentset_negated_empty_intersection(const fluent_set & set1, const fluent_set & set2);

    /** \brief Function that return the set of \ref agent that entails the obs condition.
     *
     * @param[in] map: the map that contains the tuples to check for entailment.
     * @param[in] state: the state in which to check the entailment.
     * @return the effects that are feasible in *this* with \p start as pointed world*.*/
    static agent_set get_agents_if_entailed(const observability_map & map, const pstate & state);

    /** \brief Function that return the \ref fluent_formula (effect) that entails the exe condition.
     *
     * @param[in] map: the map that contains the tuples to check for entailment.
     * @param[in] state: the state in which to check the entailment.
     * @return the effects that are feasible in \p state.*/
    static fluent_formula get_effects_if_entailed(const effects_map & map, const pstate & state);
};



