/**
 * \class planning_graph
 * \brief Class that implements the epistemic planning graph data structure for heuristics extrapolation.
 *
 * \details An e-Planning graph is a structure introduced in <https://aaai.org/ocs/index.php/ICAPS/ICAPS18/paper/view/17733>
 *  that, as the planning graph from the classical planning environment, it is used to extrapolate
 * qualitative value for the states.
 * 
 *
 * \copyright GNU Public License.
 * 
 * \author Francesco Fabiano.
 * \date September 24, 2019
 */

#pragma once

#include <utility>


#include "../utilities/Define.h"
#include "../actions/Action.h"
#include "../formulae/belief_formula.h"
#include "../domain/Domain.h"

/**
 * \class pg_action_level
 * \brief Class that implements an action level of the planning graph.
 *
 * 
 *
 * \copyright GNU Public License.
 * 
 * \author Francesco Fabiano.
 * \date September 24, 2019
 */



class pg_action_level
{
protected:
    //Waste memory
    //Use ptr
    /*\brief The set of executable actions in the relative \ref pg_state_level*/
    action_set m_actions;

    /*\brief The depth of the action level*/
    unsigned short m_depth = 0;

public:
    /*Constructor of this that calls the standard constructors for each field and set the depth to 0*/

    pg_action_level();
    /*Constructor of this that set the depth to 0 and m_actions
     *
     * @param[in] actions: the value to assign to m_actions. 
     */
    pg_action_level(const action_set & actions);
    /*Constructor of this that set the depth and m_actions
     *
     * @param[in] actions: the value to assign to m_actions. 
     * @param[in] depth: the value to assign to m_depth. 
     */
    pg_action_level(const action_set & actions, unsigned short depth);

    /*Setter of the field m_depth
     *
     * @param[in] depth: the value to assign to m_depth. 
     */
    void set_depth(unsigned short depth);
    /*Getter of the field m_depth
     *
     * @return: the value to assign to m_depth. 
     */
    unsigned short get_depth() const;

    /*Setter of the field m_actions
     *
     * @param[in] actions: the value to assign to m_actions. 
     */
    void set_actions(const action_set & actions);
    /*Function that add a single \ref action to *this*
     *
     * @param[in] act: the \ref action to add to m_actions if not present.
     */
    void add_action(const Action & act);

    /*Getter of the field m_actions
     *
     * @return: the value to assign to m_actions. 
     */
    const action_set & get_actions() const;
    /*The = operator
     * 
     * @param[in]to_assign: The object to copy in *this* */
    bool operator=(const pg_action_level& to_assign);

    //printing
    /* A printing function
     */
    void print() const;
};

/**
 * \class pg_state_level
 * \brief Class that implements a state layer of the epistemic planning graph data structure
 * 
 * \details In this implementation the state layer contains complete e-state in order to have a complete planning graph
 * 
 *
 * \copyright GNU Public License.
 * 
 * \author Francesco Fabiano.
 * \date September 24, 2019
 */

typedef std::map<fluent, short> pg_f_map;
typedef std::map<belief_formula, short> pg_bf_map;

class pg_state_level
{
private:

    /*\brief The map that associates each "grounded" \ref fluent to TRUE or FALSE**/
    pg_f_map m_pg_f_map;
    /*\brief The map that associates each "grounded" \ref belief_formula to TRUE or FALSE**/
    pg_bf_map m_pg_bf_map;

    /*\brief The depth of *this* (which state layer it is)*/
    unsigned short m_depth = 0;


    /*Setter of the field m_pg_f_map
     *
     * @param[in] to_set: the value to assign to m_pg_f_map. 
     */
    void set_f_map(const pg_f_map & to_set);

    /*Setter of the field m_pg_bf_map
     *
     * @param[in] to_set: the value to assign to m_pg_bf_map. 
     */
    void set_bf_map(const pg_bf_map & to_set);


    /*Function that return the truth value of a fluent in *this*
     *
     * @param[in] key: the key of the pair.
     * @return: the value of the pair with key \ref key.
     */
    short get_fluent_value(const fluent & key) const;

    /*Function that return the truth value of a belief_formula in *this*
     *
     * @param[in] key: the key of the pair.
     *   @return: the value of the pair with key \ref key.
     */
    short get_bf_value(const belief_formula & key) const;

    void build_init_f_map();

    void build_init_bf_map(const formula_list & goals);

    void insert_subformula_bf(const formula_list & fl, short value);

    void insert_subformula_bf(const belief_formula & bf, short value);


    template <class T>
    void build_init_f_map(T & eState);

    template <class T>
    void build_init_bf_map(const formula_list & goals, T & eState);

    template <class T>
    void insert_subformula_bf(const formula_list & fl, T & eState);

    template <class T>
    void insert_subformula_bf(const belief_formula & bf, T & eState);


    void get_base_fluents(const belief_formula & bf, FluentsSet & bf_base_fluents);

    bool exec_ontic(const Action & act, const pg_state_level & predecessor, bformula_set & false_bf);

    bool exec_epistemic(const Action & act, const pg_state_level & predecessor, bformula_set & false_bf);

    bool apply_ontic_effects(const belief_formula & bf, bformula_set & fl, const AgentSet & fully, bool & modified_pg);

    bool apply_epistemic_effects(fluent effect, const belief_formula & bf, bformula_set & fl, const AgentSet & fully, const AgentSet & partially, bool & modified_pg, unsigned short vis_cond);

public:

    //*Constructor that sets the depth to 0 and correctly initialize the maps*/
    pg_state_level();

    pg_state_level(const pg_state_level & to_assign);

    //*Constructor that sets the depth to 0 and correctly initialize the maps*/
    void initialize(const formula_list & goal);

    template <class T>
    void initialize(const formula_list & goal, T & eState);


    /*Constructor of this that set the depth and the maps.
     *
     * @param[in] f_map: the map with the fluents' truth values. 
     * @param[in] bf_map: the map with the belief_formulas' truth values. 
     * @param[in] depth: the value to assign to m_depth. 
     */
    pg_state_level(const pg_f_map & f_map, const pg_bf_map & bf_map, unsigned short depth);


    /*Getter of the field m_pg_f_map
     *
     * @return: the value to of m_pg_f_map. 
     */
    const pg_f_map & get_f_map() const;

    /*Getter of the field m_pg_bf_map
     *
     * @return: the value to of m_pg_bf_map. 
     */
    const pg_bf_map & get_bf_map() const;

    /*Getter of the field m_depth
     *
     * @return: the value to of m_depth. 
     */
    unsigned short get_depth() const;

    /*Setter of the field m_depth
     *
     * @param[in] depth: the value to assign to m_depth. 
     */
    void set_depth(unsigned short to_set);



    /*Function that modifies a pair fluent,bool in the field m_pg_f_map
     *
     * @param[in] key: the key of the pair.
     * @param[in] value: the value of the pair.
     */
    void modify_fluent_value(const fluent & key, short value);

    /*Function that modifies a pair belief_formula,bool in the field m_pg_bf_map
     *
     * @param[in] key: the key of the pair.
     * @param[in] value: the value of the pair.
     */
    void modify_bf_value(const belief_formula & key, short value);


    /*Function that checks satisfaction of a fluent on *this*.
     * 
     * @param[in] f: The fluent to check for entailment.
     * 
     * @return: true if the fluent is entailed.
     * @return: false otherwise.
     */
    bool pg_entailment(const fluent & f) const;

    /*Function that checks satisfaction of a belief_formula on *this*.
     * 
     * @param[in] bf: The belief_formula to check for entailment.
     * 
     * @return: true if the formula is entailed.
     * @return: false otherwise.
     */
    bool pg_entailment(const belief_formula & bf) const;
    /*Function that checks satisfaction of a CNF of belief_formula on *this*.
     * 
     * @param[in] fl: The CNF of belief_formula to check for entailment.
     * @return: true if the formula is entailed.
     * @return: false otherwise.
     */
    bool pg_entailment(const formula_list & fl) const;

    /*Function that checks if an action is executable on *this*.
     * 
     * @param[in] act: The act which we need to check for executability on *this*.
     * @return: true if the action's executability conditions are entailed.
     * @return: false otherwise.
     */
    bool pg_executable(const Action & act) const;

    bool compute_succ(const Action & act, const pg_state_level & predecessor, bformula_set & false_bf);

    short get_score_from_depth() const;


    /*The = operator
     * 
     * @param[in]to_assign: The object to copy in *this* */
    bool operator=(const pg_state_level& to_assign);
    void set_pg_state_level(const pg_state_level & to_assign);

};



/*\**********************************************************************
 class: planning_graph
 ************************************************************************/
//typedef std::map<fluent_set, unsigned short> pg_worlds_score; FOR FUTURE USE

class planning_graph
{
private:
    /*\brief The list of \ref pg_state_level that represents the state levels of the planning_graph*/
    std::vector< pg_state_level > m_state_levels;
    /*\brief The list of \ref pg_action_level that represents the action levels of the planning_graph*/
    std::vector< pg_action_level > m_action_levels;

    /*\brief The length of the planning_graph -- used after the goal is reached*/
    unsigned short m_pg_length = 0;
    /*\brief The sum of the depths of the levels that contain a subgoal*/
    unsigned short m_pg_sum = 0;
    /*\brief If the planning graph can find a solution*/
    bool m_satisfiable;

    /*\brief The list of the subgoals (possibly enanched by \ref heuristics_manager)*/
    formula_list m_goal;
    //pg_worlds_score m_worlds_score; FOR FUTURE USE
    /*\brief A map that contains the first level of encounter (if any) of a belief formula calculated by list_bf_classical*/
    pg_bfs_score m_bfs_score;
    /*\brief The set of action never executed in the planning_graph for optimization*/
    action_set m_never_executed;


    bformula_set m_belief_formula_false;

    /*Setter of the field m_satisfiable
     *
     * @param[in] sat: the value to assign to m_satisfiable. 
     */
    void set_satisfiable(bool sat);
    /*The main function that build the planning_graph layer by layer until the goal is found or the planning_graph is saturated*/
    void pg_build();
    /*Function add the next (depth + 1) state layer to m_state_levels
     *
     * @param[in] s_level: The level to add to m_state_levels.
     */
    void add_state_level(const pg_state_level & s_level);
    /*Function add the next (depth + 1) action layer to m_actions_levels
     * 
     * @param[in] a_level: The level to add to m_action_levels.
     */
    void add_action_level(const pg_action_level & a_level);

public:

    /*Constructor of *this* that set the initial state level and the goal from the domain
     * 
     * This version should be used when a single planning graph is created from the initial state
     */
    planning_graph();

    /*Constructor of *this* that set the initial state level from the domain and the goal as given
     * 
     *  This version should be used when a single planning graph is created from the initial state
     * @param[in] goal: The formula_list that describes the given goals.
     */
    planning_graph(const formula_list & goal);


    planning_graph(const planning_graph & pg);



    /*Constructor of *this* that set the initial state level from  a given eState and the goal from the domain
     * 
     * @param[in] eState: The initial eState from which we should extract the first state level.
     */
    template <class T>
    planning_graph(T & eState);

    /*Constructor of *this* that set the initial state level from a given eState and goal description
     * @param[in] goal: The formula_list that describes the given goals.
     * @param[in] eState: The initial eState from which we should extract the first state level.
     */
    template <class T>
    planning_graph(const formula_list & goal, T & eState);

    /*Function used by the constructors to properly initialize the various fields of the planning graph
     * 
     * @param[in] goal: The formula_list that describes the given goals.
     * @param[in] pg_init: the initial State level.
     */
    void init(const formula_list & goal, const pg_state_level & pg_init);



    /*Setter of the field m_length
     *
     * @param[in] length: the value to assign to m_length. 
     */
    void set_length(unsigned short length);
    /*Setter of the field m_sum
     *
     * @param[in] sum: the value to assign to m_sum. 
     */
    void set_sum(unsigned short sum);
    /*Setter of the field m_goal
     *
     * @param[in] goal: the value to assign to m_goal. 
     */
    void set_goal(const formula_list & goal);



    /*Function that tells if the planning graph can satisfy the domain.
     *
     * @return: True if a solution is found with the planning graph. 
     * @return: False otherwise. 
     */
    bool is_satisfiable() const;

    /*Getter of the field m_length
     *
     * @return: the value to assigned to m_length. 
     */
    unsigned short get_length() const;
    /*Getter of the field m_sum
     *
     * @return: the value to assigned to m_sum. 
     */
    unsigned short get_sum() const;

    /*Getter of the field m_bfs_score
     *
     * @return: the value to assigned to m_bfs_score. 
     */
    //  const pg_bfs_score & get_bfs_score();
    const std::vector< pg_state_level > & get_state_levels() const;
    const std::vector< pg_action_level > & get_action_levels() const;

    const formula_list & get_goal() const;
    const action_set & get_never_executed() const;
    const bformula_set & get_belief_formula_false() const;


    void set_pg(const planning_graph & to_assign);
    bool operator=(const planning_graph & to_assign);

    const pg_f_map & get_f_scores() const;

    const pg_bf_map & get_bf_scores() const;

    void print() const;
};