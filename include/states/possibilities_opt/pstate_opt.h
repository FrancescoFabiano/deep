/**
 * \brief Class that represents a possibility.
 *
 * \details  A possibility is one of the possible states description for EFP. It represent a single state in the
 *  mA^rho language.
 * 
 * @see pstate_opt and pstore_opt.
 *
 * \todo general: the && parameter in set (move) and in other places, is it better?
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date September 16, 2019
 */
#pragma once
#include <set>

#include "../../utilities/define.h"
#include "../../utilities/printer.h"

#include "../../domain/initially.h"
#include "../../actions/action.h"

class pstate_opt
{
    friend class pstate_opt_ptr;

private:

    /** \brief The set of  \ref fluent that describes how these are interpreted in *this*.*/
    fluent_set m_fluent_set;
    /** \brief The unique id of *this* computed with \ref hash_fluents_into_id().*/
    pstate_opt_id m_id;
    /** \brief The map containing the edges of the state.
     *
     * Each edge consists of <\ref agent, \ref pstate_opt_ptr_set> and link each \ref agent
     * to a set of possibilities; remember that each possibility is a state itself.
     *
     * @see pstate_opt, pstore_opt and agent.*/
    pedges_opt m_edges;
    /** \brief The number of pstate_opts with the same \ref m_fluent_set but different in a \ref m_edges.*/
    //unsigned short m_repetition = 0;

    /** \brief Function used to hash the the info of a possibility in a unique id.
     * 
     * @param[in] description: the interpretation of the \ref fluent in the possibility.
     * @return the unique id of the state.
     * 
     * @warning Useless if not moved to \ref kstore.*/
    pstate_opt_id hash_fluents_into_id(const fluent_set& description);
    /** \brief Function used to hash the the info of *this* in a unique id.
     *
     * @return the unique id of *this*. */
    pstate_opt_id hash_fluents_into_id();

    /** \brief Setter for the field \ref m_fluent_set.
     * 
     * Only accessible by the \ref pstate_opt_ptr.
     * 
     * @param[in] description: the set of \ref fluent to set as \ref m_fluent_set.*/
    void set_fluent_set(const fluent_set & description);
    /** \brief Function that uses the info of *this* to set its \ref m_id.
     * Only accessible by the \ref pstate_opt_ptr.*/
    void set_id();
    /** \brief Setter for the field \ref m_edges.
     * 
     * Only accessible by the \ref pstate_opt_ptr.
     * 
     * @param[in] edges: the map of \ref edges to set as \ref m_edges.*/
    void set_edges(const pedges_opt & edges);

    /** \brief Getter of \ref m_fluent_set.
     * 
     * Only accessible by the \ref pstate_opt_ptr.
     *     
     * @return the \ref set of fluents entailed by *this*.*/
    const fluent_set& get_fluent_set() const;
    /** \brief Getter of \ref m_id.
     *     
     * Only accessible by the \ref pstate_opt_ptr.
     * 
     * @return the short that is the unique id of *this*.*/
    pstate_opt_id get_id() const;
    /** \brief Getter of \ref m_edges.
     * 
     * Only accessible by the \ref pstate_opt_ptr.
     *     
     * @return the map of edges of *this*.*/
    const pedges_opt& get_edges() const;

    /** \brief Function that returns all the reachable \ref pstate_opt given the \ref agent and the staring \ref pstate_opt.
     * 
     * This function finds all the state that are transitively reachable from the starting state following
     * the edges labelled with the given \ref agent. It returns all the states where is necessary to check if
     * *phi* is entailed in order to verify B(\p ag, *phi*) and \p state is the starting point.
     * 
     * @see \ref belief_formula.
     * 
     * @param[in] ag: the label of the \ref kedge that the function follows to check the transitivity.
     *
     * @return a set of pointers to all the reachable states.
     *
     * 
     * \todo self-loop?*/
    const pstate_opt_ptr_set get_B_reachable_states(agent ag) const;
    /** \brief Function that returns all the reachable \ref pstate_opt given the \ref agent and the staring \ref pstate_opt.
     * 
     * This function modify the parameter so it's easy to check if a fixed point is reached.
     * This is useful for the operator C. Otherwise its better to use \ref get_B_reachable_states(agent, pstate_opt_ptr) const.
     * 
     * @see belief_formula, get_B_reachable_states(agent, pstate_opt_ptr) const and get_C_reachable_states(const agent_set &, pstate_opt_ptr state) const.
     * 
     * @param[in] ag: the label of the \ref kedge that the function follows to check the transitivity.
     *
     * @param[out] reached: the set of pointers to all the reachable states.
     *
     * 
     * \todo self-loop?*/
    bool get_B_reachable_states(agent ag, pstate_opt_ptr_set& reached) const;

    /** \brief Function that returns all the reachable \ref pstate_opt given a set of \ref agent and the staring \ref pstate_opt.
     * 
     * This function applies get_E_reachable_states(agent , pstate_opt_ptr) with all the agents inside the
     * given set and return the union of the reached \ref pstate_opt. It returns all the states where is necessary to check if
     * *phi* is entailed in order to verify E(\p ags, *phi*) and \p state is the starting point.
     * 
     * @see \ref belief_formula and get_B_reachable_states(agent, pstate_opt_ptr).
     * 
     * @param[in] ags: the set of label of the \ref kedge that the function follows to check the transitivity.
     *
     * @return a set of pointers to all the reachable states.
     *
     * 
     * \todo self-loop?*/
    const pstate_opt_ptr_set get_E_reachable_states(const agent_set & ags) const;
    /** \brief Function that returns all the reachable \ref pstate_opt given a set of \ref agent and the staring \ref pstate_opt.
     * 
     * This function modify the parameter so it's easy to check if a fixed point is reached.
     * This is useful for the operator C. Otherwise its better to use \ref get_E_reachable_states(const agent_set &, pstate_opt_ptr) const.
     * 
     * @see \ref belief_formula, get_B_reachable_states(agent, pstate_opt_ptr, pstate_opt_ptr_set) const and get_C_reachable_states(const agent_set &, pstate_opt_ptr state) const.
     * 
     * @param[in] ags: the set of label of the \ref kedge that the function follows to check the transitivity.
     *
     * @param[out] reached: the set of pointers to all the reachable states.
     *
     * 
     * \todo self-loop?*/
    bool get_E_reachable_states(const agent_set &ags, pstate_opt_ptr_set& reached) const;

    /** \brief Function that returns all the reachable \ref pstate_opt (in the *Common Knowledge* sense) given a \ref agent and the staring \ref pstate_opt.
     * 
     * This function applies get_E_reachable_states(const agent_set &, pstate_opt_ptr) on its on result until a fixed point is found.
     * It returns all the states where is necessary to check if *phi* is entailed in order to verify C(\p ags, *phi*) and \p state is the starting point.
     * 
     * @see \ref belief_formula, and get_E_reachable_states(const agent_set &, pstate_opt_ptr state).
     * 
     * @param[in] ags: the set of label of the \ref kedge that the function follows to check the *Common Knowledge* reachability.
     *
     * @return a set of pointers to all the reachable states.
     * 
     * \todo self-loop?*/
    const pstate_opt_ptr_set get_C_reachable_states(const agent_set &) const;

    /** \brief Function that builds the initial Kripke structure given the initial conditions in a structural way.
     *
     *         -# Compute the set of *phi* not know by \ref agent ag for each \ref agent;
     *         -# for each set compute the all the consistent \ref pstate_opt where *phi* changes;
     *         -# create a \ref kedge between all this \ref pstate_opt (transitively closed) labeled with ag;
     *         -# link all the classes of \ref pstate_opt with an ongoing \ref kedge labeled with the ag of the reaching class.
     * 
     * @see pstate_opt, kedge and initially.*/
    void build_initial_structural();
    /** \brief Function that builds the initial Kripke structure given the initial conditions pruning from a complete K structure.
     *
     *    - Thanks to \ref generate_initial_pstate_opts we generate all the possible states.
     *  We than store all the ones that respect \p ini_conditions *this* and all the other only in \ref pstore_opt.
     * 
     * @see pstate_opt, kedge and initially.
     *  
     * \todo If this method is the one is implemented the add functions in \ref pstore_opt must be changed to not create the object
     *       every time but just to retrieve the object since all of them are already created.*/
    void build_initial_prune();
    /** \brief Function used to build all the possible permutations of the \ref domain's \ref fluent.
     *  
     * The function creates all the permutation recursively and adds to *this* only the ones that respect \p ini_conditions.
     * All the other ones are stored in \ref pstore_opt for future uses.
     * 
     * @param[out] permutation: the permutation in construction.
     * @param[in] index: the index of the \ref fluent to add.
     * @param[out] ret: a set of pointer with all the states that entail the initially conditions*/
    void generate_initial_pstate_opts(fluent_set& permutation, int index, pstate_opt_ptr_set& ret);

    /** \brief Function used to build the \ref m_edges of the initial \ref pstate_opts.
     *  
     * The function creates all the edges and adds to *this* only the ones that respect \p ini_conditions.
     * All the other ones are stored in \ref pstore_opt for future uses.
     * @param[out] initial_pstate_opts: a set of pointer with all the states that starts with empty set.
     */
    void generate_initial_pedges(pstate_opt_ptr_set& initial_pstate_opts);

    /** \brief Function that removes the \ref pedge(s) that imply that \p ag is ignorant about \p known_ff from *this*.
     *  
     * @param[in] known_ff: the \ref fluent_formula known by \p ag.
     * @param[in] ag: the \ref agent that knows \p known_ff.
     * @param[out] initial_pstate_opts: a set of pointer with all the states that starts with all the edges.*/
    void remove_initial_pedge(const fluent_formula &known_ff, agent ag, pstate_opt_ptr_set& initial_pstate_opts);
    /** \brief Function check if a belief_formula removes some edges from the initial state.
     *  
     * For example whenever is B(i, *phi*) \/ B(i, -*phi*) is necessary to remove the edges
     * labeled with i that link \ref pstate_opt that entails *phi* and \ref pstate_opt -*phi*.
     * 
     * @param[in] to_check: the \ref belief_formula to check
     * @param[out] initial_pstate_opts: a set of pointer with all the states that starts with all the edges.
.*/
    void remove_initial_pedge_bf(const belief_formula &to_check, pstate_opt_ptr_set& initial_pstate_opts);

    /** \brief Function that return the set of \ref agent that entails the obs condition.
     *
     * @param[in] map: the map that contains the tuples to check for entailment.
     * @return the effects that are feasible in *this*.*/
    agent_set get_agents_if_entailed(const observability_map & map) const;

    /** \brief Function that return the \ref fluent_formula (effect) that entails the exe condition.
     *  
     * @param[in] map: the map that contains the tuples to check for entailment.
     * @return the effects that are feasible in *this*.*/
    fluent_formula get_effects_if_entailed(const effects_map & map) const;

    /** \brief Function that applies the transition function for the \ref action effect on *this* implementing the possibilities semantic.
     *
     * @param act[in]: the \ref action to be applied on *this*.
     * @param ret[in]: the \ref pstate_opt resulting from the \ref action.
     * @param fully_obs_agents[in]: the fully observant \ref agent set.
     * @param oblivious_obs_agents[in]: the oblivious \ref agent set.*/
    void execute_action(const action &act, pstate_opt &ret, agent_set &fully_obs_agents, agent_set &oblivious_obs_agents) const;
    /** \brief Function that recursively calculates the \ref pstate_opt resulting from the transition function.
     *
     * @param act[in]: the \ref action to be applied on *this*.
     * @param ret[in]: the \ref pstate_opt resulting from the \ref action.
     * @param current_pw[in]: the state being currently calculated.
     * @param calculated[in]: a map that keeps track of the results of the transition function.
     * @param p_obs_calculated[in]: a map that keeps track of the results of the transition function w.r.t. the beliefs of partially observant \ref agent.
     * @param fully_obs_agents[in]: the fully observant \ref agent set.
     * @param oblivious_obs_agents[in]: the oblivious \ref agent set.
     * @return the \ref pstate_opt resulting from the application of the transition function on "current_pw".*/
    pstate_opt_ptr execute_action_helper(const action &act, pstate_opt &ret, const pstate_opt_ptr &current_pw, agent_set &fully_obs_agents, agent_set &oblivious_obs_agents) const;
    /** \brief Function that applies the transition function for a \ref ONTIC \ref action effect on *this* implementing the possibilities semantic.
     *
     * The transition function is applied accordingly to mA^rho. Check the paper for more information.
     *
     * @see action.
     *
     * @param[in] act: the \ref ONTIC action to be applied on *this*.
     * @return the \ref pstate_opt that results after the execution.*/
    pstate_opt execute_ontic(const action &act) const;
    /** \brief Function that applies the transition function for a \ref SENSING \ref action effect on *this* implementing the possibilities semantic.
     *
     * The transition function is applied accordingly to mA^rho. Check the paper for more information.
     *
     * @see action.
     *
     * @param[in] act: the \ref SENSING action to be applied on *this*.
     * @return the \ref pstate_opt that results after the execution.*/
    pstate_opt execute_sensing(const action &act) const;
    /** \brief Function that applies the transition function for a \ref ANNOUNCEMENT \ref action effect on *this* implementing the possibilities semantic.
     *
     * The transition function is applied accordingly to mA^rho. Check the paper for more information.
     *
     * @see action.
     *
     * @param[in] act: the \ref ANNOUNCEMENT action to be applied on *this*.
     * @return the \ref pstate_opt that results after the execution.*/
    pstate_opt execute_announcement(const action &act) const;

public:

    /** \brief Empty constructor, call the default constructor of all the fields.*/
    pstate_opt();
    /** \brief Constructor with parameters.
     *Construct an object with the given info and then set the unique id.
     * the set of \ref fluent to set as \ref m_fluent_set.
     * 
     * @param[in] description: the set of \ref fluent to set as \ref m_fluent_set.*/
    pstate_opt(const fluent_set & description);
    /** \brief Constructor with parameters.
     *Construct an object with the given info and then set the unique id.
     * the set of \ref fluent to set as \ref m_fluent_set.
     * 
     * @param[in] description: the set of \ref fluent to set as \ref m_fluent_set.
     * @param[in] edges: the map of the edges of *this*.*/
    pstate_opt(const fluent_set & description, const pedges_opt & edges);
    /** \brief Copy constructor.
     * 
     * @param[in] state: the \ref pstate_opt to copy into *this*.*/
    pstate_opt(const pstate_opt & state);

    /** \brief Function that checks the entailment of a \ref fluent in *this*.
     *
     * The entailment of a \ref fluent in a pstate_opt is true if the \ref fluent
     * is entailed by its pointed state (\ref m_pointed).
     * 
     * @see \ref entails(fluent, pstate_opt_ptr) const.
     *  
     * @param[in] to_check: the \ref fluent that has to be checked if entailed in *this*.
     *
     * @return true: \p to_check is entailed in *this*;
     * @return false: \p -to_check is entailed in *this*.*/
    bool entails(fluent to_check) const;
    /** \brief Function that checks the entailment of a conjunctive set of \ref fluent in *this*.
     *
     * The entailment of a conjunctive set of \ref fluent in a pstate_opt is true if the conjunctive set of \ref fluent
     * is entailed by its pointed state (\ref m_pointed).
     * 
     * @see \ref entails(const fluent_set &, pstate_opt_ptr) const.
     *  
     * @param[in] to_check: the conjunctive set of \ref fluent that has to be checked if entailed in *this*.
     *
     * @return true: \p to_check is entailed in *this*;
     * @return false: \p -to_check is entailed in *this*.*/
    bool entails(const fluent_set & to_check) const;
    /**
     *\brief Function that checks the entailment of a DNF \ref fluent_formula in *this*.
     *
     * The entailment of a DNF \ref fluent_formula in a pstate_opt is true if the DNF \ref fluent_formula
     * is entailed by its pointed state (\ref m_pointed).
     * 
     * @see \ref entails(const fluent_formula &, pstate_opt_ptr) const.
     *  
     * @param[in] to_check: the DNF \ref fluent_formula that has to be checked if entailed in *this*.
     *
     * @return true: \p to_check is entailed in *this*;
     * @return false: \p -to_check is entailed in *this*.*/
    bool entails(const fluent_formula & to_check) const;

    /**
     *\brief Function that checks the entailment of a \ref belief_formula in *this*.
     *
     * The entailment of a \ref belief_formula in a pstate_opt is true if the \ref belief_formula
     * is entailed by *this*.
     * 
     * @see \ref entails(const belief_formula&, pstate_opt_ptr) const.
     *  
     * @param[in] to_check: the \ref belief_formula that has to be checked if entailed in *this*.
     *
     * @return true: \p to_check is entailed in *this*;
     * @return false: \p -to_check is entailed in *this*.*/
    bool entails(const belief_formula & to_check) const;
    /** \brief Function that checks the entailment of a \ref belief_formula on several possibilities.
     * 
     * This function eases the task to check the entailment from several starting \ref possibilities simultaneously.
     * This happens when we want to check the transitive closure of the \ref agent ag to test the formula B(ag, phi).
     * 
     * @see belief_formula.
     * 
     * @param[in] to_check: the \ref belief_formula that has to be checked if entailed from \p states.
     * @param[in] states: the pointers to the set of \ref pstate_opt where to start to check the entailment.
     *
     * @return true: \p to_check is entailed starting from all the \ref pstate_opt in \p states;
     * @return false: \p -to_check is entailed starting from all the \ref pstate_opt in \p states.
     * 
     * \todo self-loop?*/
    bool entails(const belief_formula & to_check, const pstate_opt_ptr_set & states) const;


    /** \brief Function that checks the entailment of a \ref formula_list (CNF of \ref belief_formula).
     * 
     * 
     * @param[in] to_check: the \ref formula_list that has to be checked if entailed starting from from \p world.
     *
     * @return true: \p to_check is entailed starting from \p world;
     * @return false: \p -to_check is entailed starting from \p world.
     * 
     * \todo self-loop?*/
    bool entails(const formula_list & to_check) const;
    /** \brief Function that builds the initial Kripke structure given the initial conditions.
     *
     * The building of the initial Kripke structure can be done in two different ways:
     *    - A method using a pruning techniques(\ref build_initial_prune).
     *    - A structural build (\ref build_initial_structural).
     * 
     * @see pstate_opt, kedge and initially.
     * 
     * \todo if first method is implemented the add functions in \ref pstore_opt must be changed to not create 
     *       the object every time but just to retrieve the object since all of them are already created.*/
    void build_initial();

    /** \brief Function that applies the transition function on *this* given an action.
     *
     * The transition function is applied accordingly to mA^rho. Check the paper for more information.
     *  
     * @see action.
     *  
     * @param[in] act: The action to be applied on *this*.
     * @return the \ref pstate_opt that results after the execution.
     * 
     * \todo The action must be executable on *this* otherwise it will return a null_ptr.*/
    pstate_opt compute_succ(const action & act) const;

    /** \brief Function that adds the belief of agent "ag" for the \ref pstate_opt "pw" *this*.
     *
     * @param[in] to: the \ref pstate_opt believed from \ref agent "ag"
     * @param[in] ag: the \ref agent.*/
    void add_edge(const pstate_opt_ptr & to, agent ag);
    /** \brief Function that removes the belief of \ref agent "ag" from \ref pstate_opt "from" to \ref pstate_opt "to".
     * 
     * @param[in] to: the \ref pstate_opt to remove
     * @param[in] ag: the \ref agent.*/
    void remove_edge(const pstate_opt_ptr & to, const agent ag);
    /** \brief The copy operator.
     *   
     * @param [in] to_copy: the \ref pstate_opt to assign to *this*.
     * @return true: if \p the assignment went ok.
     * @return false: otherwise.*/
    bool operator=(const pstate_opt & to_copy);

    /** \brief Function that prints the information of *this*.*/
    void print() const;

    /** \brief Function that prints the information of *this* in a Graphviz file.
     *
     * @param[in] graphviz: the ostream where to print the info of *this*.*/
    void print_graphviz(std::ostream& graphviz) const;

    /**
     *\brief The < operator based on the field \ref m_id.
     * Implemented to allow the ordering on set of \ref pstate_opt (used in \ref pstore_opt).
     * 
     * @see pstore_opt.
     *     
     * @param [in] to_compare: the \ref pstate_opt to compare with *this*.
     * @return true: if \p to_compare is smaller than *this*
     * @return false: otherwise.*/
    bool operator<(const pstate_opt& to_compare) const;

    bool is_smaller_state(const pstate_opt & to_check_1, const pstate_opt & to_check_2, pstate_opt_set& visited) const;



    /** \brief Function that return the sum_set of the two parameters by modifying the first one.
     *
     *  
     * @param[out] to_modify: the set in which is added \p factor2.
     * @param[in] factor2: the set to add to \p to_modify.*/
    template <class T>
    void sum_set(std::set<T> & to_modify, const std::set<T> & factor2) const;
    /** \brief Function that return the set difference of the two parameters by modifying the first one.
     *
     *  
     * @param[out] to_modify: the set from which is removed \p factor2.
     * @param[in] factor2: the set to remove from \p to_modify.*/
    template <class T>
    void minus_set(std::set<T> & to_modify, const std::set<T> & factor2) const;


    /*fluent_formula get_sensing_effects_if_entailed(const effects_map & map, const pstate_opt_ptr & start) const;*/
};

/**
 * \class pstate_opt_ptr
 * 
 * \brief A wrapper class for a std::shared_pointer to a \ref pstate_opt usually stored in \ref pstore_opt.
 * 
 * This class is necessary so we can use the < operator in the set of \ref pstate_opt_ptr.
 * If we dom't use the wrapper pointers to the same elements are different.
 *
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date April 29, 2019
 */
class pstate_opt_ptr
{
private:
    /**\brief the pointer that is wrapped by *this*.*/
    std::shared_ptr<pstate_opt> m_ptr;

public:
    /**\brief Constructor without parameters.*/
    pstate_opt_ptr();
    /**\brief Constructor with parameters.
     *
     * This constructor uses const pointer, this means that the pointer is copied
     * and the counter of the shared pointer is increased (std implementation).
     * 
     * @param[in] ptr: the pointer to assign to \ref m_ptr.
     * @param[in] repetition: the value to give to \ref m_repetition, default to 0.
     */
    pstate_opt_ptr(const std::shared_ptr<pstate_opt> & ptr);
    /**\brief Constructor with parameters.
     *
     * This constructor uses non-const pointer, this means that the pointer is copied
     * in \ref m_ptr and \p ptr becomes empty (std implementation).
     *  
     * @param[in] ptr: the pointer to assign to \ref m_ptr.
     * @param[in] repetition: the value to give to \ref m_repetition, default to 0.
     */
    pstate_opt_ptr(std::shared_ptr<pstate_opt>&& ptr);
    /** \brief Constructor with parameters.
     *
     * This constructor build a pointer to the given parameter.
     *  
     * @param[in] state: the \ref pstate_opt that *this* (\ref m_ptr) should point.
     * @param[in] repetition: the value to give to \ref m_repetition, default to 0.*/
    pstate_opt_ptr(const pstate_opt & state);

    /**\brief Getter for the field \ref m_ptr.
     *  
     * @return a copy of the pointer \ref m_ptr.*/
    std::shared_ptr<pstate_opt> get_ptr() const;

    /**\brief Setter for the field \ref m_ptr.
     *
     * This setter uses const parameter, this means that the pointer is copied
     * and the counter of the shared pointer is increased (std implementation).
     * 
     * @param[in] ptr: the pointer to assign to \ref m_ptr.*/
    void set_ptr(const std::shared_ptr<pstate_opt> & ptr);
    /**\brief Setter for the field \ref m_ptr (move constructor).
     * 
     * This setter uses non-const parameter, this means that the pointer is copied
     * in \ref m_ptr and \p ptr becomes empty (std implementation).
     *  
     * @param[in] ptr: the pointer to assign to \ref m_ptr.*/
    void set_ptr(std::shared_ptr<pstate_opt>&& ptr);

    /** \brief Getter of the field \ref m_repetition.
     * 
     * @return the value to of \ref m_repetition.*/
    // unsigned short get_repetition() const;

    /** \brief Function that return the field m_fluent_set of the pointed \ref pstate_opt.
     *     
     * @return the \ref fluent_set that is the description of the \ref pstate_opt pointed by \ref m_ptr.*/
    const fluent_set & get_fluent_set() const;

    /** \brief Function that return the field m_id of the pointed \ref pstate_opt + \ref m_repetition.
     *     
     * @return the \ref pstate_opt_id that is the id of the \ref pstate_opt pointed by \ref m_ptr + \ref m_repetition.*/
    pstate_opt_id get_id() const;

    /** \brief Function that return the field m_edges of the pointed \ref pstate_opt.
     *     
     * @return the \ref m_edges of the \ref pstate_opt pointed by \ref m_ptr.*/
    const pedges_opt & get_edges() const;


    /** \brief Function that check the entailment of a single \ref fluent in \ref m_ptr.
     * 
     * 
     * @param[in] to_check: the \ref fluent that has to be checked if entailed by \ref m_ptr (if is present in \ref m_ptr).
     *
     * @return true: \p to_check is entailed;
     * @return false: \p -to_check is entailed.*/
    bool entails(fluent to_check) const;
    /**
     *\brief Function that check the entailment of a conjunctive set of \ref fluent in \ref m_ptr.
     * 
     * @param[in] to_check: the conjunctive set of \ref fluent that has to be checked if entailed by \ref m_ptr.
     *
     * @return true: \p to_check is entailed;
     * @return false: \p -to_check is entailed.*/
    bool entails(const fluent_set& to_check) const;
    /**
     *\brief Function that check the entailment of a DNF \ref fluent_formula in \ref m_ptr.
     * 
     * @param[in] to_check: the DNF formula that has to be checked if entailed by \ref m_ptr.
     *
     * @return true: \p to_check is entailed;
     * @return false: \p -to_check is entailed.*/
    bool entails(const fluent_formula & to_check) const;
    /**
     *\brief Function that checks the entailment of a \ref belief_formula in the element pointed by *this*.
     *
     * The entailment of a \ref belief_formula in a pstate_opt is true if the \ref belief_formula
     * is entailed by the element pointed by *this*.
     * 
     * @see \ref entails(const belief_formula&, pstate_opt_ptr) const.
     *  
     * @param[in] to_check: the \ref belief_formula that has to be checked if entailed in the element pointed by *this*.
     *
     * @return true: \p to_check is entailed in the element pointed by *this*;
     * @return false: \p -to_check is entailed in the element pointed by *this*.*/
    bool entails(const belief_formula & to_check) const;
    /** \brief Function that checks the entailment of a \ref formula_list (CNF of \ref belief_formula).
     * 
     * 
     * @param[in] to_check: the \ref formula_list that has to be checked if entailed starting from from \p world.
     *
     * @return true: \p to_check is entailed starting from \p the element pointed by *this*;
     * @return false: \p -to_check is entailed starting from \p the element pointed by *this*.
     * 
     * \todo self-loop?*/
    bool entails(const formula_list & to_check) const;

    /**\brief The operator =.
     *
     * This operator assign the parameter without destroying \p to_copy.
     * 
     * @param[in] to_copy: the \ref pstate_opt_ptr to assign to *this*.
     * @return true: if the assignment went through.
     * @return false: otherwise.*/
    bool operator=(const pstate_opt_ptr & to_copy);
    /**\brief The operator < for set operation.
     *
     * The ordering is based on the pointed object and not on the pointer itself so we have one
     * copy of each pointed object.
     * 
     * @param[in] to_compare: the \ref pstate_opt_ptr to check for odering.
     * 
     * @return true: if *this* is smaller than \p to_compare.
     * @return false: otherwise.*/

    bool operator<(const pstate_opt_ptr & to_compare) const;
    /**\brief The operator ==.
     * 
     * @param[in] to_compare: the \ref pstate_opt_ptr to confront with *this*.
     * @return true: if *this* is equal to \p to_compare.
     * @return false: otherwise.*/
    bool operator==(const pstate_opt_ptr & to_compare) const;
};