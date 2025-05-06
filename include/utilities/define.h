/**
 * \class define
 * \brief Class containing most of the custom types and symbols defined for the planner.
 *
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date March 31, 2019
 */

#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <queue>
#include <stack>
#include <memory>
#include <list>
#include <boost/dynamic_bitset.hpp>
#include <pthread.h>
#include <unistd.h>



/*****************************************************************
 * Constants
 ****************************************************************/
#define NEGATION_SYMBOL "-" /**< \brief The negation symbol to negate the \ref fluent */

/****************************************************************
 * Non class specific
 ****************************************************************/
typedef std::set<std::string> string_set; /**< \brief A representation of a conjunctive set of \ref fluent.
                                            *
                                            * Each element of the formula is a std::string (not grounded) */
typedef std::set<string_set> string_set_set; /**< \brief A representation of a formula in DNF formula.
                                              *
                                              * Each element of the formula is a \ref string_set (not grounded) */

/****************************************************************
 * Domain Related
 ****************************************************************/
typedef boost::dynamic_bitset<> fluent; /**< \brief A representation of a fluent through an unique id (short).
                                * 
                                * This representation is derived by applying \ref grounder::ground_fluent(const std::string&) const
                                * to the elements of \ref reader::m_fluents.
                                *
                                * The unique id of the readed fluent is always an odd number while to
                                * obtain the negated version is necessary to add 1.
                                *
                                * \todo is maybe better to store only the positive?
                                */
typedef std::set<fluent> fluent_set; /**< \brief A representation of a conjunctive set of \ref fluent.
                                      * 
                                      * This representation is used, for example, to represent
                                      * a world \ref kworld in a Kripke structure (\ref kstate).
                                      *
                                      * Each element of the set is a \ref fluent*/
typedef std::set<fluent_set> fluent_formula; /**< \brief A representation of a fluent formula in DNF.
                                      *
                                      * This representation is used, for example, to represent
                                      * a set of world (\ref kworld) of a Kripke structure (\ref kstate).
                                      *
                                      * \todo How the set of fluent has < implemented?
                                      * Each element of the set is a \ref fluent_set*/

typedef boost::dynamic_bitset<> agent; /**< \brief A representation of an agent through an unique id (short).
                                      *
                                      *  This representation is derived by applying \ref grounder::ground_agent(const std::string&) const
                                      * to the element of \ref reader::m_agents.*/
typedef std::set<agent> agent_set; /**< \brief A set of \ref agent.*/
typedef std::vector<agent> agent_list; /**< \brief A list of \ref agent.*/

typedef boost::dynamic_bitset<> action_id; /**< \brief The unique id (short) associated with each action.
                                      *
                                      *  This id is derived by applying \ref grounder::ground_action(const std::string&) const
                                      * to the element of \ref reader::m_actions.*/
typedef std::vector<action_id> action_id_list; /**< \brief A list of \ref action_id.*/

/*****************Grounder*****************/
typedef std::map<std::string, fluent> fluent_map; /**< \brief The map that associates to each \ref fluent name its grounded value.
                                                           * 
                                                           * @see grounder.*/
typedef std::map<std::string, agent> agent_map; /**< \brief The map that associates to each \ref agent name its grounded value.
                                                           * 
                                                           * @see grounder.*/
typedef std::map<std::string, action_id> action_name_map; /**< \brief The map that associates to each \ref agent name its grounded value.
                                                           * 
                                                           * @see grounder.*/

typedef std::map<fluent, std::string> reverse_fluent_map; /**< \brief The map that associates to each \ref fluent its name (for speed).
                                                           * 
                                                           * @see grounder.*/
typedef std::map<agent, std::string> reverse_agent_map; /**< \brief The map that associates to each \ref agent its name (for speed).
                                                           * 
                                                           * @see grounder.*/
typedef std::map<action_id, std::string> reverse_action_name_map; /**< \brief The map that associates to each \ref action_id its name (for speed).
                                                           * 
                                                           * @see grounder.*/

/** \brief The possible heuristics applicable to the domain.*/
enum heuristics
{
    NO_H, /**< Breadth first search is executed*/
    L_PG, /**< A planning graph is used to calculate the distance of each state from the goal.
                    * 
                    * For now for every state is computed a planning graph and then the lenght of it is returned.
                    * 
                    * \todo Optimize and maybe create only one planning graph and check where the state belongs.
                    * .*/
    S_PG, /**< A planning graph is used to calculate the sungoals distance of each state from the goal.
                    * 
                    * For now for every state is computed a planning graph and then the sum of the distances of each subgoals is returned
                    * 
                    * \todo Optimize and maybe create only one planning graph and check where the state belongs.
                    * .*/
    C_PG, /**< A single planning graph is used to calculate the distance of each grounded belief formula from initial state and goal.
                    * 
                    * This follows the idea of converting belief formula in fluent and just execute classical planning graph.
                    * .*/
    SUBGOALS, /**< For each state is calculate dthe number of (found or) missing subgoals
                    * 
                    * The group operator C is splitted in more belief_formulae.*/
    GNN /**< For each state a GNN is called to check against the perfect heuristics (training)
    * 
    * Work with Giovanni and Francois.*/
};

enum search_type
{
    BFS, /** Breadth first search is executed*/
    DFS, /** Depth first search is executed*/
    I_DFS /** Iterative Depth first search is executed*/
};

/** \brief The possible types of bisimulation algorithms.*/
enum bis_type
{
    BIS_NONE, /**< No Bisimulation is used*/
    PaigeTarjan, /**< The standard Paige-Tarjan Algorithm..*/
    FastBisimulation /**< The Fast-Bisimulation Algorithm introduced by Dovier et al, 2001.*/
};

/** \brief The possible restriction applicable to the domain.
 *
 * These restrictions have mostly to do with the Kripke representation and
 * with the type of inference that you could do on them.
 *
 * It has been introduced because it helps with the computation of the initial state (Finitary S5).*/
enum domain_restriction
{
    S5, /**< \brief Restricts only to S5 models.
         *
         * This is the default Initial description (\ref initially).
         * When this flag is on only C(phi) are accepted and they must cover all the fluents.
         * The possible cases are:
         * - *phi* -> all worlds must entail *phi*.
         * - C(B(i,*phi*)) -> all worlds must entail *phi*.
         * - C(B(i,*phi*) \ref BF_OR B(i,-*phi*)) -> only edges conditions.
         * - C(-B(i,*phi*) \ref BF_AND -B(i,-*phi*)) -> only edges conditions.
         *
         * With this type of model is associate the concept of *Knowledge* given
         * that the model gurantee the relations to be equivalent.*/
    K45, /**< \brief Restricts only to K45 models.
         *
         * With this type of model is associate the concept of *Belief*.*/
    NONE, /**< \brief No restrictions applied.
         */
    NONEG, /**< \brief No negative belief_formula are accepted.
            *
            * This is used when the planning graph heuristic is involved
            * because it cannot deal with negative \ref belief_formula
            * in the goal (no mutex yet).*/
    RESTRICTION_FAIL, /**< \brief The default case to guarantee consistency.*/
};

/** \brief The possible way of checking an action effects executability.*/
enum action_check
{
    EXE_POINTED__COND_POINTED, /**< \brief Both the executability and the conditional effects are checked only on the state.*/
    EXE_POINTED__COND_WORLD, /**< \brief The executionability is checked only on the state but the conditional effects are checked in every worlds.*/
    EXE_WORLD__COND_WORLD/**< \brief Both the executability and the conditional effects are checked in every world.*/
};


/*****************Parallelism*****************/

/** \brief The possible implementations of parallelism*/
enum parallel_type
{
    P_SERIAL, /**< Parallelism is disabled.*/
    P_PTHREAD, /**< parallelism is implemented using posix threads.*/
    P_FORK, /**< parallelism is implemented using forked processes.*/
    P_CHILD /**< denotes that the current process is a child*/
};

typedef struct {
    parallel_type ptype = P_SERIAL;
    bool pwait = false;
} parallel_input;

typedef struct {
    bool results_file;
    parallel_input pin;
    heuristics used_heur;
    search_type used_search;
    short IDFS_d;
    short IDFS_s;
    } pthread_params;

typedef struct {
    bool generate = false;
    bool useDFS   = true; // true denotes DFS while false denotes BFS
    int  depth    = 10;
    } ML_Dataset_Params; 



/****************** Agent's Attitudes related *******************/


/****************************************************************
 * Actions Related
 ****************************************************************/
class belief_formula;
typedef std::list<belief_formula> formula_list; /**< \brief A CNF formula of \ref belief_formula.
                                                 *
                                                 * Each element of the formula is a \ref belief_formula.*/

typedef std::set<belief_formula> bformula_set; /**< \brief A set of \ref belief_formula.*/

typedef std::map<agent, belief_formula> observability_map; /**< \brief Used to express the obsverbability conditions.
                                * 
                                * Each element associates an \ref agent to the observability conditions for an \ref effff.*/
//Associate each effect the condition
typedef std::map<fluent_formula, belief_formula> effects_map; /**< \brief Used to express the conditions of an action effects.
                                * 
                                * Each element associates an \ref action effect to its conditions.*/

enum event_type
{
    EPSILON, /**< \brief The null event.*/
    SIGMA, /**< \brief Event corresponding to ... */
    TAU /**< \brief Event corresponding to ... */
};

typedef std::set<event_type> event_type_set;
typedef std::set<std::pair<event_type, event_type> > event_type_relation;

/****************************************************************
 * States Related
 ****************************************************************/

/** \brief The possible states representation.
 *
 * Different state representation imply different advantages.
 * 
 * This type is used to allocate the class \ref state with the chosen representation.
 *
 */
enum state_type
{
    KRIPKE, /**< \brief The default state representation, through Kripke structures.
            * 
            * It encodes the mA* semantic and bases its entailment and its transition function
            * on reachability on Kripke structure. @see kstate.*/
    POSSIBILITIES, /**< \brief State representation based on possibilities and nwf-set theory.
            * 
            * It encodes the mA^rho semantic and bases its entailment and its transition function
            * on possibilities. @see poss_state. */
    OBDD /**< \brief State representation based on OBDD.
            * 
            *\todo Yet to study.*/
};

/*****************Kripke States*****************/
class kstate;

class kedge;
typedef std::string kedge_id; /**< \brief The id of an \ref kedge in a \ref kstate.
                                * 
                                * The id is calculated through an hash (kedge::hash_info_into_id()) of the info of the \ref kedge.
                                *
                                * This id is used to extract the \ref kedge from the \ref kstore to not create a new every time
                                * but to access the already created ones.
                                * @see kedge and kstore*/
/*typedef std::shared_ptr<const kedge> kedge_ptr;*/ /*< \brief A std::shared_pointer to a \ref kedge usually stored in \ref kstore.
                                                * @see kedge and kstore.
                                                * \todo transform into class for < into set, also add = and empty constructor, operator*,operator ->, = with string.*/
class kedge_ptr;
typedef std::set<kedge_ptr> kedge_ptr_set; /**< \brief A set of \ref kedge_ptr.
                                            * 
                                            * Mainly used to store all the \ref kedge of a \ref kstate without wasting memory.
                                            *
                                            * \todo The operator < for std::shared_ptr must be implemented
                                            * @see kedge and kstate*/

class kworld;
///\todo: remove string from code because they have terrible performances
typedef std::string kworld_id; /**< \brief The id of a \ref kworld in a \ref kstate.
                                * 
                                * The id is calculated through an hash (kworld::hash_fluents_into_id()) of the info of the \ref kworld.
                                *
                                * This id is used to extract the \ref kworld from the \ref kstore to not create a new every time
                                * but to access the already created ones.
                                * @see kworld and kstore*/
/*typedef std::shared_ptr<const kworld> kworld_ptr;*/ /*< \brief A std::shared_pointer to a \ref kworld usually stored in \ref kstore.
                                                   * 
                                                   * @see kworld and kstore.
                                                   * \todo transform into class for < into set.*/
class kworld_ptr;
typedef std::set<kworld_ptr> kworld_ptr_set; /**< \brief A set of \ref kworld_ptr.
                                            * 
                                            * Mainly used to store all the \ref kworld of a \ref kstate without wasting memory.
                                            *
                                            * \todo The operator < for std::shared_ptr must be implemented.
                                            * @see kworld and kstate.*/

typedef std::map<std::pair<kworld_ptr, event_type>, kworld_ptr> kstate_map;


/*****************Possibilities*****************/
class pstate;

class pworld;
typedef std::size_t pworld_id; /**< \brief The id of a \ref pworld in a \ref pstate.
                                *
                                * The id is calculated through an hash (pworld::hash_fluents_into_id()) of the info of the \ref pworld.
                                *
                                * This id is used to extract the \ref pworld from the \ref pstore to not create a new every time
                                * but to access the already created ones.
                                * @see pworld and pstore*/
/*typedef std::shared_ptr<const pworld> pworld_ptr;*/ /*< \brief A std::shared_pointer to a \ref pworld usually stored in \ref pstore.
                                                   *
                                                   * @see pworld and pstore.
                                                   * \todo transform into class for < into set.*/
class pworld_ptr;
typedef std::set<pworld_ptr> pworld_ptr_set; /**< \brief A set of \ref pworld_ptr.
                                            *
                                            * Mainly used to store all the \ref pworld of a \ref pstate without wasting memory.
                                            *
                                            * \todo The operator < for std::shared_ptr must be implemented.
                                            * @see pworld and pstate.*/

typedef std::map<agent, pworld_ptr_set> pworld_map; /**< \brief A map between agents and set of \ref pworld_ptr.
                                                    *
                                                    * @see pworld and pstate.*/

typedef std::map<pworld_ptr, pworld_map> pworld_transitive_map; /**< \brief A map, for every \ref pworld, of the beleifs of its agents.
                                                                 *
                                                                 * @see pworld and pstate.*/

typedef std::queue<pworld_ptr> pworld_queue; /**< \brief A queue of \ref pworld_ptr.
                                            *
                                            * \todo The operator < for std::shared_ptr must be implemented.
                                            * @see pworld and pstate.*/

typedef std::map<pworld_ptr, pworld_ptr> transition_map; /**< \brief A map that keeps track of the results of the transition function.
                                                          *
                                                          * @see pworld and pstate.*/

typedef std::vector<std::tuple<pworld_ptr, pworld_ptr, agent> > beliefs_vector; /**< \brief A vector of tuples (pw1, pw2, ag) that represent that in \ref pworld pw1 the \agent ag believes that the \ref \pworld pw2 is possible
                                                                               *
                                                                               * @see pworld and pstate.*/
typedef std::map<belief_formula, unsigned short> pg_bfs_score; /** A map to store the info of the classical planning graph*/


/*****************Possibilities Optimized*****************/
class pstate_opt;
typedef std::set<pstate_opt> pstate_opt_set; /**< \brief A set of \ref pstate_opt, used to store all the created ones.*/

typedef std::string pstate_opt_id; /**< \brief The id of a \ref pstate_opt.
                                *
                                * The id is calculated through an hash (pstate_opt::hash_fluents_into_id()) of the info of the \ref pstate_opt.
                                *
                                * This id is used to extract the \ref pstate_opt from the \ref p_opt_store to not create a new every time
                                * but to access the already created ones.
                                * @see pstate_opt and p_opt_store*/
class pstate_opt_ptr;
typedef std::set<pstate_opt_ptr> pstate_opt_ptr_set; /**< \brief A set of \ref pstate_opt_ptr.
                                            *
                                            * Mainly used to store all the reachable \ref pstate_opt of a \ref pstate_opt without wasting memory.
                                            *
                                            * \todo The operator < for std::shared_ptr must be implemented.
                                            * @see pstate_opt.*/
typedef std::map<agent, pstate_opt_ptr_set> pedges_opt; /**< \brief A map between agents and set of \ref pstate_opt_ptr
                                                       * 
                                                       * Each element consists of <\ref agent, \ref pstate_opt_ptr_set> and link each 
                                                       * \ref agent to a set of possibilities;
                                                       * remember that each possibility is a state itself.
                                                       *
                                                       * @see pstate_opt.*/

/****************** Delphic *******************/

// EVENTS

enum e_meta_condition
{
    act_eff,
    neg_act_eff,
    act_pre,
    neg_act_pre,
    none
};

class event;
typedef short event_id;
//typedef std::map<fluent, belief_formula> event_postconditions;
typedef std::set<e_meta_condition> event_metacond;
typedef fluent_formula event_postconditions;


class event_ptr;
typedef std::set<event_ptr> event_ptr_set; /**< \brief A set of \ref event_ptr.
//                                        *
//                                        * Mainly used to store all the reachable \ref event of a \ref event without wasting memory.
//                                        *
//                                        * \todo The operator < for std::shared_ptr must be implemented.
//                                        * @see event.*/

// PEMS
class pem;
typedef short pem_id;

//typedef std::pair<event_ptr, event_ptr> pem_edge;
typedef short agent_group;
//typedef std::map<agent_group, std::set<pem_edge>> pem_edges;

//enum action_type
//{
//    ONT, /** \brief Ontic actions */
//    //    ONT_NO,  /** \brief Ontic actions with No Oblivious agents */
//    SEN, /** \brief Sensing actions */
//    //    SEN_NO,  /** \brief Sensing actions with No Oblivious agents */
//    //    SEN_NP,  /** \brief Sensing actions with No Partially observant agents */
//    //    SEN_NOP, /** \brief Sensing actions with No Oblivious and Partially observant agents */
//    ANN, /** \brief Announcement actions */
//    //    ANN_NO,  /** \brief Announcement actions with No Oblivious agents */
//    //    ANN_NP,  /** \brief Announcement actions with No Partially observant agents */
//    //    ANN_NOP, /** \brief Announcement actions with No Oblivious and Partially observant agents */
//    SIZE /** \brief Number of action types */
//};
//
//enum agent_type
//{
//    FUL, /** \brief Fully observant agent */
//    PAR, /** \brief Partially observant agent */
//    OBL, /** \brief Oblivious agent */
//    ALL /** \brief All agents */
//};

/**< \brief A map between agents and a set of pairs of \ref event_ptr.
 *
 * Each element consists of <\ref agent, <\ref event_ptr, \ref event_ptr>> and
 * specifies for each \ref agent_type a set of pairs of \ref event_ptr (i.e.,
 * a set of edges.
 *
 * @see event.*/

//typedef std::map<event_id, event_ptr> pem_map; /**< \brief A map that stores the event model for a given type of action (identified through an id).*/

class pem_ptr;
typedef std::set<pem_ptr> pem_ptr_set;

typedef std::map<agent_group, event_ptr_set> event_map;
typedef std::map<event_ptr, event_map> pem_edges;

typedef std::map<std::pair<pworld_ptr, event_ptr>, pworld_ptr> update_map;

typedef std::map<agent, agent_group> agent_group_map;

class pem_store;
typedef std::set<event> event_set; /**< \brief A set of \ref event, used to store all the created ones.*/
typedef std::set<pem> pem_set; /**< \brief A set of \ref pem, used to store all the created ones.*/

/****************** Bisimulation *******************/

typedef unsigned short bis_label;
typedef std::set<bis_label> bis_label_set;

typedef std::map<kworld_ptr, std::map<kworld_ptr, bis_label_set> > kbislabel_map;
typedef std::map<pworld_ptr, std::map<pworld_ptr, bis_label_set> > pbislabel_map;

/****************** Bisimulation *******************/

/** \brief The list of the possible agents' attitude.*/
enum agents_attitudes
{
    //Partially Agents' Attitudes
    P_KEEPER, /** represents the set of those agents that will Keep their beliefs after the epistemic action.*/
    P_INSECURE, /** represents the set of those agents that are Insecure about their beliefs and,
                 * after the action, will not know the value of the fluent even if they knew it before*/

    //Fully Agents' Attitudes  
    F_TRUSTY, /**represents the Trusty agents that will believe what has been announced/sensed and
             * update their beliefs accordingly.*/
    F_MISTRUSTY, /**represents the Mistrusty agents that believe the opposite of what has been annnounced/sensed.*/
    F_UNTRUSTY, /**agents that formalizes the will not change their beliefs about the world no matter
                 * what the announcement/sensing says.*/
    F_STUBBORN, /**represents the Stubborn agents that will maintain their beliefs on f if they already know it.
                * If they do not know it they will update their belief.*/
    oblivious_att, /**represents the oblivious agents, to simplify the switch in the transition function.*/
    executor_att /**represents the executing agent, to simplify the switch in the transition function.*/
};

enum sub_functionIndex
{
    PHI_func,
    K_func,
    I_func,
    T_func,
    M_func,
    TRUE_U_func,
    FALSE_U_func,
    S_func,
    TRUE_CHI_func,
    FALSE_CHI_func,
    exec_FUNC
};

struct comp
{

    template<typename T>
    bool operator()(const T &l, const T &r) const
    {
        if (l.first == r.first)
            return l.second > r.second;

        return l.first < r.first;
    }
};

typedef std::map<std::pair< std::pair<pworld_ptr, unsigned short>, sub_functionIndex>, pworld_ptr, comp> transition_map_att; /**< \brief A map that keeps track of the results of the transition function when attitudes are involved.
                                                          *
                                                          * @see pworld and pstate.*/

/****************** Bisimulation *******************/

class bisimulation;

typedef int BIS_indexType;

//adjacency list

struct adjList
{
    BIS_indexType node;
    struct counter *countxS; //pointer to the count(x,S) of Paige&Tarjan
    struct adjList *next;
};

//adjacency list of G_1

struct adjList_1
{
    BIS_indexType node;
    struct adjList *adj; //pointer to the corresponding edge in G
    struct adjList_1 *next;
};

//counters for Paige and Tarjan

struct counter
{
    BIS_indexType value;
    BIS_indexType node;
};

//graph: array of nodes

struct graph
{
    int label; //label of the node
    BIS_indexType rank; //rank of the node
    bool WFflag; //identifies if the node is WF(0 NWF,1 WF)
    BIS_indexType nextInBlock; //pointer to the next node in the same block
    BIS_indexType prevInBlock; //pointer to the previous node in the same block
    BIS_indexType block; //pointer to the block
    struct counter *countxB; //pointer to count(x,B)
    struct adjList *adj; //pointer to the adjacency list
    struct adjList_1 *adj_1; //pointer to the adjacency list of G-1
};

//information related to Q-Blocks

struct qPartition
{
    BIS_indexType size; //number of nodes in the block
    //normalisation of the rank values: normalisation array
    BIS_indexType nextBlock; //pointer to the next Q-Block
    //normalisation of the rank values: copy of the forefather's rank
    BIS_indexType prevBlock; //pointer to the prvious Q-Block
    //during DFS-visit: color of the nodes
    BIS_indexType superBlock; //pointer to the X-Block containing the current Q-Block
    //during DFS: forefather
    BIS_indexType firstNode; //pointer to the first node in the block
    //during the first DFS visit in SCC(): finishing time
};

//information related to X-Blocks

struct xPartition
{
    BIS_indexType nextXBlock; //pointer to the next X-Block
    BIS_indexType prevXBlock; //pointer to the previous X-Block
    BIS_indexType firstBlock; //pointer to the first Block in X
};

/****************Automa related******************/
//DATA STRUCTURES BIS_USED FOR THE INPUT AND THE OUTPUT FROM FC2 FILES
typedef struct e_elem_struct e_elem;
typedef struct v_elem_struct v_elem;
// typedef struct bhtab_struct bhtab;
typedef struct automa_struct automa;

struct e_elem_struct
{
    int nbh; // Number of lables (behaviors) of a single Edge
    int *bh; // Array of Behaviors
    int tv; // Index (of the array of vertices) of the "To" Vertex
};

struct v_elem_struct
{
    int ne; // Number of Edges
    e_elem *e; // Vettore di edges
};

struct automa_struct
{
    int Nvertex;
    int Nbehavs;
    v_elem *Vertex;
};