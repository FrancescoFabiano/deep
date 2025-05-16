#pragma once

#include <string>
#include <vector>
#include <set>
#include <map>
#include <queue>
#include <memory>
#include <list>
#include <boost/dynamic_bitset.hpp>


/**
 * \def NEGATION_SYMBOL
 * \brief The negation symbol to negate a fluent.
 */
#define NEGATION_SYMBOL "-" 

/// \name Non-class specific types
///@{
using StringsSet = std::set<std::string>; ///< Conjunctive set of fluents (not grounded).
using StringSetsSet = std::set<StringsSet>; ///< Formula in DNF (not grounded).
///@}

/// \name Domain Related
///@{
using Fluent = boost::dynamic_bitset<>; ///< Unique id representation of a fluent.
using FluentsSet = std::set<Fluent>; ///< Conjunctive set of fluents.
using FluentFormula = std::set<FluentsSet>; ///< Fluent formula in DNF.

using Agent = boost::dynamic_bitset<>; ///< Unique id representation of an agent.
using AgentsSet = std::set<Agent>;
using AgentsList = std::vector<Agent>;

using ActionId = boost::dynamic_bitset<>; ///< Unique id for each action.
using ActionIdsList = std::vector<ActionId>;

using FluentMap = std::map<std::string, Fluent>; ///< Map from fluent name to grounded value.
using AgentsMap = std::map<std::string, Agent>;
using ActionNamesMap = std::map<std::string, ActionId>;

using ReverseFluentsMap = std::map<Fluent, std::string>;
using ReverseAgentsMap = std::map<Agent, std::string>;
using ReverseActionNamesMap = std::map<ActionId, std::string>;
///@}

/**
 * \enum heuristics
 * \brief The possible heuristics applicable to the domain.
 */
enum class heuristics {
    NO_H,    ///< Breadth first search.
    L_PG,    ///< Planning graph for state-goal distance.
    S_PG,    ///< Planning graph for sum of subgoals distances.
    C_PG,    ///< Classical planning graph for belief formulae.
    SUBGOALS,///< Number of found/missing subgoals.
    GNN      ///< GNN-based heuristic.
};

/**
 * \enum search_type
 * \brief The possible search strategies.
 */
enum class search_type {
    BFS,   ///< Breadth first search.
    DFS,   ///< Depth first search.
    I_DFS,  ///< Iterative deepening DFS.
    HFS,   ///< Heuristic first search.
};

/**
 * \enum parallel_type
 * \brief The possible implementations of parallelism.
 */
enum class parallel_type {
    P_SERIAL,   ///< Parallelism disabled.
    P_PTHREAD,  ///< POSIX threads.
    P_FORK,     ///< Forked processes.
    P_CHILD     ///< Current process is a child.
};

/**
 * \struct parallel_input
 * \brief Input parameters for parallel execution.
 */
struct parallel_input {
    parallel_type ptype = parallel_type::P_SERIAL;
    bool pwait = false;
};

/**
 * \struct pthread_params
 * \brief Parameters for pthread-based execution.
 */
struct pthread_params {
    bool results_file = false;
    parallel_input pin;
    heuristics used_heur = heuristics::NO_H;
    search_type used_search = search_type::BFS;
    short IDFS_d = 0;
    short IDFS_s = 0;
};

/**
 * \struct ML_Dataset_Params
 * \brief Parameters for ML dataset generation.
 */
struct ML_Dataset_Params {
    bool generate = false;
    bool useDFS = true; ///< True for DFS, false for BFS.
    int depth = 10;
};

/// \name Actions Related
///@{
class BeliefFormula;
using FormulaeList = std::list<BeliefFormula>; ///< CNF formula of BeliefFormula.
using FormulaeSet = std::set<BeliefFormula>;
using observability_map = std::map<Agent, BeliefFormula>; ///< Agent to observability conditions.
using effects_map = std::map<FluentFormula, BeliefFormula>; ///< Action effect to its conditions.
///@}

/**
 * \enum event_type
 * \brief Types of events.
 */
enum class event_type {
    EPSILON, ///< Null event.
    SIGMA,   ///< Event corresponding to ...
    TAU      ///< Event corresponding to ...
};

using event_type_set = std::set<event_type>;
using event_type_relation = std::set<std::pair<event_type, event_type>>;

/// \name eState
///@{
class KripkeState;
class KripkeWorld;
using KripkeWorldId = std::size_t; ///< ID of a KripkeWorld
class KripkeWorldPointer;
using KripkeWorldPointersSet = std::set<KripkeWorldPointer>;
using KripkeWorldPointersMap = std::map<Agent, KripkeWorldPointersSet>;
using KripkeWorldPointersTransitiveMap = std::map<KripkeWorldPointer, KripkeWorldPointersMap>;
using KripkeWorldPointersQueue = std::queue<KripkeWorldPointer>;
using TransitionMap = std::map<KripkeWorldPointer, KripkeWorldPointer>;
using BeliefsVector = std::vector<std::tuple<KripkeWorldPointer, KripkeWorldPointer, Agent>>;
using pg_bfs_score = std::map<BeliefFormula, unsigned short>;
///@}

/// \name Bisimulation
///@{
using bis_label = unsigned short;
using bis_label_set = std::set<bis_label>;
using pbislabel_map = std::map<KripkeWorldPointer, std::map<KripkeWorldPointer, bis_label_set>>;
class bisimulation;
using BIS_indexType = int;
///@}

/**
 * \struct adjList
 * \brief Adjacency list node for bisimulation graph.
 */
struct adjList {
    BIS_indexType node;
    struct counter* countxS; ///< Pointer to the count(x,S) of Paige&Tarjan.
    adjList* next;
};

/**
 * \struct adjList_1
 * \brief Adjacency list for G_1.
 */
struct adjList_1 {
    BIS_indexType node;
    adjList* adj;
    adjList_1* next;
};

/**
 * \struct counter
 * \brief Counter for Paige and Tarjan algorithm.
 */
struct counter {
    BIS_indexType value;
    BIS_indexType node;
};

/**
 * \struct graph
 * \brief Node in the bisimulation graph.
 */
struct graph {
    int label;
    BIS_indexType rank;
    bool WFflag;
    BIS_indexType nextInBlock;
    BIS_indexType prevInBlock;
    BIS_indexType block;
    counter* countxB;
    adjList* adj;
    adjList_1* adj_1;
};

/**
 * \struct qPartition
 * \brief Information related to Q-Blocks.
 */
struct qPartition {
    BIS_indexType size;
    BIS_indexType nextBlock;
    BIS_indexType prevBlock;
    BIS_indexType superBlock;
    BIS_indexType firstNode;
};

/**
 * \struct xPartition
 * \brief Information related to X-Blocks.
 */
struct xPartition {
    BIS_indexType nextXBlock;
    BIS_indexType prevXBlock;
    BIS_indexType firstBlock;
};

/// \name Automata related
///@{
struct e_elem_struct;
struct v_elem_struct;
struct automa_struct;

using e_elem = e_elem_struct;
using v_elem = v_elem_struct;
using automa = automa_struct;

/**
 * \struct e_elem_struct
 * \brief Edge element for automata.
 */
struct e_elem_struct {
    int nbh; ///< Number of labels (behaviors) of a single edge.
    int* bh; ///< Array of behaviors.
    int tv;  ///< Index of the "To" vertex.
};

/**
 * \struct v_elem_struct
 * \brief Vertex element for automata.
 */
struct v_elem_struct {
    int ne;      ///< Number of edges.
    e_elem* e;   ///< Array of edges.
};

/**
 * \struct automa_struct
 * \brief Automaton structure.
 */
struct automa_struct {
    int Nvertex;
    int Nbehavs;
    v_elem* Vertex;
};
///@}