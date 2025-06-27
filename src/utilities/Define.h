#pragma once

#include <boost/dynamic_bitset.hpp>
#include <list>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "VectorBisWrapper.h"

/**
 * \struct OutputPaths
 * \brief Contains static constants for commonly used output and log folder
 * paths.
 */
struct OutputPaths {
  static constexpr auto LOGS_FOLDER = "logs";  ///< Path to the log folder.
  static constexpr auto OUTPUT_FOLDER = "out"; ///< Path to the output folder.
  static constexpr auto EXEC_PLAN_FOLDER =
      "out/plan_exec"; ///< Path to the output folder.
  static constexpr auto DATASET_NN_OUTPUT_FOLDER =
      "out/NN"; ///< Path to the NN output folder.
  static constexpr auto DATASET_TRAINING_FOLDER =
      "out/NN/Training"; ///< Path to the training folder.
  static constexpr auto DATASET_INFERENCE_FOLDER =
      "out/NN/Inference"; ///< Path to the inference folder.
  static constexpr auto DATASET_NN_DATASET_MAPPED =
      "map"; ///< The suffix for the NN dataset mapped file.
  static constexpr auto DATASET_NN_DATASET_HASHED =
      "hash"; ///< The suffix for the NN dataset hashed file.
  static constexpr auto DATASET_NN_DATASET_MERGED =
      "merged"; ///< The suffix for the NN dataset hashed file.
  static constexpr auto DATASET_NN_DATASET_SEPARATED =
      "separated"; ///< The suffix for the NN dataset hashed file.
};

/**
 * \def NEGATION_SYMBOL
 * \brief The negation symbol to negate a fluent.
 */
#define NEGATION_SYMBOL "-"

/// \name Non-class specific types
///@{
using StringsSet =
    std::set<std::string>; ///< Conjunctive set of fluents (not grounded).
using StringSetsSet = std::set<StringsSet>; ///< Formula in DNF (not grounded).
///@}

/// \name Domain Related
///@{
using Fluent =
    boost::dynamic_bitset<>;         ///< Unique id representation of a fluent.
using FluentsSet = std::set<Fluent>; ///< Conjunctive set of fluents.
using FluentFormula = std::set<FluentsSet>; ///< Fluent formula in DNF.

using Agent =
    boost::dynamic_bitset<>; ///< Unique id representation of an agent.
using AgentsSet = std::set<Agent>;
using AgentsList = std::vector<Agent>;

using ActionId = boost::dynamic_bitset<>; ///< Unique id for each action.
using ActionIdsList = std::vector<ActionId>;

using FluentMap =
    std::map<std::string, Fluent>; ///< Map from fluent name to grounded value.
using AgentsMap = std::map<std::string, Agent>;
using ActionNamesMap = std::map<std::string, ActionId>;

using ReverseFluentsMap = std::map<Fluent, std::string>;
using ReverseAgentsMap = std::map<Agent, std::string>;
using ReverseActionNamesMap = std::map<ActionId, std::string>;
///@}

/**
 * \enum Heuristics
 * \brief The possible heuristics applicable to the domain.
 */
enum class Heuristics {
  L_PG,     ///< Planning BisGraph for State-goal distance.
  S_PG,     ///< Planning BisGraph for sum of subgoals distances.
  C_PG,     ///< Classical planning BisGraph for belief formulae.
  SUBGOALS, ///< Number of found/missing subgoals.
  GNN,      ///< GNN-based heuristic.
  ERROR,    ///< Error state, used to detect uninitialized heuristics.
};

/**
 * \enum SearchType
 * \brief The possible search strategies.
 */
enum class SearchType {
  BFS,  ///< Breadth first search.
  DFS,  ///< Depth first search.
  IDFS, ///< Iterative depth first search.
  HFS,  ///< Heuristic first search.
};

/// \name Belief Formulae Related
///@{
class BeliefFormula;
using FormulaeList =
    std::list<BeliefFormula>; ///< CNF formula of BeliefFormula.
using FormulaeSet = std::set<BeliefFormula>;
using ObservabilitiesMap =
    std::map<Agent, BeliefFormula>; ///< Agent to observability conditions.
using EffectsMap = std::map<FluentFormula, BeliefFormula>; ///< Action effect to
                                                           ///< its conditions.
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
inline std::size_t max_dimension_KripkeWorldID_size =
    std::numeric_limits<KripkeWorldId>::max();
inline std::size_t max_KripkeWorldID_digits =
    std::to_string(max_dimension_KripkeWorldID_size).length();
class KripkeWorldPointer;
using KripkeWorldPointersSet = std::set<KripkeWorldPointer>;
using KripkeWorldPointersMap = std::map<Agent, KripkeWorldPointersSet>;
using KripkeWorldPointersTransitiveMap =
    std::map<KripkeWorldPointer, KripkeWorldPointersMap>;
using KripkeWorldPointersQueue = std::queue<KripkeWorldPointer>;
using TransitionMap = std::map<KripkeWorldPointer, KripkeWorldPointer>;
using BeliefsVector =
    std::vector<std::tuple<KripkeWorldPointer, KripkeWorldPointer, Agent>>;
using pg_bfs_score = std::map<BeliefFormula, unsigned short>;
///@}
///

/// \name Bisimulation
///@{\todo Make sure the pointer cannot be unique
// STRUCTURE DECLARATION
// index is the type of the implicit pointers to the array
// hence is a number that belongs to the range [-1...numberOfNodes]
// at the moment we simply use integer;
// later it could become log(numberOfNodes) bits

using BisLabel = unsigned short;
using BisLabelsSet = std::set<BisLabel>;
using BisLabelsMap =
    std::map<KripkeWorldPointer, std::map<KripkeWorldPointer, BisLabelsSet>>;
using BisIndexType = int;

// CONSTANT DECLARATION
constexpr int BisMAXbhtmp = 400;
constexpr int BisUsed = 1;
constexpr int BisNotUsed = 0;
constexpr int BisDeleted = -1;
constexpr int BisToDelete = -2;
constexpr int BisPreAllocatedIndex = 100;
constexpr BisIndexType BIS_WHITE = 0;
constexpr BisIndexType BIS_GRAY = 1;
constexpr BisIndexType BIS_BLACK = 2;
constexpr BisIndexType BIS_NIL = -1;

struct BisCounter;
/**
 * \struct BisAdjList
 * \brief Adjacency list node for bisimulation graph.
 */
struct BisAdjList {
  BisIndexType node{}; ///< Node index.
  std::shared_ptr<BisCounter>
      countxS{}; ///< Pointer to the count(x,S) of Paige&Tarjan.
  std::shared_ptr<BisAdjList> next{}; ///< Next adjacency node.
};

/**
 * \struct BisAdjList_1
 * \brief Adjacency list for G_1.
 */
struct BisAdjList_1 {
  BisIndexType node{};                  ///< Node index.
  std::shared_ptr<BisAdjList> adj{};    ///< Pointer to adjacency list.
  std::shared_ptr<BisAdjList_1> next{}; ///< Next adjacency node.
};

/**
 * \struct BisCounter
 * \brief Counter for Paige and Tarjan algorithm.
 */
struct BisCounter {
  BisIndexType value{}; ///< Counter value.
  BisIndexType node{};  ///< Node index.
};

/**
 * \struct BisGraph
 * \brief Node in the bisimulation graph.
 */
struct BisGraph {
  int label{};                           ///< Node label.
  BisIndexType rank{};                   ///< Node rank.
  bool WFflag{};                         ///< Well-founded flag.
  BisIndexType nextInBlock{};            ///< Next node in block.
  BisIndexType prevInBlock{};            ///< Previous node in block.
  BisIndexType block{};                  ///< Block index.
  std::shared_ptr<BisCounter> countxB{}; ///< Pointer to count(x,B).
  std::shared_ptr<BisAdjList> adj{};     ///< Pointer to adjacency list.
  std::shared_ptr<BisAdjList_1> adj_1{}; ///< Pointer to G_1 adjacency list.
};

/**
 * \struct Bis_qPartition
 * \brief Information related to Q-Blocks.
 */
struct Bis_qPartition {
  BisIndexType size{};       ///< Size of the block.
  BisIndexType nextBlock{};  ///< Next block index.
  BisIndexType prevBlock{};  ///< Previous block index.
  BisIndexType superBlock{}; ///< Super block index.
  BisIndexType firstNode{};  ///< First node in block.
};

/**
 * \struct Bis_xPartition
 * \brief Information related to X-Blocks.
 */
struct Bis_xPartition {
  BisIndexType nextXBlock{}; ///< Next X-block index.
  BisIndexType prevXBlock{}; ///< Previous X-block index.
  BisIndexType firstBlock{}; ///< First block in X-block.
};

///@}

/// \name Automata related
///@{
struct Bis_eElemStruct;
struct Bis_vElemStruct;
struct BisAutomataStruct;

using Bis_eElem = Bis_eElemStruct;
using Bis_vElem = Bis_vElemStruct;
using BisAutomata = BisAutomataStruct;

/**
 * \struct Bis_eElemStruct
 * \brief Edge element for automata.
 */
struct Bis_eElemStruct {
  int nbh{}; ///< Number of labels (behaviors) of a single edge.
  VectorBisWrapper<int> bh{}; ///< Array of behaviors.
  int tv{};                   ///< Index of the "To" vertex.
};

/**
 * \struct Bis_vElemStruct
 * \brief Vertex element for automata.
 */
struct Bis_vElemStruct {
  int ne{};                        ///< Number of edges.
  VectorBisWrapper<Bis_eElem> e{}; ///< Array of edges.
};

/**
 * \struct BisAutomataStruct
 * \brief Automaton structure.
 */
struct BisAutomataStruct {
  int Nvertex{};                        ///< Number of vertices.
  int Nbehavs{};                        ///< Number of behaviors.
  VectorBisWrapper<Bis_vElem> Vertex{}; ///< Array of vertices.
};

///@}