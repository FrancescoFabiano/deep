/**
 * \class Bisimulation
 * \brief Stores a compact version of the Bisimulation algorithm following
 * Dovier, Piazza, Policriti.
 *
 * \details This class collects all the code from the various original classes
 * (Bisimulation.zip) in one file to ease their usage. We also applied some
 * conversion:
 * - All the defined constants are preceded by "Bis";
 * - Some of the "extern" elements are transformed into class fields.
 *
 * For detailed information on this code see
 * utils/resources/original_code_bisimulation.zip and the papers: [An Efficient
 * Algorithm for Computing Bisimulation Equivalence (Dovier, Piazza, Policriti),
 * Three Partition Refinement Algorithms (Paige-Tarjan)]
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 19, 2025
 */

#pragma once

#include <map>

#include "BisimulationVectorStorage.h"
#include "VectorBisWrapper.h"
#include "utilities/Define.h"

/**
 * \brief Main class for Bisimulation minimization.
 */
class Bisimulation {
public:
  /// \name Constructors & Destructor
  ///@{

  /**
   * \brief Default constructor.
   */
  Bisimulation();

  /**
   * \brief Destructor.
   */
  ~Bisimulation() = default;

  /**
   * \brief Deleted copy constructor.
   */
  Bisimulation(const Bisimulation &) = delete;

  /**
   * \brief Deleted copy assignment.
   */
  Bisimulation &operator=(const Bisimulation &) = delete;

  /**
   * \brief Deleted move constructor.
   */
  Bisimulation(Bisimulation &&) = delete;

  /**
   * \brief Deleted move assignment.
   */
  Bisimulation &operator=(Bisimulation &&) = delete;
  ///@}

  /// \name Main API
  ///@{
  /**
   * \brief Visualizes the automaton.
   * \param[in] A The automaton to visualize.
   */
  static void print(const BisAutomata *A);

  /**
   * @brief Function that minimizes the KripkeState using bisimulation.
   * @param[out] kstate The input Kripke state to minimize.
   */
  void calc_min_bisimilar(KripkeState &kstate);
  ///@}

private:
  // --- Data members ---
  int numberOfNodes = 0;
  BisIndexType C = 0;

  BisIndexType freeQBlock = 0, QBlockLimit = 0;
  BisIndexType freeXBlock = 0;

  int t = 0; // timestamp

  BisIndexType maxRank = BIS_NIL;
  BisIndexType rankPartition = BIS_NIL;

  BisIndexType b1List = 0, b_1List = 0, dList = 0;

  std::map<int, int> m_compact_indices;

  VectorBisWrapper<BisGraph> G{BisPreAllocatedIndex};
  // VectorBisWrapper<Bis_qPartition> Q{BisPreAllocatedIndex};
  // VectorBisWrapper<Bis_xPartition> X{BisPreAllocatedIndex};
  // VectorBisWrapper<BisIndexType> B1{BisPreAllocatedIndex};
  // VectorBisWrapper<BisIndexType> B_1{BisPreAllocatedIndex};
  // VectorBisWrapper<BisIndexType> splitD{BisPreAllocatedIndex};
  // VectorBisWrapper<std::shared_ptr<BisAdjList_1>> borderEdges{};

  // VectorBisWrapper<BisGraph>& G =
  // BisimulationVectorStorage::get_instance().G;
  VectorBisWrapper<Bis_qPartition> &Q =
      BisimulationVectorStorage::get_instance().Q;
  VectorBisWrapper<Bis_xPartition> &X =
      BisimulationVectorStorage::get_instance().X;
  VectorBisWrapper<BisIndexType> &B1 =
      BisimulationVectorStorage::get_instance().B1;
  VectorBisWrapper<BisIndexType> &B_1 =
      BisimulationVectorStorage::get_instance().B_1;
  VectorBisWrapper<BisIndexType> &splitD =
      BisimulationVectorStorage::get_instance().splitD;
  VectorBisWrapper<std::shared_ptr<BisAdjList_1>> &borderEdges =
      BisimulationVectorStorage::get_instance().borderEdges;

  // --- Internal methods ---

  /**
   * \brief Runs the Paige-Tarjan partition refinement algorithm.
   */
  void PaigeTarjan();

  /**
   * \brief Initializes the Paige-Tarjan algorithm.
   * \return 0 on success, nonzero on failure.
   */
  int InitPaigeTarjan();

  /**
   * \brief Initializes the Fast Bisimulation Algorithm.
   * \return 0 on success, nonzero on failure.
   */
  int InitFBA();

  /**
   * \brief Runs the Paige-Tarjan algorithm for a specific rank.
   * \param[in] rank The rank to process.
   */
  void PaigeTarjan(BisIndexType rank);

  /**
   * \brief Splits a block during partition refinement.
   * \param[in] B The block to split.
   */
  void Split(BisIndexType B);

  /**
   * \brief Computes the rank of nodes for fast bisimulation.
   */
  void Rank();

  /**
   * \brief First DFS visit for fast bisimulation.
   * \param[in] i The node index.
   */
  void FirstDFS_visit(BisIndexType i);

  /**
   * \brief Second DFS visit for fast bisimulation.
   * \param[in] i The node index.
   * \param[in] ff The parent node index.
   */
  void SecondDFS_visit(BisIndexType i, BisIndexType ff);

  /**
   * \brief Runs the Fast Bisimulation Algorithm.
   */
  void FastBisimulationAlgorithm();

  /**
   * \brief Transforms an automaton with only labeled edges into one with only
   * labeled states and initializes X and Q. \param[in] A The automaton to
   * convert.
   */
  void FillStructures(const BisAutomata &A);

  void CreateG(int num_v, const VectorBisWrapper<Bis_vElem> &G_temp);


  void SetPointers(int n);


  void GetMinimizedAutoma(BisAutomata &A);

  /**
   * \brief Marks nodes to be deleted.
   */
  void MarkDeletedNodes();

  void DeleteNodes(BisAutomata &A) const;

  /**
   * \brief Constructs the inverse of the BisGraph represented by the adjacency
   * list.
   */
  void Inverse();

  /**
   * \brief Returns true if Bisimulation has been executed, false otherwise.
   * \param[in] A The automaton to minimize.
   * \return True if minimization was successful.
   */
  [[nodiscard]] bool MinimizeAutomaPT(BisAutomata &A);

  /**
   * \brief Returns true if fast Bisimulation has been executed, false
   * otherwise.
   * \param[in] A The automaton to minimize.
   * \return True if
   * minimization was successful.
   */
  [[nodiscard]] bool MinimizeAutomaFB(BisAutomata &A);

  /**
   * @brief Converts a KripkeState into a Bisimulation Automaton (BisAutomata).
   *
   * This function takes a KripkeState and translates it into a bisimulation
   * automaton, assigning each world a compact index and labeling transitions
   * based on agent beliefs.
   *
   * @param pworld_vec Output vector that stores the ordered list of Kripke
   * world pointers used in the automaton.
   * @param agent_to_label A mapping from each Agent to a corresponding
   * BisLabel.
   * @param kstate The input Kripke state from which the automaton is
   * constructed.
   * @return BisAutomata The resulting bisimulation automaton.
   */
  static BisAutomata
  kstate_to_automaton(VectorBisWrapper<KripkeWorldPointer> &pworld_vec,
                      const std::map<Agent, BisLabel> &agent_to_label,
                      const KripkeState &kstate);

  /**
   * @brief Converts a Bisimulation Automaton (BisAutomata) into a KripkeState.

   *
   * @param a The automaton to convert (usually minimized).
   * @param world_vec A vector containing the worlds to insert into the state.
   * @param label_to_agent Conversion to transform the edges of the automaton
   into edges labelled with agents.
   * @param[out] kstate The input Kripke state in the automaton is constructed.
   */
  static void
  automaton_to_kstate(const BisAutomata &a,
                      const VectorBisWrapper<KripkeWorldPointer> &world_vec,
                      const std::map<BisLabel, Agent> &label_to_agent,
                      KripkeState &kstate);
};
