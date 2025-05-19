/**
 * \class Bisimulation
 * \brief Stores a compact version of the Bisimulation algorithm following Dovier, Piazza, Policriti.
 *
 * \details This class collects all the code from the various original classes (Bisimulation.zip) in one file to ease their usage.
 * We also applied some conversion:
 * - All the defined constants are preceded by "Bis";
 * - Some of the "extern" elements are transformed into class fields.
 *
 * For detailed information on this code see resources/original_code_bisimulation.zip and the papers:
 * [An Efficient Algorithm for Computing Bisimulation Equivalence (Dovier, Piazza, Policriti), Three Partition Refinement Algorithms (Paige-Tarjan)]
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 19, 2025
 */

#pragma once

#include <map>
#include "utilities/Define.h"

/**
 * \brief Main class for Bisimulation minimization.
 */
class Bisimulation
{
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
    Bisimulation(const Bisimulation&) = delete;

    /**
     * \brief Deleted copy assignment.
     */
    Bisimulation& operator=(const Bisimulation&) = delete;

    /**
     * \brief Deleted move constructor.
     */
    Bisimulation(Bisimulation&&) = delete;

    /**
     * \brief Deleted move assignment.
     */
    Bisimulation& operator=(Bisimulation&&) = delete;
    ///@}

    /// \name Main API
    ///@{
    /**
     * \brief Visualizes the automaton.
     * \param[in] A The automaton to visualize.
     * \param[in] os The output stream to print the automaton.
     */
    static void print(const BisAutomata* A, std::ostream& os);

    /**
     * \brief Returns true if Bisimulation has been executed, false otherwise.
     * \param[in] A The automaton to minimize.
     * \return True if minimization was successful.
     */
    [[nodiscard]] bool MinimizeAutomaPT(BisAutomata& A);

    /**
     * \brief Returns true if fast Bisimulation has been executed, false otherwise.
     * \param[in] A The automaton to minimize.
     * \return True if minimization was successful.
     */
    [[nodiscard]] bool MinimizeAutomaFB(BisAutomata& A);
    ///@}

private:
    // --- Data members ---
    int numberOfNodes = 0;
    BisIndexType C = 0;

    BisIndexType freeQBlock = 0, QBlockLimit = 0;
    BisIndexType freeXBlock = 0;

    std::vector<BisGraph> G {BisPreAllocatedIndex};
    std::vector<Bis_qPartition> Q {BisPreAllocatedIndex};
    std::vector<Bis_xPartition> X {BisPreAllocatedIndex};

    std::vector<std::shared_ptr<BisAdjList_1>> borderEdges{};
    int t = 0; // timestamp

    BisIndexType maxRank = BIS_NIL;
    BisIndexType rankPartition = BIS_NIL;


    std::vector<BisIndexType> B1{BisPreAllocatedIndex};
    std::vector<BisIndexType> B_1{BisPreAllocatedIndex};
    std::vector<BisIndexType> splitD{BisPreAllocatedIndex};
    BisIndexType b1List = 0, b_1List = 0, dList = 0;

    std::map<int, int> m_compact_indices;

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
     * \brief Transforms an automaton with only labeled edges into one with only labeled states and initializes X and Q.
     * \param[in] A The automaton to convert.
     */
    void FillStructures(const BisAutomata& A);

    /**
     * \brief Converts the BisGraph with only labeled edges into one with only labeled states.
     * \param[in] num_v The number of vertices.
     * \param[in] G_temp The BisGraph with only labeled edges.
     */
    void CreateG(int num_v, const std::vector<Bis_vElem>& G_temp);

    /**
     * \brief Manages the pointers between the array X and the array G.
     * \param[in] n The total number of labels.
     */
    void SetPointers(int n);

    /**
     * \brief Updates the automaton with the minimized structure.
     * \param[in] A The automaton to update.
     */
    void GetMinimizedAutoma(BisAutomata & A);

    /**
     * \brief Marks nodes to be deleted.
     */
    void MarkDeletedNodes();

    /**
     * \brief Deletes nodes from the automaton.
     * \param[in] A The automaton to update.
     */
    void DeleteNodes(BisAutomata& A) const;

    /**
     * \brief Constructs the inverse of the BisGraph represented by the adjacency list.
     */
    void Inverse();
};
