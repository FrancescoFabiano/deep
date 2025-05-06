/**
 * \brief Class used to store compact version of the code implemented following [An Efficient Algorithm for Computing Bisimulation Equivalence (Dovier, Piazza, Policriti)].
 *
 * This class collects all the code from the various original classes (bisimulation.zip) in one file to easy their usage.
 * We also applied some conversion:
 * - All the defined constant are preceded by "BIS_";
 * - Some of the "extern" elements are transformed into class fields.
 *
 * For detailed information on this code see bisimlation.zip and the papers
 * [An Efficient Algorithm for Computing Bisimulation Equivalence (Dovier, Piazza, Policriti), Three Partition Refinement Algorithms (Paige-Tarjan)]
 *
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date January 30, 2020, 10:34 AM
 */

#pragma once
#include <iostream>
#include <stdlib.h>
#include <cstring>

#include "../../include/domain/domain.h"
#include "../../include/utilities/define.h"
#include "../../include/states/kripke/kworld.h"
#include "../../include/states/kripke/kstate.h"

//STRUCTURE DECLARATION
//index is the type of the implicit pointers to the array
//hence is a number that belongs to the range [-1..numberOfNodes]
//at the moment we simply use integer;
//later it could become log(numberOfNodes) bits

#define     BIS_MAX_bhtmp      400
#define     BIS_USED             1
#define     BIS_NOTUSED          0
#define     BIS_DELETED         -1
#define     BIS_DA_CANCELLARE   -2

//CONSTANT DECLARATION
//max lenght of the array
const int BIS_MAXINDEX = 100000;
//DFS constants
const int BIS_WHITE = 0;
const int BIS_GRAY = 1;
const int BIS_BLACK = 2;

const int BIS_NIL = -1;

//extern struct graph G[];
//extern struct qPartition Q[];
//extern struct xPartition X[];
// extern adjList_1 *borderEdges[];

class bisimulation
{
private:
    int numberOfNodes;
    //pointer to the complete structure of the graph (G,Q,X)
    BIS_indexType C;

    //pointers to maintain the free space in Q and X
    BIS_indexType freeQBlock, QBlockLimit;
    BIS_indexType freeXBlock;

    struct graph* G;
    struct qPartition* Q;
    struct xPartition* X;


    BIS_indexType maxRank;
    BIS_indexType rankPartition;

    adjList_1 *borderEdges[BIS_MAXINDEX];
    int t; //timestamp
    //borderEdges[i] stores the edges going to i from nodes of different rank


    BIS_indexType B1[BIS_MAXINDEX]; //copy of B
    BIS_indexType B_1[BIS_MAXINDEX]; //list to maintain E-1(B) and E-1(S-B)
    BIS_indexType splitD[BIS_MAXINDEX]; //list of split blocks
    BIS_indexType b1List, b_1List, dList; //starting pointers of the lists above

    //  std::map<kworld_ptr, int> m_index_map; // From kworld to int
    std::vector<kworld_ptr> m_kworld_vec; // Vector of all kworld_ptr
    std::map<int, int> m_compact_indices;

    //from  PaigeTarjan
    void PaigeTarjan();
    int InitPaigeTarjan();
    //from  PaigeTarjanBonic
    //bool MultipleNodes(BIS_indexType);
    // void PaigeTarjanBonic();
    //from FastBisimulation
    // void Inverse();
    //void Rank();
    //void FastBisimulationAlgorithm();
    void DisposeGraph();
    //int InitFBA();



    int InitFBA();

    //compute Paige and Tarjan modified for the fast bisimulation algorithm.
    //It analysed only the nodes of Rank rank that are in the Xblock C.
    void PaigeTarjan(BIS_indexType rank);

    //split computes a single split as regards the single block B.
    /*The function consists of step 3 and step 4 of PaigeTarjan, even if there are
    some differences: once used for the split, B is not anymore necessary; so B1, S1
    and the counters are not computed; since we are interested in the edges between
    nodes of different rank we scan borderEdges[] instead of G[].adj_1*/
    void Split(BIS_indexType B);
    /* modified strongly connected component;
   the first visit is for G-1 and the second for G;
   Q[].prevBlock represents the color of the nodes during the DFS visit,
   Q[].superBlock represents the forefathers in the SCC,
   Q[].firstNode represents the finishing time of the first DFS visit in SCC()*/
    void Rank();
    void FirstDFS_visit(BIS_indexType i);
    void SecondDFS_visit(BIS_indexType i, BIS_indexType ff);

    //compute FastBisimulationAlgorithms
    void FastBisimulationAlgorithm();

    /***********Automata Related**************/
    void FillStructures(automa *A);



    void CreateG(int num_v, v_elem *Gtemp);
    void SetPointers(int n);

    void GetMinimizedAutoma(automa *A);
    void MarkDeletedNodes();
    void DeleteNodes(automa *A);

    /***********Not Sure*************/
    void Inverse();

public:
    /* Rilascia la memoria allocata per l'automa */
    void DisposeAutoma(automa *A);
    void VisAutoma(automa *A);


    /* Our implementation */

    /*Return true if bisimulation has been executed, false otherwise*/
    bool MinimizeAutomaPT(automa *A);
    bool MinimizeAutomaFB(automa *A);

    automa* merge_kstate_to_automaton(const kstate & ks1, const kstate & ks2, int & root2) const;
    //automa* merge_automata(const kstate & ks1, const kstate & ks2, int & root2);
    bool compare_automata(const kstate & ks1, const kstate & ks2);
    bool compare_automata_eq(const kstate & ks1, const kstate & ks2);

    bisimulation();

    //DEBUG
    const automa compare_automata_debug(const kstate & ks1, const kstate & ks2, std::vector<kworld_ptr> & kworld_vec);
    automa* merge_kstate_to_automaton_debug(const kstate & ks1, const kstate & ks2, std::vector<kworld_ptr> & kworld_vec) const;


};