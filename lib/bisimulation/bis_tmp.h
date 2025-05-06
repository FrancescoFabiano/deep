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

//STRUCTURE DECLARATION
//index is the type of the implicit pointers to the array
//hence is a number that belongs to the range [-1..numberOfNodes]
//at the moment we simply use integer;
//later it could become log(numberOfNodes) bits
typedef int BIS_indexType;


//#define     BIS_MAX_bhtmp      400
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
typedef struct bhtab_struct bhtab;
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

struct bhtab_struct
{
    int ap; // Azioni Predefinite (set to 0)
    int n; // Number of Behaviors (Agents)
    char **bh; // Name of the Behaviors (Agents)
};

struct automa_struct
{
    int Nvertex;
    int Nbehavs;
    v_elem *Vertex;
    bhtab *behavs;
};

//extern struct graph G[];
//extern struct qPartition Q[];
//extern struct xPartition X[];
// extern adjList_1 *borderEdges[];


//extern BIS_indexType B1[]; //copy of B
//extern BIS_indexType B_1[]; //list to maintain E-1(B) and E-1(S-B)
//extern BIS_indexType splitD[]; //list of split blocks
//extern BIS_indexType b1List, b_1List, dList; //starting pointers of the lists above
//
//
//extern struct graph G[];
//extern struct qPartition Q[];
//extern struct xPartition X[];
//extern adjList_1 *borderEdges[];

class bisimulation
{
private:
    int numberOfNodes = 0;
    //pointer to the complete structure of the graph (G,Q,X)
    BIS_indexType C;

    //pointers to maintain the free space in Q and X
    BIS_indexType freeQBlock, QBlockLimit;
    BIS_indexType freeXBlock;

    int t; //timestamp


    BIS_indexType maxRank;
    BIS_indexType rankPartition;



    BIS_indexType *B1; //copy of B
    BIS_indexType *B_1; //list to maintain E-1(B) and E-1(S-B)
    BIS_indexType *splitD; //list of split blocks
    BIS_indexType b1List, b_1List, dList; //starting pointers of the lists above


    graph* G;
    qPartition* Q;
    xPartition* X;
    adjList_1 **borderEdges;

    //borderEdges[i] stores the edges going to i from nodes of different rank





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

    /* Our implementation */

    /*Return true if bisimulation has been executed, false otherwise*/
    bool MinimizeAutomaPT(automa *A);
    bool MinimizeAutomaFB(automa *A);
    bisimulation();
    ~bisimulation();
};