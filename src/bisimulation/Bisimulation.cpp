#include "Bisimulation.h"

#include "ArgumentParser.h"
#include "Configuration.h"
#include "Domain.h"
#include "KripkeState.h"

/***IO_FC2.cpp****/
void Bisimulation::FillStructures(const BisAutomata &A) {
    X[0].prevXBlock = BIS_NIL;
    X[0].nextXBlock = 1;
    X[0].firstBlock = BIS_NIL;
    X[A.Nbehavs - 1].prevXBlock = A.Nbehavs - 2;
    X[A.Nbehavs - 1].nextXBlock = BIS_NIL;
    X[A.Nbehavs - 1].firstBlock = BIS_NIL;

    for (BisIndexType i = 1; i < (A.Nbehavs - 1); i++) {
        X[i].nextXBlock = i + 1;
        X[i].prevXBlock = i - 1;
        X[i].firstBlock = BIS_NIL;
    }

    CreateG(A.Nvertex, A.Vertex);
    SetPointers(A.Nbehavs);
}

/**
 * \brief Converts a graph with only labeled edges into one with only labeled states.
 *
 * This function performs the actual conversion of the input graph (with only labeled edges)
 * into a graph with only labeled states. It also determines, for each state, which block it belongs to.
 * A block is an element of the array X, which uniquely represents a label of the graph.
 * Initially, all states with the same label will belong to the same block.
 * The array X has a size equal to the total number of labels.
 *
 * \param[in] num_v The number of vertices in the input graph G_temp.
 * \param[in] G_temp The graph with only labeled edges.
 *
 * \note The output is not returned since the array G is global.
*/
void Bisimulation::CreateG(const int num_v, const VectorBisWrapper<Bis_vElem> &G_temp) {
    BisIndexType v;

    // Create "num_v" vertices in G. These are the states also present in the input graph G_temp
    // (with only labeled edges) and, by implementation convention, will all have label 0
    // (in reality, their label could take any value). All these states will therefore belong,
    // as indicated by "G[v].block = 0", to block 0, i.e., to the element of the array X with index 0.
    for (v = 0; v < num_v; v++) {
        G[v].block = 0;
        G[v].label = 0;
    }

    // The variable "numberOfNodes" indicates the total number of states in the graph on
    // which the PaigeTarjan and FastBisimulation algorithms will be executed. At this point,
    // "num_v" states have been created, equal to the total number of states in the graph
    // with only labeled edges; its initial value will therefore be "num_v" and it will grow
    // as the edges are unlabeled.
    numberOfNodes = num_v;

    // I create all the other states (by un-labelling the edges). For each state of the original
    // graph G_temp, I scan its adjacency list. The possible cases are:
    // 1. an edge with a single label
    // 2. an edge with multiple labels
    // In the first case, I "split" the edge in two and insert between them a state with a
    // label equal to that of the "split" edge. In the second case, assuming the edge has "n"
    // labels, I will create "n" states, each of which will have one of the labels from the edge.
    // These states will maintain the order in which the labels are stored on the edge, meaning that
    // if label "i" preceded label "j" on the edge, there will be an edge from the state with
    // label "i" to the one with label "j". Obviously, the first of the "n" states will have
    // an incoming edge from the state "v" we're considering, while the last of the "n" states
    // will have an outgoing edge to the state the original edge was pointing to.
    for (v = 0; v < num_v; v++) {
        // Pointer to the adjacency list of the current state, which will be created by the following
        // lines of code
        std::shared_ptr<BisAdjList> *curr_adj = &(G[v].adj);

        // Loop that processes all outgoing edges from "v"
        for (BisIndexType e = 0; e < G_temp[v].ne; e++) {
            // I create a new state (the first in the chain of labels). As mentioned earlier,
            // the "block" field is initialized with the index of the element in X that represents
            // the label of the state being created
            G[numberOfNodes].block = G_temp[v].e[e].bh[0];
            G[numberOfNodes].label = G_temp[v].e[e].bh[0];
            numberOfNodes++;

            // Update of the adjacency list of v. A new element is added to the adjacency list of v,
            // that is, a new edge reaching the newly created state
            *curr_adj = std::make_shared<BisAdjList>();
            (*curr_adj)->node = numberOfNodes - 1;
            (*curr_adj)->next = nullptr;
            curr_adj = &((*curr_adj)->next);

            // I create the states from the second onward in the chain of labels. This loop is
            // executed only if the edge we're considering has more than one label
            for (BisIndexType b = 1; b < G_temp[v].e[e].nbh; b++) {
                // I create a new state
                G[numberOfNodes].block = G_temp[v].e[e].bh[b];
                G[numberOfNodes].label = G_temp[v].e[e].bh[b];

                // I update the adjacency list of the previously created state, i.e., I add
                // an edge from the second-to-last state to the one just created
                G[numberOfNodes - 1].adj = std::make_shared<BisAdjList>();
                G[numberOfNodes - 1].adj->node = numberOfNodes;
                G[numberOfNodes - 1].adj->next = nullptr;

                numberOfNodes++;
            }

            // Update of the adjacency list of the last state we created
            G[numberOfNodes - 1].adj = std::make_shared<BisAdjList>();
            G[numberOfNodes - 1].adj->node = G_temp[v].e[e].tv;
            G[numberOfNodes - 1].adj->next = nullptr;
        }
    }
}

/**
*    This function manages the pointers between the X array and the G array as follows:
*    It was previously stated that the X array has a size equal to the total number of labels
*    (behaviors) present in the graph.
*
*    Each element of this array therefore represents a class of states — that is, all the states
*    that share the same label (i.e., all states with the same label will have the same "block" field).
*
*    Moreover, each element of the X array holds a representative state (indicated by the "firstBlock"
*    field of X), and each state will have two "pointers" to the previous and next elements in the block.
*
*    As a result, given any element (block) in X, by looking at its "firstBlock" and following the
*    previous and next pointers, it's possible to identify all the states belonging to that block.
*
*   \param[in] n the total number of distinct labels
*/
void Bisimulation::SetPointers(const int n) {
    // Allocate temporary array used in the following loop
    VectorBisWrapper<BisIndexType> lastNodeInBlock(n, BIS_NIL);

    for (BisIndexType i = 0; i < numberOfNodes; i++) {
        // Retrieve the block to which state "i" belongs
        BisIndexType block = G[i].block;

        // If this block does not yet have a representative
        // (i.e., the "firstBlock" field is still NIL),
        // then state "i" becomes its representative.
        // That is, the first state found with label "k"
        // becomes the representative of the block for label "k".

        if (X[block].firstBlock == BIS_NIL) {
            X[block].firstBlock = i;
            G[i].prevInBlock = BIS_NIL;
            G[i].nextInBlock = BIS_NIL;
        }
        // Otherwise, the block "x" already has a representative.
        // So we link state "i" to the last state we found with the same label,
        // using the "prevInBlock" and "nextInBlock" pointers.
        // The "curr_node" array is used to keep track of the last state seen
        // for each label (hence the size of curr_node is equal to the total number of labels).
        else {
            const BisIndexType prev = lastNodeInBlock[block];
            G[i].prevInBlock = prev;
            G[i].nextInBlock = BIS_NIL;
            G[prev].nextInBlock = i;
        }

        // Update the temporary array
        lastNodeInBlock[block] = i;
    }
}

void Bisimulation::Inverse() {
    for (BisIndexType i = 0; i < numberOfNodes; i++) {
        auto adj = G[i].adj;
        while (adj != nullptr) {
            adj->countxS = nullptr;
            const auto a = std::make_shared<BisAdjList_1>();
            a->node = i;
            a->next = G[adj->node].adj_1;
            G[adj->node].adj_1 = a;
            a->adj = adj;
            adj = adj->next;
        }
    }
}

void Bisimulation::print(const BisAutomata *A) {
    // Extract basic automaton information
    int numVertices = A->Nvertex;
    int numBehaviors = A->Nbehavs;
    const VectorBisWrapper<Bis_vElem> &vertices = A->Vertex;

    auto &os = ArgumentParser::get_instance().get_output_stream();
    // Output summary of the automaton
    os << "Number of Vertices: " << numVertices << "\n"
            << "Number of Behaviors: " << numBehaviors << "\n\n";

    os << "Vertices and their outgoing edges:\n";

    // Iterate through all vertices
    for (int i = 0; i < numVertices; ++i) {
        const auto &vertex = vertices[i];
        os << "\nVertex[" << i << "] has " << vertex.ne << " edges:\n";

        // Iterate through each edge of the vertex
        for (int j = 0; j < vertex.ne; ++j) {
            const auto &edge = vertex.e[j];

            // Print all labels (behaviors) on the edge
            for (int k = 0; k < edge.nbh; ++k) {
                os << edge.bh[k];
                if (k < edge.nbh - 1) os << ".";
            }

            // Output destination vertex
            os << " -> " << edge.tv << "\n";
        }
    }
}

Bisimulation::Bisimulation() {
    VectorBisWrapper<BisGraph> G{BisPreAllocatedIndex};
    VectorBisWrapper<Bis_qPartition> Q{BisPreAllocatedIndex};
    VectorBisWrapper<Bis_xPartition> X{BisPreAllocatedIndex};
}


/**
 * @brief Builds the minimized automaton with labeled edges.
 *
 * This function constructs the minimized automaton (with only labeled edges)
 * from the minimized graph (which only has labeled states and is stored in `G`)
 * and from the original input automaton `A` (which only has labeled edges).
 *
 * @param[in] A The input automaton with only labeled edges.
 *
 * @note The Paige-Tarjan and FastBisimulation algorithms do not actually remove
 *       the eliminated nodes. This is handled later by `MarkDeletedNodes` and
 *       `DeleteNodes`.
 */
void Bisimulation::GetMinimizedAutoma(BisAutomata &A) {
    // Marks states to be deleted that were eliminated during minimization.
    MarkDeletedNodes();

    // Removes nodes that were marked for deletion.
    DeleteNodes(A);
}

/**
 * @brief Marks states to be deleted after minimization.
 *
 * This function marks the states that should be deleted by evaluating the values
 * of some fields in the arrays `G` and `Q` after the PaigeTarjan and FastBisimulation
 * algorithms have been executed.
 *
 * The actual removal of these states is performed by the functions `DeleteNodes` and `SaveToFC2`.
 *
 * @note The minimization algorithms implemented here modify (destroy) the arrays `Q` and `X`.
 *       Therefore, the only way to determine bisimilar states is by checking the `block`
 *       field of each state, which indicates the block it belongs to.
 *
 */
void Bisimulation::MarkDeletedNodes() {
    // Variables
    BisIndexType i;

    // After minimization, some fields in Q are reused for different purposes.
    // Here, the `size` field tracks whether a block in Q has a representative state.
    // Initialize `size` to BisNotUsed to indicate no block has a representative yet.
    for (i = 0; i < QBlockLimit; i++) // QBlockLimit might be larger than numberOfNodes
        Q[i].size = BisNotUsed;

    // Mark states that should be deleted
    for (i = 0; i < numberOfNodes; i++) {
        // Get the block to which state ''i'' belongs

        // If the block has no representative yet, mark this state as the representative
        if (BisIndexType q = G[i].block; Q[q].size == BisNotUsed) {
            Q[q].size = BisUsed;
            Q[q].firstNode = i;
        }
        // Otherwise, mark the state for deletion because it's bisimilar to the block's representative.
        // Use `nextInBlock` field as it will no longer be used.
        else {
            G[i].nextInBlock = BisToDelete;
        }
    }
}

/**
 * @brief Updates the automaton A to mark states for deletion and redirects edges.
 *
 * This function identifies which states in the input automaton A should be deleted based on the
 * `nextInBlock` field of the `G` array, which was set by `MarkDeletedNodes`.
 *
 * States marked for deletion will have their outgoing edge count (`ne`) set to `BIS_DELETED`.
 * For states to be kept, any outgoing edges pointing to deleted states are redirected to the
 * representative state of that deleted state's block.
 *
 * The modified automaton A is then ready to be saved by the `SaveToFC2` function, which will
 * write only the states and edges that remain.
 *
 * @param A Pointer to the input automaton with labeled edges.
 */
void Bisimulation::DeleteNodes(BisAutomata &A) const {
    // Iterate over all vertices in the automaton
    for (BisIndexType i = 0; i < A.Nvertex; ++i) {
        // If this state is marked for deletion
        if (G[i].nextInBlock == BisToDelete) {
            // Mark state as deleted by setting number of edges to BIS_DELETED
            A.Vertex[i].ne = BisDeleted;
        } else {
            // For states that remain, update edges pointing to deleted states
            for (int j = 0; j < A.Vertex[i].ne; ++j) {
                auto &edge = A.Vertex[i].e[j];
                if (G[edge.tv].nextInBlock == BisToDelete) {
                    // Redirect edge to the representative of the deleted state's block
                    edge.tv = Q[G[edge.tv].block].firstNode;
                }
            }
        }
    }
}


/*\***paigeTarjan.cpp****/
//initialise Paige and Tarjan
int Bisimulation::InitPaigeTarjan() {
    BisIndexType i, end = 0, temp;

    //initialisation of the graph (G,Q,X)
    for (BisIndexType l = 0; l != BIS_NIL; l = temp) {
        //for each label block
        temp = X[l].nextXBlock;
        if (temp == BIS_NIL)
            end = l;
        Q[l].prevBlock = X[l].prevXBlock;
        Q[l].nextBlock = X[l].nextXBlock;
        Q[l].firstNode = X[l].firstBlock;
        Q[l].superBlock = 0;
        //compute Q[].size
        Q[l].size = 0;
        for (i = X[l].firstBlock; i != BIS_NIL; i = G[i].nextInBlock) //for each node
            (Q[l].size)++;
        X[l].prevXBlock = BIS_NIL;
        X[l].firstBlock = BIS_NIL;
        X[l].nextXBlock = l + 1;

        B1[l] = numberOfNodes;
        B_1[l] = numberOfNodes;
        B_1[l] = numberOfNodes;
        splitD[l] = numberOfNodes;
    }

    X[0].nextXBlock = BIS_NIL;
    X[0].prevXBlock = BIS_NIL;
    X[0].firstBlock = 0;

    if (end == numberOfNodes)
        freeQBlock = BIS_NIL;
    else
        freeQBlock = end + 1;
    QBlockLimit = numberOfNodes;
    freeXBlock = 1;

    for (i = end + 1; i < numberOfNodes; i++) {
        Q[i].size = 0;
        Q[i].nextBlock = i + 1;
        Q[i].superBlock = BIS_NIL;
        Q[i].prevBlock = BIS_NIL;
        Q[i].firstNode = BIS_NIL;

        X[i].nextXBlock = i + 1;
        X[i].prevXBlock = BIS_NIL;
        X[i].firstBlock = BIS_NIL;

        B1[i] = numberOfNodes;
        B_1[i] = numberOfNodes;
        B_1[i] = numberOfNodes;
        splitD[i] = numberOfNodes;
    }
    Q[numberOfNodes - 1].nextBlock = BIS_NIL;
    X[numberOfNodes - 1].nextXBlock = BIS_NIL;


    if (Q[0].nextBlock == BIS_NIL) //P&T not necessary
        return 1;

    C = 0;
    //initialisation of the counters
    //initially there is a count per node count(x,U)=|E({x})|
    for (i = 0; i < numberOfNodes; i++) {
        auto adj = G[i].adj;
        //to avoid the creation of a BisCounter set to zero
        if (adj == nullptr)
            continue;
        const auto cxS = std::make_shared<BisCounter>();
        cxS->value = 0;
        while (adj != nullptr) {
            (cxS->value)++;
            /*each edge xEy contains a pointer to count(x,U);
            remember that each edge y(E-1)x contains a pointer to the edge xEy!*/
            adj->countxS = cxS;
            adj = adj->next;
        }
    }
    return 0;
}


//compute Paige and Tarjan
void Bisimulation::PaigeTarjan() {
    //pointer to the X-Blocks S and S1
    BisIndexType B, S_B; //pointer to the Q-Blocks B and S-B
    BisIndexType oldD, newD; //old and new block of x belonging to E-1(B)
    BisIndexType x, e;
    std::shared_ptr<BisAdjList_1> adj = nullptr;
    std::shared_ptr<BisCounter> cxS = nullptr;

    while (C != BIS_NIL) {
        /*Step 1(select a refining block) & Step 2(update X)*/
        //select some block S from C
        const BisIndexType S = C;
        /*if S has more than two blocks, it has to be put back to C;
        hence it is not removed from X until we are sure it is not still
        compound after removing B from it*/

        /*examine the first two blocks in the of blocks of Q contained in S;
        let B be the smaller, remove B from S*/
        if (Q[X[S].firstBlock].size < Q[Q[X[S].firstBlock].nextBlock].size) {
            B = X[S].firstBlock;
            S_B = Q[X[S].firstBlock].nextBlock;
            X[S].firstBlock = S_B;
            Q[B].nextBlock = BIS_NIL;
            Q[S_B].prevBlock = BIS_NIL;
        } else {
            B = Q[X[S].firstBlock].nextBlock;
            S_B = X[S].firstBlock;
            Q[S_B].nextBlock = Q[B].nextBlock;
            if (Q[S_B].nextBlock != BIS_NIL)
                Q[Q[S_B].nextBlock].prevBlock = S_B;
            Q[B].nextBlock = BIS_NIL;
            Q[B].prevBlock = BIS_NIL;
        }

        //and create a new simple block S1 of X containing B as its only block of Q
        const BisIndexType S1 = freeXBlock;
        freeXBlock = X[freeXBlock].nextXBlock;
        Q[B].superBlock = S1;
        X[S1].nextXBlock = BIS_NIL;
        //X[S1].prevXBlock = BIS_NIL;
        X[S1].firstBlock = B;
        //X[S1].countxS is initialised in step 3

        //check if S is still compound
        if (Q[S_B].nextBlock == BIS_NIL) {
            //not compound: remove S from C
            C = X[C].nextXBlock;
            if (C != BIS_NIL)
                X[C].prevXBlock = BIS_NIL;
            X[S].nextXBlock = BIS_NIL;
            //free the space of S as XBlock
            /*WE DO NOT FREE THE BLOCK S: the XBlock still exists, but it is not in the chain of C*/
        }

        /*Step 3(compute E-1(B))*/
        /*by scanning the edges xEy such that y belongs to B
        and adding each element x in such an edge to E-1(B),
        if it has not already been added.
        Duplicates are suppressed by marking elements: B_1
        Side effect: copy the elements of B in B1
        During the same scan, compute count(x,B)=count(x,S1) because S1={B};
        create a new BisCounter record and make G[x].countxB point to it*/
        BisIndexType y = b1List = Q[B].firstNode;
        b_1List = BIS_NIL;
        while (y != BIS_NIL) {
            //for each y belonging to B
            B1[y] = G[y].nextInBlock; //copy the elements of B in B1
            adj = G[y].adj_1;
            while (adj != nullptr) {
                //for each node in the adj_1 of y
                x = adj->node;
                if (B_1[x] == numberOfNodes) {
                    //node not already added to E-1(B)
                    B_1[x] = b_1List;
                    b_1List = x;
                    //create a new BisCounter: it is pointed by G[x].countxB    /*1*/
                    cxS = std::make_shared<BisCounter>();
                    cxS->node = x;
                    cxS->value = 1;
                    G[x].countxB = cxS; /*1*/
                } else
                    (G[x].countxB->value)++;
                adj = adj->next; //next node in the adj_1 of y
            }
            y = G[y].nextInBlock; //next node y belonging to B
        }

        /*Step 4(refine Q with respect to B)*/
        /*for each block D of Q containing some element of E-1(B)
        split D into D1 = D ^ E-1(B) and D2 = D - D1*/
        dList = BIS_NIL;
        //do this by scanning the elements of E-1(B)
        x = b_1List;
        while (x != BIS_NIL) {
            //for each x belonging to E-1(B)
            //determine the block D of Q containing it
            oldD = G[x].block; //index of D (old block of x)
            //and create an associated block D1 if one does not already exist
            if (splitD[oldD] == numberOfNodes) {
                //block D not already split
                splitD[oldD] = dList;
                dList = oldD;
                //create a new block D1
                if (freeQBlock == BIS_NIL) {
                    freeQBlock = QBlockLimit++;
                    Q[freeQBlock].size = 0;
                    Q[freeQBlock].nextBlock = BIS_NIL;
                    splitD[freeQBlock] = numberOfNodes;
                    //not necessary to initialise
                    //Q[freeQBlock].prevBlock = BIS_NIL;
                    //Q[freeQBlock].superBlock = BIS_NIL;
                    //Q[freeQBlock].firstNode = BIS_NIL;
                }
                newD = freeQBlock; //index of D1 (new block of x)
                freeQBlock = Q[freeQBlock].nextBlock;
                Q[newD].firstNode = BIS_NIL;
                /*insert D1 just after D, so we know that, if D has already been
                split, the associated D1 is the next block*/
                Q[newD].nextBlock = Q[oldD].nextBlock;
                Q[oldD].nextBlock = newD;
                Q[newD].prevBlock = oldD;
                if (Q[newD].nextBlock != BIS_NIL)
                    Q[Q[newD].nextBlock].prevBlock = newD;
                Q[newD].superBlock = Q[oldD].superBlock;
            } else
                newD = Q[oldD].nextBlock;
            //move x from D to D1
            if (G[x].prevInBlock != BIS_NIL)
                G[G[x].prevInBlock].nextInBlock = G[x].nextInBlock;
            else
                Q[G[x].block].firstNode = G[x].nextInBlock;
            if (G[x].nextInBlock != BIS_NIL)
                G[G[x].nextInBlock].prevInBlock = G[x].prevInBlock;
            G[x].block = newD;
            G[x].nextInBlock = Q[newD].firstNode;
            G[x].prevInBlock = BIS_NIL;
            if (Q[newD].firstNode != BIS_NIL)
                G[Q[newD].firstNode].prevInBlock = x;
            Q[newD].firstNode = x;
            (Q[oldD].size)--;
            (Q[newD].size)++;

            y = x;
            x = B_1[x];
            //re-initialisation of B_1
            B_1[y] = numberOfNodes;
        } //endwhile

        //dList points to the list of new blocks splitD
        BisIndexType d = dList;
        while (d != BIS_NIL) {
            if (Q[d].firstNode == BIS_NIL) {
                //D empty: remove it and free its space
                if (Q[d].prevBlock != BIS_NIL)
                    Q[Q[d].prevBlock].nextBlock = Q[d].nextBlock;
                else
                    X[Q[d].superBlock].firstBlock = Q[d].nextBlock;
                //we are sure that after D,there is D1
                Q[Q[d].nextBlock].prevBlock = Q[d].prevBlock;
                //re-initialise Q[d]
                //Q[d].size is already zero
                Q[d].prevBlock = BIS_NIL;
                Q[d].superBlock = BIS_NIL;
                Q[d].firstNode = BIS_NIL;
                //free Q[d]
                Q[d].nextBlock = freeQBlock;
                freeQBlock = d;
            } else {
                /*if D nonempty and the superBlock containing D and D1 has been
                made compound by the split, add this block to C*/
                if (Q[d].prevBlock == BIS_NIL && Q[Q[d].nextBlock].nextBlock == BIS_NIL) {
                    //D and D1 are the only blocks in this just split Xblock
                    X[Q[d].superBlock].nextXBlock = C;
                    X[Q[d].superBlock].prevXBlock = BIS_NIL;
                    C = Q[d].superBlock;
                    /*when D became the only block of an XBlock (see the end of step 2)
                    we did not free its space, and now we can re-use it!!*/
                }
            }

            e = d;
            d = splitD[d];
            //re-initialisation of splitD
            splitD[e] = numberOfNodes;
        }

        /*Step 5(compute E-1(B) - E-1(S_B))*/
        /*Scan each x such that xEy and y belongs to B; determine count(x,B)
        to which G[x].countxB points and count(x,S) to which xEy points
        (y belongs to B -> scan y(E-1)x -> y(E-1)x points to xEy ->
          xEy points to count(x,S))
        To save space we use again the array B_1 to store E-1(B) - E-1(S - B)*/
        y = b1List;
        b_1List = BIS_NIL;
        while (y != BIS_NIL) {
            //for each y belonging to B1
            adj = G[y].adj_1;
            while (adj != nullptr) {
                //for each node in the adj_1 of y -> scan xEy, y in B
                x = adj->node;
                if (G[x].countxB->value == adj->adj->countxS->value)
                    if (B_1[x] == numberOfNodes) {
                        //x is a node not already added to E-1(S - B)
                        B_1[x] = b_1List;
                        b_1List = x;
                    }
                adj = adj->next;
            }
            y = B1[y];
        }

        /*Step 6(refine Q with respect to S_B)*/
        /*proceed exactly as in Step 4, but scan E-1(B) - E_1(S - B)
        For each block D of Q containing some element of E-1(B) - E-1(S - B)
        split D into D1 = D ^ (E-1(B) - E-1(S - B)) and D2 = D - D1*/
        dList = BIS_NIL;
        //do this by scanning the elements of E-1(B) - E-1(S - B)
        x = b_1List;
        while (x != BIS_NIL) {
            //to process an element x belonging to E-1(B) - E-1(S - B)
            //determine the block D of Q containing it
            oldD = G[x].block; //index of D (old block of x)
            //and create an associated block D1 if one does not already exist
            if (splitD[oldD] == numberOfNodes) {
                //block D not already split
                splitD[oldD] = dList;
                dList = oldD;
                //create a new block D1
                if (freeQBlock == BIS_NIL) {
                    freeQBlock = QBlockLimit++;
                    Q[freeQBlock].size = 0;
                    Q[freeQBlock].nextBlock = BIS_NIL;
                    splitD[freeQBlock] = numberOfNodes;
                    //not necessary to initialise
                    //Q[freeQBlock].prevBlock = BIS_NIL;
                    //Q[freeQBlock].superBlock = BIS_NIL;
                    //Q[freeQBlock].firstNode = BIS_NIL;
                }
                newD = freeQBlock; //index of D1 (new block of x)
                freeQBlock = Q[freeQBlock].nextBlock;
                Q[newD].firstNode = BIS_NIL;
                /*insert D1 just after D, so we know that, if D has already
                been split, the associated D1 is the next block*/
                Q[newD].nextBlock = Q[oldD].nextBlock;
                Q[oldD].nextBlock = newD;
                Q[newD].prevBlock = oldD;
                if (Q[newD].nextBlock != BIS_NIL)
                    Q[Q[newD].nextBlock].prevBlock = newD;
                Q[newD].superBlock = Q[oldD].superBlock;
            } else
                newD = Q[oldD].nextBlock;
            //move x from D to D1
            if (G[x].prevInBlock != BIS_NIL)
                G[G[x].prevInBlock].nextInBlock = G[x].nextInBlock;
            else
                Q[G[x].block].firstNode = G[x].nextInBlock;
            if (G[x].nextInBlock != BIS_NIL)
                G[G[x].nextInBlock].prevInBlock = G[x].prevInBlock;
            G[x].block = newD;
            G[x].nextInBlock = Q[newD].firstNode;
            G[x].prevInBlock = BIS_NIL;
            if (Q[newD].firstNode != BIS_NIL)
                G[Q[newD].firstNode].prevInBlock = x;
            Q[newD].firstNode = x;
            (Q[oldD].size)--;
            (Q[newD].size)++;

            y = x;
            x = B_1[x];
            //re-initialisation of B_1
            B_1[y] = numberOfNodes;
        } //endwhile

        //dList points to the list of new blocks splitD
        d = dList;
        while (d != BIS_NIL) {
            if (Q[d].firstNode == BIS_NIL) {
                //D empty: remove it and free its space
                if (Q[d].prevBlock != BIS_NIL)
                    Q[Q[d].prevBlock].nextBlock = Q[d].nextBlock;
                else
                    X[Q[d].superBlock].firstBlock = Q[d].nextBlock;
                Q[Q[d].nextBlock].prevBlock = Q[d].prevBlock;
                //re-initialise Q[d]
                //Q[d].size is already zero
                Q[d].prevBlock = BIS_NIL;
                Q[d].superBlock = BIS_NIL;
                Q[d].firstNode = BIS_NIL;
                //free Q[d]
                Q[d].nextBlock = freeQBlock;
                freeQBlock = d;
            } else {
                /*if D nonempty and the superBlock containing D and D1 has been
                made compound by the split, add this block to C*/
                if (Q[d].prevBlock == BIS_NIL && Q[Q[d].nextBlock].nextBlock == BIS_NIL) {
                    //D and D1 are the only blocks in this just split Xblock
                    X[Q[d].superBlock].nextXBlock = C;
                    X[Q[d].superBlock].prevXBlock = BIS_NIL;
                    C = Q[d].superBlock;
                    /*when D became the only block of an XBlock (see the end of step 2)
                    we did not free its space, and now we can re-use it!!*/
                }
            }
            e = d;
            d = splitD[d];
            //re-initialisation of splitD
            splitD[e] = numberOfNodes;
        }

        /*Step 7(update counts)*/
        /*scan the edges xEy tc y belongs to B1.
        To process an edge decrement count(x,S) (to which xEy points).
        If this count becomes zero delete the count record,
        and make xEy point to count(x,B) (to which x points).
        Discard B1 (re-initialise it).*/
        y = b1List;
        while (y != BIS_NIL) {
            //for each y belonging to B
            adj = G[y].adj_1;
            while (adj != nullptr) {
                //for each node in the adj_1 of y -> scan xEy, y in B
                x = adj->node;
                cxS = adj->adj->countxS;
                if (cxS->value != 1) {
                    (cxS->value)--;
                    adj->adj->countxS = G[x].countxB;
                } else {
                    //count(x,S) becomes zero
                    //make xEy point to count(x,B)
                    adj->adj->countxS = G[x].countxB;
                    //delete count(x,S)
                }
                adj = adj->next;
            }
            x = y;
            y = B1[y];
            //re-initialisation of B1
            B1[x] = numberOfNodes;
        }
    } //end while
}


/*\***fastBisimulation.cpp****/
/* modified strongly connected component;
   the first visit is for G-1 and the second for G;
   Q[].prevBlock represents the color of the nodes during the DFS visit,
   Q[].superBlock represents the forefathers in the SCC,
   Q[].firstNode represents the finishing time of the first DFS visit in SCC()*/
void Bisimulation::Rank() {
    BisIndexType i;

    //initialisation of the nodes
    for (i = 0; i < numberOfNodes; i++) {
        //color is BIS_WHITE
        Q[i].prevBlock = BIS_WHITE;
        //'i' is forefather of itself
        Q[i].superBlock = i;
        //all the nodes are well-founded
        G[i].WFflag = true;
        //to normalize the ranks
        Q[i].size = 0;
    }
    //timestamp is set to zero
    t = 0;

    //first DFS visit
    for (i = 0; i < numberOfNodes; i++)
        if (Q[i].prevBlock == BIS_WHITE)
            FirstDFS_visit(i);

    /*to avoid a second initialisation of the colors of the nodes(Q[].prevBlock)
    the meaning of the colors is inverted: BIS_WHITE <--> BIS_BLACK*/

    /*second DFS visit in order of decreasing finishing time (Q[].firstNode)
    as computed in the first DFS visit*/
    for (i = numberOfNodes - 1; i >= 0; i--) {
        /*(1)*/
        BisIndexType temp = Q[i].firstNode; //node that has to be visited
        if ((Q[temp].prevBlock) == BIS_BLACK) {
            SecondDFS_visit(temp, temp);
            //to normalize the ranks:
            BisIndexType r = G[temp].rank;
            if (r != -1) {
                if ((r % 2) != 0)
                    Q[r / 2].size = 2;
                else {
                    if (Q[r / 2].size == 0)
                        Q[r / 2].size = 1;
                }
            }
            /*we mark the values of the rank actually used.
            Q[i].size==0 means that there are no nodes of rank i
            Q[i].size==1 means that there are only nodes with even rank i/2
            Q[i].size==2 means that there are nodes with even rank (i-1)/2
                and odd rank (i-1)/2 - 1
            At the end Q[].size contains a sequence of 1 and 2 followed by 0
            N.B.: we need to analyse the different values of the rank; hence we can
            scan only the rank of the forefathers and avoid scanning the rank of all
            the other nodes*/

            /*the rank of the forefathers has to be duplicated since it can
            be overwritten during the normalization of the ranks in initFBA()*/
            Q[temp].nextBlock = G[temp].rank;
        }
    }
    //at the end of the second DFS visit all the nodes are BIS_WHITE (=0)
}

/*PRE-CONDITION OF Rank():
  a graph G and his inverse;
  POST-CONDITION OF Rank():
Q[i].superBlock == 'i' => 'i' is a forefather,
Q[i].superBlock == x != 'i' => x is the forefather of 'i';
NB: the forefather is found during the second DFS visit of SCC():
  in fact each time we find a BIS_BLACK node during the for-loop (1),
  this node is the forefather of all the nodes we will discover
  (their color is BIS_BLACK) in the following DFS visit;
G[i].rank: rank of the nodes in G;
  for each i: G[Q[i].superBlock].rank is the rank of i,
  forefathers have proper rank,
  collapsing nodes have not significant value and hence
    should refer to their forefather for the rank;
G[i].WFflag represents the condition of well-fondness

Q[].size is used to normalize the ranks: for an explanation see InitFba*/

/*modified version of DFS_visit to optimise the computation of Rank();
  firstDFS_visit visits G-1 and stores the finishing time in Q[].firstNode;
  'i' is the node being visited*/
void Bisimulation::FirstDFS_visit(const BisIndexType i) {
    //visit G-1
    auto adj_1 = G[i].adj_1;
    Q[i].prevBlock = BIS_GRAY;
    while (adj_1 != nullptr) {
        BisIndexType j = adj_1->node;
        if (Q[j].prevBlock == BIS_WHITE)
            FirstDFS_visit(j);
        adj_1 = adj_1->next;
    }
    Q[i].prevBlock = BIS_BLACK;
    //store the finishing time of the first DFS visit
    Q[t++].firstNode = i;
    /*the node 'i' is stored at the index t corresponding to its finishing time
     to avoid re-ordering*/
}

/*modified version of DFS_visit to optimise the computation of Rank();
  secondDFS_visit visits G and stores the forefather of the nodes in
  Q[].superBlock;
  'i' is the node being visited, ff is its forefather;
  remember that the meaning of the colors is inverted: BIS_WHITE <--> BIS_BLACK*/
void Bisimulation::SecondDFS_visit(const BisIndexType i, const BisIndexType ff) {
    //stores the temporary value of the rank computed from the children of a node
    BisIndexType tempRank;

    //visit G
    Q[i].prevBlock = BIS_GRAY;
    G[i].rank = -1;

    /*if a node is not a forefather, it is certain that it is not well-founded;
    the forefathers are well-founded only if all their children are well-founded
    (and it will result during the visit)*/
    if (i != ff)
        G[i].WFflag = false;

    auto adj = G[i].adj;
    while (adj != nullptr) {
        BisIndexType j = adj->node;
        //it is the first time we visit j that is in the same SCC of i
        if (Q[j].prevBlock == BIS_BLACK) {
            Q[j].superBlock = ff;
            SecondDFS_visit(j, ff);
        }
        if (Q[i].superBlock == Q[j].superBlock) {
            //SCC(i) == SCC(j)
            //the rank should not increase
            tempRank = G[j].rank;
            //'i' is non-well-founded
            G[i].WFflag = false;
        } else {
            //SCC(i) != SCC(j)
            if (G[j].WFflag == true) //j well-founded
            //the rank should increase of 1
                tempRank = G[Q[j].superBlock].rank + 1;
            else {
                //j not well-founded
                //the rank should not increase
                tempRank = G[Q[j].superBlock].rank;
                //j is non-well-founded then also i
                G[i].WFflag = false;
            }
        }
        //the rank of a node is the max among the ranks of the children
        if (tempRank > G[i].rank)
            G[i].rank = tempRank;
        adj = adj->next;
    } //endwhile

    //well-founded nodes have even rank
    if (G[i].WFflag == true)
        (G[i].rank)++;
    Q[i].prevBlock = BIS_WHITE;
}

/*it returns an exit code: 0 means proceed the computation with
FastBisimulationAlgorithm()*/
int Bisimulation::InitFBA() {
    BisIndexType i;

    //to normalize the ranks
    for (i = 1; i < numberOfNodes && Q[i].size != 0; i++)
        Q[i].size = Q[i].size + Q[i - 1].size;
    maxRank = Q[i - 1].size - 1;
    /*the technique used to normalize the ranks is the well-know one,
    used for example in counting sort*/

    /*here we could find out some special cases, in which
    we know what kind of graph we have:  */
    if (maxRank == numberOfNodes - 1) {
        //std::cout << "LINEAR GRAPH";
        return 1;
    }
    /*if we allow nodes with labels, the following is not anymore true
    if (maxRank==-1){
      //std::cout << "OMEGA";
      return 2;
    }
    if (maxRank==0){  //not allowing not connected graph
      //std::cout << "EMPTY";
      return 3;
    }*/

    for (i = 0; i < numberOfNodes; i++) {
        //all the nodes will have proper rank
        BisIndexType temp = Q[Q[i].superBlock].nextBlock; //is the rank of the forefather
        if (temp == -1 || temp == 0)
            G[i].rank = temp;
        else if ((temp % 2) == 0)
            G[i].rank = Q[temp / 2 - 1].size;
        else
            G[i].rank = Q[temp / 2 - 1].size + 1;
        /*the technique used to normalize the ranks is the well-know one,
        used for example in counting sort*/

        //clearing for the next initialisation phase
        Q[i].firstNode = BIS_NIL;
        B1[i] = numberOfNodes;
        B_1[i] = numberOfNodes;
        splitD[i] = numberOfNodes;
    }

    BisIndexType j, tmpi;


    //initialization of the limit of the arrays
    QBlockLimit = numberOfNodes;
    freeQBlock = maxRank + 2; //one for each rank included rank -1 and 0

    //initialisation of G
    /*we are going to scan the nodes regarding the label-blocks
    so we can build the bisimulation structures that are composed by
    X-blocks and Q-blocks regarding both ranks and labels*/
    for (BisIndexType l = 0; l != BIS_NIL; l = X[l].nextXBlock) //for each label block
        for (i = X[l].firstBlock; i != BIS_NIL; i = tmpi) {
            //for each node
            //the node 'i' has to be inserted in the block j corresponding to rank j-1
            tmpi = G[i].nextInBlock;
            j = G[i].rank + 1;
            if (Q[j].firstNode == BIS_NIL) {
                //the block is empty: add i
                G[i].nextInBlock = BIS_NIL;
                G[i].prevInBlock = BIS_NIL;
                G[i].block = j;
                Q[j].firstNode = i;
                Q[j].size = 1;
                Q[j].nextBlock = BIS_NIL;
                Q[j].superBlock = j;
                Q[j].prevBlock = BIS_NIL;
            } else if (G[Q[j].firstNode].label == G[i].label) {
                /*the block is not empty and contains nodes with the same label of
                the node being inserted*/
                G[i].nextInBlock = Q[j].firstNode;
                G[Q[j].firstNode].prevInBlock = i;
                G[i].prevInBlock = BIS_NIL;
                G[i].block = j;
                Q[j].firstNode = i;
                (Q[j].size)++;
            } else {
                /*the block is not empty and contains nodes with a different label (A)
                from the label (B) of the node being inserted; it is necessary to
                create a new block. The nodes with label A should be moved to the new
                block and the nodes with label B should be associated with the block of
                index rank(+1)*/

                //create a new block
                const BisIndexType newBlock = freeQBlock++; //index of the new block

                //move nodes with label A to the new block
                Q[newBlock].size = Q[j].size;
                Q[newBlock].firstNode = Q[j].firstNode;
                Q[newBlock].superBlock = Q[j].superBlock;
                if (Q[j].nextBlock != BIS_NIL)
                    Q[Q[j].nextBlock].prevBlock = newBlock;
                Q[newBlock].nextBlock = Q[j].nextBlock;
                Q[newBlock].prevBlock = j;
                /*moving these nodes we need to update G[].block to the new value
                of newBlock. The complexity can arise only to O(nON): we can update
                G[].block only once for each node*/
                for (BisIndexType k = Q[newBlock].firstNode; k != BIS_NIL; k = G[k].nextInBlock)
                    G[k].block = newBlock;
                /*instead of updating G[].block we could have chosen to move the nodes
                with label B to the new block. In this case it would have been necessary
                to scan the chain of different (as regards the label) blocks but with
                the same rank: in the worst case the complexity can become O((nON)^2) */

                //insert the new node with label B in the block just freed
                Q[j].nextBlock = newBlock;
                //Q[j].prevBlock = BIS_NIL; already BIS_NIL
                //Q[j].superBlock = j; already j
                Q[j].size = 1;
                Q[j].firstNode = i;

                G[i].nextInBlock = BIS_NIL;
                G[i].prevInBlock = BIS_NIL;
                G[i].block = j;
            } //end else
        } //end fors

    //initialisation of Q and X
    for (i = 0; i < maxRank + 2; i++) {
        X[i].nextXBlock = i + 1;
        X[i].prevXBlock = i - 1;
        X[i].firstBlock = i;
    }
    X[0].prevXBlock = BIS_NIL;
    X[maxRank + 1].nextXBlock = BIS_NIL;
    freeXBlock = maxRank + 2;

    C = 0;

    //clearing of X (continue initialisation)
    for (i = maxRank + 2; i < freeQBlock; i++) {
        //freeQBlock < nON
        X[i].prevXBlock = BIS_NIL;
        X[i].firstBlock = BIS_NIL;
        X[i].nextXBlock = i + 1;
    }

    //clearing of Q and X (continue initialisation)
    for (i = freeQBlock; i < numberOfNodes; i++) {
        Q[i].size = 0;
        Q[i].nextBlock = i + 1;
        Q[i].prevBlock = BIS_NIL;
        Q[i].superBlock = BIS_NIL;
        Q[i].firstNode = BIS_NIL;

        X[i].nextXBlock = i + 1;
        X[i].prevXBlock = BIS_NIL;
        X[i].firstBlock = BIS_NIL;
    }
    X[numberOfNodes - 1].nextXBlock = BIS_NIL;
    if (numberOfNodes != freeQBlock)
        Q[numberOfNodes - 1].nextBlock = BIS_NIL;
    else
        freeQBlock = BIS_NIL;


    //initialisation of the counters
    //initially there is a count per node count(x,U)=|E({x})|
    std::shared_ptr<BisAdjList> adj = nullptr;
    std::shared_ptr<BisCounter> cxS = nullptr;
    for (i = 0; i < numberOfNodes; i++) {
        adj = G[i].adj;
        //to avoid the creation of a BisCounter set to zero
        if (adj == nullptr)
            continue;
        cxS = std::make_shared<BisCounter>();
        cxS->value = 0;
        while (adj != nullptr) {
            (cxS->value)++;
            //each edge xEy contains a pointer to count(x,U);
            //remember that each edge y(E-1)x contains a pointer to the edge xEy!
            adj->countxS = cxS;
            adj = adj->next;
        }
        cxS->node = i;
    }

    /*divide the edges: borderEdges[i] stores the edges going to 'i' from nodes
    of different rank and G[i].adj_1 stores only the edges going to 'i' from
    nodes having the same rank*/

    for (i = 0; i < numberOfNodes; i++) {
        j = G[i].rank;
        std::shared_ptr<BisAdjList_1> adj_1 = G[i].adj_1;
        std::shared_ptr<BisAdjList_1> a = nullptr;
        std::shared_ptr<BisAdjList_1> b = nullptr;
        while (adj_1 != nullptr) {
            std::shared_ptr<BisAdjList_1> next = adj_1->next;
            if (j == G[adj_1->node].rank) {
                adj_1->next = a;
                a = adj_1;
            } else {
                adj_1->next = b;
                b = adj_1;
            }
            adj_1 = next;
        }
        G[i].adj_1 = a;
        borderEdges[i] = b;
    }

    return 0;
}

//compute Paige and Tarjan modified for the fast bisimulation algorithm.
//It analysed only the nodes of Rank: rank that are in the Xblock C.

void Bisimulation::PaigeTarjan(const BisIndexType rank) {
    //pointer to the X-Blocks S and S1
    BisIndexType B, S_B; //pointer to the Q-Blocks B and S-B
    BisIndexType oldD, newD; //old and new block of x belonging to E-1(B)
    std::shared_ptr<BisAdjList_1> adj = nullptr;
    std::shared_ptr<BisCounter> cxS = nullptr;
    BisIndexType x;
    BisIndexType e;
    BisIndexType super;

    //REMINDER: XBlock that are not in C but have Rank: rank
    rankPartition = BIS_NIL;

    while (C != BIS_NIL && rank == G[Q[X[C].firstBlock].firstNode].rank) {
        /*Step 1(select a refining block) & Step 2(update X)*/
        //select some block S from C
        BisIndexType S = C;
        /*if S has more than two blocks, it has to be put back to C;
        hence it is not removed from X until we are sure it is not still
        compound after removing B from it*/

        /*examine the first two blocks in the of blocks of Q contained in S;
        let B be the smaller, remove B from S*/
        if (Q[X[S].firstBlock].size < Q[Q[X[S].firstBlock].nextBlock].size) {
            B = X[S].firstBlock;
            S_B = Q[X[S].firstBlock].nextBlock;
            X[S].firstBlock = S_B;
            Q[B].nextBlock = BIS_NIL;
            Q[S_B].prevBlock = BIS_NIL;
        } else {
            B = Q[X[S].firstBlock].nextBlock;
            S_B = X[S].firstBlock;
            Q[S_B].nextBlock = Q[B].nextBlock;
            if (Q[S_B].nextBlock != BIS_NIL)
                Q[Q[S_B].nextBlock].prevBlock = S_B;
            Q[B].nextBlock = BIS_NIL;
            Q[B].prevBlock = BIS_NIL;
        }

        //and create a new simple block S1 of X containing B as its only block of Q;
        //REMINDER S1 is not in C, hence has to be added to rankPartition
        const BisIndexType S1 = freeXBlock;
        freeXBlock = X[freeXBlock].nextXBlock;
        Q[B].superBlock = S1;
        if (rankPartition != BIS_NIL)
            X[rankPartition].prevXBlock = S1;
        X[S1].nextXBlock = rankPartition;
        rankPartition = S1;
        //X[S1].prevXBlock = BIS_NIL;
        X[S1].firstBlock = B;
        //X[S1].countxS is initialised in step 3

        //check if S is still compound
        if (Q[S_B].nextBlock == BIS_NIL) {
            //not compound: remove S from C
            C = X[C].nextXBlock;
            if (C != BIS_NIL)
                X[C].prevXBlock = BIS_NIL;
            //REMINDER: S has to be added to rankPartition
            if (rankPartition != BIS_NIL)
                X[rankPartition].prevXBlock = S;
            X[S].nextXBlock = rankPartition;
            rankPartition = S;
            X[S].prevXBlock = BIS_NIL;
        }

        /*Step 3(compute E-1(B))*/
        /*by scanning the edges xEy such that y belongs to B
        and adding each element x in such an edge to E-1(B),
        if it has not already been added and if it has the same rank of y.
        REMINDER: the adjacency list of G-1 has been cut from the edges to the
        nodes having different rank, hence we don't need to do anything special!
        Duplicates are suppressed by marking elements: B_1
        Side effect: copy the elements of B in B1
        During the same scan, compute count(x,B)=count(x,S1) because S1={B};
        create a new BisCounter record and make G[x].countxB point to it*/
        BisIndexType y = b1List = Q[B].firstNode;
        b_1List = BIS_NIL;
        while (y != BIS_NIL) {
            //for each y belonging to B
            B1[y] = G[y].nextInBlock; //copy the elements of B in B1
            adj = G[y].adj_1;
            while (adj != nullptr) {
                //for each node in the adj_1 of y
                x = adj->node;
                if (B_1[x] == numberOfNodes) {
                    //node not already added to E-1(B)
                    B_1[x] = b_1List;
                    b_1List = x;
                    //create a new BisCounter: it is pointed by G[x].countxB    /*1*/
                    cxS = std::make_shared<BisCounter>();
                    cxS->node = x;
                    cxS->value = 1;
                    G[x].countxB = cxS; /*1*/
                } else
                    (G[x].countxB->value)++;
                adj = adj->next; //next node in the adj_1 of y
            }
            y = G[y].nextInBlock; //next node y belonging to B
        }

        /*Step 4(refine Q with respect to B)*/
        /*for each block D of Q containing some element of E-1(B)
        split D into D1=D ^ E-1(B) and D2=D - D1*/
        dList = BIS_NIL;
        //do this by scanning the elements of E-1(B)
        x = b_1List;
        while (x != BIS_NIL) {
            //for each x belonging to E-1(B)
            //determine the block D of Q containing it
            oldD = G[x].block; //index of D (old block of x)
            //and create an associated block D1 if one does not already exist
            if (splitD[oldD] == numberOfNodes) {
                //block D not already split
                splitD[oldD] = dList;
                dList = oldD;
                //create a new block D1
                if (freeQBlock == BIS_NIL) {
                    freeQBlock = QBlockLimit++;
                    Q[freeQBlock].size = 0;
                    Q[freeQBlock].nextBlock = BIS_NIL;
                    splitD[freeQBlock] = numberOfNodes;
                    //not necessary to initialise
                    //Q[freeQBlock].prevBlock = BIS_NIL;
                    //Q[freeQBlock].superBlock = BIS_NIL;
                    //Q[freeQBlock].firstNode = BIS_NIL;
                }
                newD = freeQBlock; //index of D1 (new block of x)
                freeQBlock = Q[freeQBlock].nextBlock;
                Q[newD].firstNode = BIS_NIL;
                /*insert D1 just after D, so we know that, if D has already been
                split, the associated D1 is the next block
                REMINDER: it maintains the invariant that the Qblocks in a Xblock
                have all the same rank*/
                Q[newD].nextBlock = Q[oldD].nextBlock;
                Q[oldD].nextBlock = newD;
                Q[newD].prevBlock = oldD;
                if (Q[newD].nextBlock != BIS_NIL)
                    Q[Q[newD].nextBlock].prevBlock = newD;
                Q[newD].superBlock = Q[oldD].superBlock;
            } else //block D already split
                newD = Q[oldD].nextBlock;
            //move x from D to D1
            if (G[x].prevInBlock != BIS_NIL)
                G[G[x].prevInBlock].nextInBlock = G[x].nextInBlock;
            else
                Q[G[x].block].firstNode = G[x].nextInBlock;
            if (G[x].nextInBlock != BIS_NIL)
                G[G[x].nextInBlock].prevInBlock = G[x].prevInBlock;
            G[x].block = newD;
            G[x].nextInBlock = Q[newD].firstNode;
            G[x].prevInBlock = BIS_NIL;
            if (Q[newD].firstNode != BIS_NIL)
                G[Q[newD].firstNode].prevInBlock = x;
            Q[newD].firstNode = x;
            (Q[oldD].size)--;
            (Q[newD].size)++;

            y = x;
            x = B_1[x];
            //re-initialisation of B_1
            B_1[y] = numberOfNodes;
        } //endwhile

        //dList points to the list of new blocks splitD
        BisIndexType d = dList;
        while (d != BIS_NIL) {
            super = Q[d].superBlock;
            if (Q[d].firstNode == BIS_NIL) {
                //D empty: remove it and free its space
                if (Q[d].prevBlock != BIS_NIL)
                    Q[Q[d].prevBlock].nextBlock = Q[d].nextBlock;
                else
                    X[super].firstBlock = Q[d].nextBlock;
                //we are sure that after D,there is D1
                Q[Q[d].nextBlock].prevBlock = Q[d].prevBlock;
                //re-initialise Q[d]
                //Q[d].size is already zero
                //Q[d].firstNode = BIS_NIL;
                Q[d].prevBlock = BIS_NIL;
                Q[d].superBlock = BIS_NIL;
                //free Q[d]
                Q[d].nextBlock = freeQBlock;
                freeQBlock = d;
            } else {
                /*if D nonempty and the superBlock containing D and D1 (super) has been
                made compound by the split, add this block to C
                REMINDER: and remove super from rankPartition
                REMINDER: super has to be added near the other Xblocks with same rank
                since we are computing within a particular rank (BisAdjList_1 of G has been
                cut) D and hence D1 have Rank: rank (and also super has) we just need to
                add super at the beginning of the list pointed by C of */
                if (Q[d].prevBlock == BIS_NIL && Q[Q[d].nextBlock].nextBlock == BIS_NIL) {
                    //D and D1 are the only blocks in this just split Xblock
                    //REMINDER: remove super from rankPartition
                    if (X[super].prevXBlock != BIS_NIL)
                        X[X[super].prevXBlock].nextXBlock = X[super].nextXBlock;
                    else
                        rankPartition = X[super].nextXBlock;
                    if (X[super].nextXBlock != BIS_NIL)
                        X[X[super].nextXBlock].prevXBlock = X[super].prevXBlock;
                    X[super].nextXBlock = BIS_NIL;
                    X[super].prevXBlock = BIS_NIL;
                    //REMINDER: we insert the super in the C chain at the first place
                    X[super].nextXBlock = C;
                    X[super].prevXBlock = BIS_NIL;
                    if (C != BIS_NIL)
                        X[C].prevXBlock = super;
                    C = super;
                }
            }

            e = d;
            d = splitD[d];
            //re-initialisation of splitD
            splitD[e] = numberOfNodes;
        }

        /*Step 5(compute E-1(B) - E-1(S - B))*/
        /*Scan each x such that xEy and y belongs to B; determine count(x,B)
        to which G[x].countxB points and count(x,S) to which xEy points
        (y belongs to B -> scan y(E-1)x -> y(E-1)x points to xEy ->
          xEy points to count(x,S))
        To save space we use again the array B_1 to store E-1(B) - E-1(S - B)
        REMINDER: the adjacency list of G-1 has been cut from the edges to the
        nodes having different rank, hence we don't need to do anything special!*/
        y = b1List;
        b_1List = BIS_NIL;
        while (y != BIS_NIL) {
            //for each y belonging to B1
            adj = G[y].adj_1;
            while (adj != nullptr) {
                //for each node in the adj_1 of y -> scan xEy, y in B
                x = adj->node;
                if (G[x].countxB->value == adj->adj->countxS->value)
                    if (B_1[x] == numberOfNodes) {
                        //x is a node not already added to E-1(S - B)
                        B_1[x] = b_1List;
                        b_1List = x;
                    }
                adj = adj->next;
            }
            y = B1[y];
        }

        /*Step 6(refine Q with respect to S_B)*/
        /*proceed exactly as in Step 4, but scan E-1(B) - E-1(S - B)
        For each block D of Q containing some element of E-1(B) - E-1(S - B)
        split D into D1 = D ^ (E-1(B) - E-1(S - B)) and D2 = D - D1*/
        dList = BIS_NIL;
        //do this by scanning the elements of E-1(B) - E-1(S - B)
        x = b_1List;
        while (x != BIS_NIL) {
            //to process an element x belonging to E-1(B) - E-1(S - B)
            //determine the block D of Q containing it
            oldD = G[x].block; //index of D (old block of x)
            //and create an associated block D1 if one does not already exist
            if (splitD[oldD] == numberOfNodes) {
                //block D not already split
                splitD[oldD] = dList;
                dList = oldD;
                //create a new block D1
                if (freeQBlock == BIS_NIL) {
                    freeQBlock = QBlockLimit++;
                    Q[freeQBlock].size = 0;
                    Q[freeQBlock].nextBlock = BIS_NIL;
                    splitD[freeQBlock] = numberOfNodes;
                    //not necessary to initialise
                    //Q[freeQBlock].prevBlock = BIS_NIL;
                    //Q[freeQBlock].superBlock = BIS_NIL;
                    //Q[freeQBlock].firstNode = BIS_NIL;
                }
                newD = freeQBlock; //index of D1 (new block of x)
                freeQBlock = Q[freeQBlock].nextBlock;
                Q[newD].firstNode = BIS_NIL;
                /*insert D1 just after D, so we know that, if D has already
                been split, the associated D1 is the next block
                REMINDER: it maintains the invariant that the Qblocks in a Xblock
                have all the same rank*/
                Q[newD].nextBlock = Q[oldD].nextBlock;
                Q[oldD].nextBlock = newD;
                Q[newD].prevBlock = oldD;
                if (Q[newD].nextBlock != BIS_NIL)
                    Q[Q[newD].nextBlock].prevBlock = newD;
                Q[newD].superBlock = Q[oldD].superBlock;
            } else //block D already split
                newD = Q[oldD].nextBlock;
            //move x from D to D1
            if (G[x].prevInBlock != BIS_NIL)
                G[G[x].prevInBlock].nextInBlock = G[x].nextInBlock;
            else
                Q[G[x].block].firstNode = G[x].nextInBlock;
            if (G[x].nextInBlock != BIS_NIL)
                G[G[x].nextInBlock].prevInBlock = G[x].prevInBlock;
            G[x].block = newD;
            G[x].nextInBlock = Q[newD].firstNode;
            G[x].prevInBlock = BIS_NIL;
            if (Q[newD].firstNode != BIS_NIL)
                G[Q[newD].firstNode].prevInBlock = x;
            Q[newD].firstNode = x;
            (Q[oldD].size)--;
            (Q[newD].size)++;

            y = x;
            x = B_1[x];
            //re-initialisation of B_1
            B_1[y] = numberOfNodes;
        } //endwhile

        //dList points to the list of new blocks splitD
        d = dList;
        while (d != BIS_NIL) {
            super = Q[d].superBlock;
            if (Q[d].firstNode == BIS_NIL) {
                //D empty: remove it and free its space
                if (Q[d].prevBlock != BIS_NIL)
                    Q[Q[d].prevBlock].nextBlock = Q[d].nextBlock;
                else
                    X[super].firstBlock = Q[d].nextBlock;
                Q[Q[d].nextBlock].prevBlock = Q[d].prevBlock;
                //re-initialise Q[d]
                //Q[d].size is already zero
                //Q[d].firstNode = BIS_NIL;
                Q[d].prevBlock = BIS_NIL;
                Q[d].superBlock = BIS_NIL;
                //free Q[d]
                Q[d].nextBlock = freeQBlock;
                freeQBlock = d;
            } else {
                /*if D nonempty and the superBlock containing D and D1 (super) has been
                made compound by the split, add this block to C
                REMINDER: and remove super from rankPartition
                REMINDER: super has to be added near the other Xblocks with same rank
                since we are computing within a particular rank (BisAdjList_1 of G has been
                cut) D and hence D1 have Rank: rank (and also super has) we just need to
                add super at the beginning of the list pointed by C of */
                if (Q[d].prevBlock == BIS_NIL && Q[Q[d].nextBlock].nextBlock == BIS_NIL) {
                    //D and D1 are the only blocks in this just split Xblock
                    //REMINDER: remove super from rankPartition
                    if (X[super].prevXBlock != BIS_NIL)
                        X[X[super].prevXBlock].nextXBlock = X[super].nextXBlock;
                    else
                        rankPartition = X[super].nextXBlock;
                    if (X[super].nextXBlock != BIS_NIL)
                        X[X[super].nextXBlock].prevXBlock = X[super].prevXBlock;
                    X[super].nextXBlock = BIS_NIL;
                    X[super].prevXBlock = BIS_NIL;
                    /*REMINDER: we insert the super in the C chain at the first place*/
                    X[super].nextXBlock = C;
                    X[super].prevXBlock = BIS_NIL;
                    if (C != BIS_NIL)
                        X[C].prevXBlock = super;
                    C = super;
                }
            }
            e = d;
            d = splitD[d];
            //re-initialisation of splitD
            splitD[e] = numberOfNodes;
        }

        /*Step 7(update counts)*/
        /*scan the edges xEy tc y belongs to B1.
        To process an edge decrement count(x,S) (to which xEy points).
        If this count becomes zero delete the count record,
        and make xEy point to count(x,B) (to which x points).
        Discard B1 (re-initialise it).*/
        y = b1List;
        while (y != BIS_NIL) {
            //for each y belonging to B
            adj = G[y].adj_1;
            while (adj != nullptr) {
                //for each node in the adj_1 of y -> scan xEy, y in B
                x = adj->node;
                cxS = adj->adj->countxS;
                if (cxS->value != 1) {
                    (cxS->value)--;
                    adj->adj->countxS = G[x].countxB;
                } else {
                    //count(x,S) becomes zero
                    //make xEy point to count(x,B)
                    adj->adj->countxS = G[x].countxB;
                    //delete count(x,S)
                }
                adj = adj->next;
            }
            x = y;
            y = B1[y];
            //re-initialisation of B1
            B1[x] = numberOfNodes;
        }
    } //end while
}

//split computes a single split as regards the single block B.

/*The function consists of step 3 and step 4 of PaigeTarjan, even if there are
some differences: once used for the split, B is not anymore necessary; so B1, S1
and the counters are not computed; since we are interested in the edges between
nodes of different rank we scan borderEdges[] instead of G[].adj_1*/
void Bisimulation::Split(const BisIndexType B) {
    BisIndexType newD; //old and new block of x belonging to E-1(B)
    std::shared_ptr<BisAdjList_1> adj = nullptr;
    BisIndexType x;

    /*Step 3(compute E-1(B))*/
    /*by scanning the edges xEy such that y belongs to B
    and adding each element x in such an edge to E-1(B),
    if it has not already been added.
    REMINDER: we have to scan only the edges between nodes
    of different rank
    Duplicates are suppressed by marking elements: B_1*/
    BisIndexType y = b1List = Q[B].firstNode;
    b_1List = BIS_NIL;
    while (y != BIS_NIL) {
        //for each y belonging to B
        adj = borderEdges[y]; //instead G[y].adj_1;
        while (adj != nullptr) {
            //for each node in the adj_1 of y
            x = adj->node;
            if (B_1[x] == numberOfNodes) {
                //node not already added to E-1(B)
                B_1[x] = b_1List;
                b_1List = x;
            }
            adj = adj->next; //next node in the adj_1 of y
        }
        y = G[y].nextInBlock; //next node y belonging to B
    }

    /*Step 4(refine Q with respect to B)*/
    /*for each block D of Q containing some element of E-1(B)
    split D into D1 = D ^ E-1(B) and D2 = D - D1*/
    dList = BIS_NIL;
    //do this by scanning the elements of E-1(B)
    x = b_1List;
    while (x != BIS_NIL) {
        //for each x belonging to E-1(B)
        //determine the block D of Q containing it
        BisIndexType oldD = G[x].block; //index of D (old block of x)
        //and create an associated block D1 if one does not already exist
        if (splitD[oldD] == numberOfNodes) {
            //block D not already split
            splitD[oldD] = dList;
            dList = oldD;
            //create a new block D1
            if (freeQBlock == BIS_NIL) {
                //check for free space in memory
                freeQBlock = QBlockLimit++;
                Q[freeQBlock].size = 0;
                Q[freeQBlock].nextBlock = BIS_NIL;
                splitD[freeQBlock] = numberOfNodes;
                //not necessary to initialise
                //Q[freeQBlock].prevBlock = BIS_NIL;
                //Q[freeQBlock].superBlock = BIS_NIL;
                //Q[freeQBlock].firstNode = BIS_NIL;
            }
            newD = freeQBlock; //index of D1 (new block of x)
            freeQBlock = Q[freeQBlock].nextBlock;
            Q[newD].firstNode = BIS_NIL;
            /*insert D1 just after D, so we know that, if D has already been
            split, the associated D1 is the next block
            REMINDER: it maintains the invariant that the Qblocks in a Xblock
            have all the same rank*/
            Q[newD].nextBlock = Q[oldD].nextBlock;
            Q[oldD].nextBlock = newD;
            Q[newD].prevBlock = oldD;
            if (Q[newD].nextBlock != BIS_NIL)
                Q[Q[newD].nextBlock].prevBlock = newD;
            Q[newD].superBlock = Q[oldD].superBlock;
        } else
            newD = Q[oldD].nextBlock;
        //move x from D to D1
        if (G[x].prevInBlock != BIS_NIL)
            G[G[x].prevInBlock].nextInBlock = G[x].nextInBlock;
        else
            Q[G[x].block].firstNode = G[x].nextInBlock;
        if (G[x].nextInBlock != BIS_NIL)
            G[G[x].nextInBlock].prevInBlock = G[x].prevInBlock;
        G[x].block = newD;
        G[x].nextInBlock = Q[newD].firstNode;
        G[x].prevInBlock = BIS_NIL;
        if (Q[newD].firstNode != BIS_NIL)
            G[Q[newD].firstNode].prevInBlock = x;
        Q[newD].firstNode = x;
        (Q[oldD].size)--;
        (Q[newD].size)++;

        y = x;
        x = B_1[x];
        //re-initialisation of B_1
        B_1[y] = numberOfNodes;
    } //endwhile

    //dList points to the list of new blocks splitD
    BisIndexType d = dList;
    while (d != BIS_NIL) {
        if (Q[d].firstNode == BIS_NIL) {
            //D empty: remove it and free its space
            if (Q[d].prevBlock != BIS_NIL)
                Q[Q[d].prevBlock].nextBlock = Q[d].nextBlock;
            else
                X[Q[d].superBlock].firstBlock = Q[d].nextBlock;
            //we are sure that after D,there is D1
            Q[Q[d].nextBlock].prevBlock = Q[d].prevBlock;
            //re-initialise Q[d]
            //Q[d].size is already zero
            Q[d].prevBlock = BIS_NIL;
            Q[d].superBlock = BIS_NIL;
            Q[d].firstNode = BIS_NIL;
            //free Q[d]
            Q[d].nextBlock = freeQBlock;
            freeQBlock = d;
        }
        /*REMINDER: if D is not empty, D and D1 have as superBlock the Xblock
        containing all the nodes of their rank; this superBlock is already in C
        from the initialization*/

        const BisIndexType e = d;
        d = splitD[d];
        //re-initialisation of splitD
        splitD[e] = numberOfNodes;
    }
}

//compute FastBisimulationAlgorithms

void Bisimulation::FastBisimulationAlgorithm() {
    /*VERY IMPORTANT:
    before computing the minimisation for that rank component,
    each rank component is represented by only ONE XBlock
    and eventually more than one QBlock*/

    /* for each Rank i: {R(ij)=P&T(Ri); for each j: split(Rij);}
    Exception 1 : remember that well-founded blocks can be refined only by blocks
    having lower rank (there are no edges between nodes in the same rank
    component). Hence, we do not need to compute PT or PTB on this kind of blocks.
    Exception 2: also if there is only a QBlock representing a rank component it
    is not necessary to compute PT or PTB on it*/
    for (BisIndexType i = -1; i <= maxRank; i++) {
        /*in the case of the Exceptions above, we remove the XBlock, representing
        the rank component, from C and store it in rankPartition, ready to the
        successive sequence of simple splits */
        if ((Q[X[C].firstBlock].nextBlock == BIS_NIL) ||
            G[Q[X[C].firstBlock].firstNode].WFflag) {
            rankPartition = C;
            C = X[C].nextXBlock;
            X[C].prevXBlock = BIS_NIL;
            X[rankPartition].prevXBlock = BIS_NIL;
            X[rankPartition].nextXBlock = BIS_NIL;
        } else {
            //if (MultipleNodes(C))
            PaigeTarjan(i); //rank = i
            //else
            // PaigeTarjanBonic();
        }

        /*the sub-graph of rank 'i' is minimised and all the blocks are in the chain
        pointed by rankPartition*/
        if (i != maxRank) //i==maxRak means last rank and no more split
            while (rankPartition != BIS_NIL) {
                for (BisIndexType l = X[rankPartition].firstBlock; l != BIS_NIL; l = Q[l].nextBlock)
                    Split(l);
                //free the XBlock just used in the split
                const BisIndexType rP = X[rankPartition].nextXBlock;
                X[rankPartition].nextXBlock = freeXBlock;
                freeXBlock = rankPartition;
                X[rankPartition].prevXBlock = BIS_NIL;
                X[rankPartition].firstBlock = BIS_NIL;
                rankPartition = rP;
            }
    }
}

/*----------------------------------------------------------------------------*/

bool Bisimulation::MinimizeAutomaPT(BisAutomata &A) {

    if (InitPaigeTarjan() == 0) {
        // std::cerr << "\nDEBUG: [MinimizeAutomaPT] done init...\n";
        PaigeTarjan();

        // std::cerr << "\nDEBUG: [MinimizeAutomaPT] done PaigeTarjan...\n";

        GetMinimizedAutoma(A);
        // std::cerr << "\nDEBUG: [MinimizeAutomaPT] done minimization...\n";
        return true;
    }
    return false;
}

bool Bisimulation::MinimizeAutomaFB(BisAutomata &A) {

    Rank();
    if (InitFBA() == 0) {
        FastBisimulationAlgorithm();

        GetMinimizedAutoma(A);
        return true;
    }
    return false;
}


/*-------------------------------Conversion---------------------------------------------*/
BisAutomata Bisimulation::kstate_to_automaton(VectorBisWrapper<KripkeWorldPointer> &pworld_vec,
                                              const std::map<Agent, BisLabel> &agent_to_label,
                                              const KripkeState &kstate) {
    std::map<int, int> compact_indices;
    std::map<KripkeWorldPointer, int> index_map;
    BisLabelsMap label_map;

    const auto &worlds = kstate.get_worlds();
    const auto &agents = Domain::get_instance().get_agents();
    int Nvertex = static_cast<int>(worlds.size());
    const int ag_set_size = static_cast<int>(agents.size());

    VectorBisWrapper<Bis_vElem> Vertex(Nvertex);

    const auto &pointed = kstate.get_pointed();
    index_map[pointed] = 0;
    pworld_vec.push_back(pointed);
    compact_indices[static_cast<int>(pointed.get_internal_world_id())] = 0;

    Vertex[0].ne = 0;

    int idx = 1, compact_id = 1;

    for (const auto &world: worlds) {
        if (world != pointed) {
            index_map[world] = idx;
            pworld_vec.push_back(world);

            if (compact_indices.insert({world.get_internal_world_id(), compact_id}).second) {
                compact_id++;
            }

            Vertex[idx].ne = 0;
            ++idx;
        }

        label_map[world][world].insert(compact_indices[static_cast<int>(world.get_internal_world_id())] + ag_set_size);
    }

    int bhtabSize = ag_set_size + compact_id;

    for (const auto &[source, belief_map]: kstate.get_beliefs()) {
        for (const auto &[agent, targets]: belief_map) {
            for (const auto &target: targets) {
                label_map[source][target].insert(agent_to_label.at(agent));
                Vertex[index_map[source]].ne++;
            }
        }
    }

    for (int i = 0; i < Nvertex; ++i) {
        Vertex[i].ne++; // For self-loop?
        Vertex[i].e = VectorBisWrapper<Bis_eElem>(Vertex[i].ne);
    }

    for (const auto &[from_world, edges]: label_map) {
        int from = index_map[from_world];
        int j = 0;

        for (const auto &[to_world, labels]: edges) {
            const int to = index_map[to_world];

            for (const auto &label: labels) {
                Vertex[from].e[j].nbh = 1;
                Vertex[from].e[j].bh = VectorBisWrapper<int>(1);
                Vertex[from].e[j].tv = to;
                Vertex[from].e[j].bh[0] = label;
                ++j;
            }
        }
    }

    auto *a = new BisAutomata;
    a->Nvertex = Nvertex;
    a->Nbehavs = bhtabSize;
    a->Vertex = Vertex;

    return *a;
}

void Bisimulation::automaton_to_kstate(const BisAutomata &a, const VectorBisWrapper<KripkeWorldPointer> &world_vec,
                                       const std::map<BisLabel, Agent> &label_to_agent, KripkeState &kstate) {
    KripkeWorldPointersSet worlds;
    kstate.clear_beliefs();

    auto agents_size = Domain::get_instance().get_agents().size();

    for (int i = 0; i < a.Nvertex; i++) {
        if (a.Vertex[i].ne > 0) {
            worlds.insert(world_vec[i]);
            for (int j = 0; j < a.Vertex[i].ne; j++) {
                for (int k = 0; k < a.Vertex[i].e[j].nbh; k++) {
                    if (const int label = a.Vertex[i].e[j].bh[k]; static_cast<size_t>(label) < agents_size) {
                        kstate.add_edge(world_vec[i], world_vec[a.Vertex[i].e[j].tv], label_to_agent.at(label));
                    }
                }
            }
        }
    }

    kstate.set_worlds(worlds);
}

void Bisimulation::calc_min_bisimilar(KripkeState &kstate) {
    VectorBisWrapper<KripkeWorldPointer> pworld_vec;
    pworld_vec.reserve(kstate.get_worlds().size());

    std::map<BisLabel, Agent> label_to_agent;
    std::map<Agent, BisLabel> agent_to_label;

    const auto &agents = Domain::get_instance().get_agents();
    BisLabel ag_label = 0;

    for (const auto &agent: agents) {
        label_to_agent[ag_label] = agent;
        agent_to_label[agent] = ag_label++;
    }

    BisAutomata automaton = kstate_to_automaton(pworld_vec, agent_to_label, kstate);

    FillStructures(automaton);
    Inverse();

    /*Removed check from configuration, is kinda of redundant
    if (!Configuration::get_instance().get_bisimulation())
    {
        return;
    }*/

    const bool use_FB = Configuration::get_instance().get_bisimulation_type_bool();
    const bool success = use_FB
                             ? MinimizeAutomaFB(automaton)
                             : MinimizeAutomaPT(automaton);

    if (success) {
        automaton_to_kstate(automaton, pworld_vec, label_to_agent, kstate);
    } else {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::BisimulationFailed,
            use_FB
                ? "Bisimulation with FB failed.\n"
                : "Bisimulation with PT failed.\n"
        );
    }
}
