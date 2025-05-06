#include "bisimulation.h"
//
//BIS_indexType B1[BIS_MAXINDEX]; //copy of B
//BIS_indexType B_1[BIS_MAXINDEX]; //list to maintain E-1(B) and E-1(S-B)
//BIS_indexType splitD[BIS_MAXINDEX]; //list of split blocks
//BIS_indexType b1List, b_1List, dList; //starting pointers of the lists above
//
//
//struct graph G[BIS_MAXINDEX];
//struct qPartition Q[BIS_MAXINDEX];
//struct xPartition X[BIS_MAXINDEX];
//adjList_1 *borderEdges[BIS_MAXINDEX];

/*\***IO_FC2.cpp****/
void bisimulation::DisposeAutoma(automa *a)
/* Rilascia la memoria allocata per l'automa */
{
	BIS_indexType i, j, Nv, Ne;
	v_elem *Vertex;

	Nv = a->Nvertex;
	Vertex = a->Vertex;

	for (i = 0; i < Nv; i++) {
		Ne = Vertex[i].ne;
		for (j = 0; j < Ne; j++) free(Vertex[i].e[j].bh);
		free(Vertex[i].e);
	}
	free(a);
}

void bisimulation::FillStructures(automa *A)
{
	/*
	    Questa funzione si occupa della trasformazione di un automa con i soli archi etichettati
	    in uno con i soli stati etichettati e dell'inizializzazione delle strutture dati X e Q
	    che vengono utilizzate dalle procedure che eseguono gli algoritmi di PaigeTarjan e di
	    FastBisimulation.
	    La chiamata a "fill_structures" verra' effettuata alla fine di "LoadFromFC2", che
	    memorizza in una struttura dati di tipo "automa" il grafo rappresentato all'interno del
	    file .FC2 passatogli come input. Tale struttura dati verra' passata come argomento a
	    "fill_structure" che la analizzera' e creera' un grafo equivalente con i soli stati
	    etichettati. La struttura dati in cui tale automa viene memorizzato e' G, anch'essa
	    utilizzata dalle procedure che implementano PaigeTarjan e FastBisimulation.

	    INPUT:      A, l'automa con soli archi etichettati che dev'essere convertito

	    OUTPUT:     nessuno (Gli array X, Q e G sono globali, quindi non e' stato necessario
			ritornare alcunche' come output)
	 */

	// Variabili
	BIS_indexType i;

	// Inizializzazione della struttura dati X. Tale struttura dati avra' dimensione iniziale
	// pari al numero totale di etichette presenti nell'automa. Ogni elemento dell'array X ha
	// un "puntatore" all'elemento precedente ed uno a quello successivo. L'elemento di indice
	// 0 non avra' alcun elemento precedente e cio' verra' indicato dal valore BIS_NIL, cosi' come
	// l'elemento di indice "A->Nbehavs" non avra' alcun elemento successivo.
	// Il campo "firstBlock" viene qui inizializzato a BIS_NIL, ma verra' riconsiderato e aggiornato
	// correttamente dalla funzione "setpointers".

	// Inizializzazione del primo e dell'ultimo blocco di X
	X[0].prevXBlock = BIS_NIL;
	X[0].nextXBlock = 1;
	X[0].firstBlock = BIS_NIL;
	X[A->Nbehavs - 1].prevXBlock = A->Nbehavs - 2;
	X[A->Nbehavs - 1].nextXBlock = BIS_NIL;
	X[A->Nbehavs - 1].firstBlock = BIS_NIL;

	// Inizializzazione dei restanti blocchi di X
	for (i = 1; i < (A->Nbehavs - 1); i++) {
		X[i].nextXBlock = i + 1;
		X[i].prevXBlock = i - 1;
		X[i].firstBlock = BIS_NIL;
	};

	// Questa funzione si occupa effettivamente della conversione del grafo "A", con i soli
	// archi etichettati, in uno con i soli stati etichettati. Inoltre assegna ad ogni stato
	// l'elemento di X che rappresenta la sua etichetta (per i dettagli si vedano le spiegazioni
	// delle funzioni "createG" e "setpointers")
	CreateG(A->Nvertex, A->Vertex);

	// Gestione dei puntatori tra le strutture dati G ed X
	SetPointers(A->Nbehavs);

	// Procedura che inizializza i restanti campi di G (quelli non inizializzati dalle precedenti
	// funzioni) e quelli di Q
}

void bisimulation::CreateG(int num_v, v_elem *Gtemp)
{
	/*
	    Questa funzione si occupa effettivamente della conversione del grafo con soli archi
	    etichettati, passatogli come input, in uno con soli stati etichettati.
	    Stabilisce inoltre, per ogni stato, a quale blocco esso appartenga (per blocco si intende
	    un elemento dell'array X, che rappresenta in modo univoco una etichetta del grafo). In altre
	    parole, all'inizio, tutti gli stati con uguale etichetta apparterranno ad uno stesso blocco.
	    L'array X ha dimensione pari al numero totale delle etichette.

	    INPUT:      num_v, il numero di vertici del grafo Gtemp
			Gtemp, il grafo con i soli archi etichettati

	    OUTPUT:     nessuno (L'array G e' globale, quindi non e' stato necessario ritornare
			alcunche' come output)
	 */

	// Variabili
	BIS_indexType v, e, b;
	struct adjList **curr_adj;

	// Creo "num_v" vertici in G. Questi sono gli stati presenti anche nel grafo con soli archi
	// etichettati Gtemp e che, per convenzione implementativa, avranno tutti etichetta 0 (in
	// realta' la loro etichetta potrebbe assumere un qualunque valore). Tutti questi stati
	// apparterranno quindi, come si puo' notare dall'assegnamento "G[v].block = 0", al blocco
	// 0, ovvero all'elemento dell'array X di indice 0.
	for (v = 0; v < num_v; v++) {
		G[v].block = 0;
		G[v].label = 0;
	}

	// La variabile "numberOfNodes" e' globale e indica il numero totale di stati del grafo su
	// cui verranno eseguiti gli algorimi di PaigeTarjan e FastBisimulation. A questo punto sono
	// stati creati "num_v" stati, pari al numero totale di stati del grafo con i soli archi
	// etichettati; il suo valore iniziale sara' quindi "num_v" e crescera' man mano che gli archi
	// vengono disetichettati.
	numberOfNodes = num_v;

	// Creo tutti gli altri stati (disetichettando gli archi). Per ogni stato del grafo di partenza
	// Gtemp, scandisco la sua lista di adiacenza. I casi che si possono presentare sono i seguenti:
	// 1. un arco con un'unica etichetta
	// 2. un arco con piu' etichette
	// Nel primo caso, "spezzo" in due l'arco ed inserisco tra i due uno stato con etichetta uguale
	// a quella posseduta dall'arco "spezzato". Nel secondo caso, supponendo che l'arco abbia "n"
	// etichette, creero' "n" stati, ognuno dei quali avra' una delle etichette possedute dall'arco.
	// Tali stati manterranno l'ordine in cui le etichette sono memorizzate sull'arco, nel senso che
	// se l'etichetta "i" precedeva l'etichetta "j" sull'arco, vi sara' un arco dallo stato con
	// etichetta "i" verso quello con etichetta "j". Ovviamente il primo degli "n" stati avra' un
	// arco entrante proveniente dallo stato "v" che stiamo considerando, mentre l'ultimo degli "n"
	// stati, avra' un arco uscente verso lo stato in cui giungeva l'arco che stiamo "spezzando".
	for (v = 0; v < num_v; v++) {
		// Puntatore alla lista di adiacenza dello stato corrente, che verra' creata dalle seguenti
		// linee di codice
		curr_adj = &(G[v].adj);

		// Ciclo che considera tutti gli archi uscenti da "v"
		for (e = 0; e < Gtemp[v].ne; e++) {
			// Creo un nuovo stato (il primo nella catena di etichette). Come detto precedentemente
			// il campo "block" viene qui inizializzato con l'indice dell'elemento di X che rappresenta 
			// l'etichetta dello stato che stiamo creando
			G[numberOfNodes].block = Gtemp[v].e[e].bh[0];
			G[numberOfNodes].label = Gtemp[v].e[e].bh[0];
			numberOfNodes++;

			// Aggiornamento della lista di adiacenza di v. Viene aggiunto un nuovo elemento nella
			// lista di adiacenza di v, ovvero un nuovo arco che raggiunge lo stato appenza creato
			*curr_adj = new struct adjList;
			(*curr_adj)->node = numberOfNodes - 1;
			(*curr_adj)->next = NULL;
			curr_adj = &((*curr_adj)->next);

			// Creo gli stati dal secondo in poi nella catena delle etichette. Questo ciclo viene
			// eseguito solamente nel caso in cui l'arco che stiamo considerando possegga piu' di
			// una etichetta
			for (b = 1; b < Gtemp[v].e[e].nbh; b++) {
				// Creo un nuovo stato
				G[numberOfNodes].block = Gtemp[v].e[e].bh[b];
				G[numberOfNodes].label = Gtemp[v].e[e].bh[b];

				// Aggiorno la lista di adiacenza dello stato precedentemente creato, ovvero aggiungo
				// un arco dal penultimo stato che abbiamo creato a quello appena creato
				G[numberOfNodes - 1].adj = new struct adjList;
				G[numberOfNodes - 1].adj->node = numberOfNodes;
				G[numberOfNodes - 1].adj->next = NULL;

				numberOfNodes++;
			}

			// Aggiornamento della lista di adiacenza dell'ultimo stato che abbiamo creato
			G[numberOfNodes - 1].adj = new struct adjList;
			G[numberOfNodes - 1].adj->node = Gtemp[v].e[e].tv;
			G[numberOfNodes - 1].adj->next = NULL;
		}
	}
}

void bisimulation::SetPointers(int n)
{
	/*
	    Questa funzione gestisce i puntatori tra l'array X e l'array G nel seguente modo:
	    si e' detto che l'array X ha dimesione pari al numero totale di etichette presenti nel grafo.
	    Ogni elemento di questo array rappresenta quindi una classe di stati, ovvero tutti quelli che
	    hanno come etichetta quella di cui tale elemento e' il rappresentante (quindi tutti gli stati
	    con uguale etichetta avranno lo stesso campo "block"). Inoltre, ogni elemento dell'array X
	    possiede uno stato rappresentante (indicato dal campo "firstBlock" di X) ed ogni stato avra'
	    due "puntatori" al precedente ed al successivo elemento nel blocco. Cosicche', dato un
	    qualunque elemento (blocco) di X, guardando il campo "firstBlock" e seguendo i puntatori agli
	    elementi precedente e successivo, e' possibile individuare tutti gli stati appartenenti a tale
	    blocco.

	    INPUT:      n, il numero totale di etichette

	    OUTPUT:     nessuno (G ed X sono globali)
	 */

	// Variabili
	BIS_indexType *curr_node, i, x;

	// Allocazione dell'array temporaneo utilizzato dal ciclo seguente
	curr_node = new int[n];

	//
	for (i = 0; i < numberOfNodes; i++) {
		// Prelevo il blocco a cui appartiene lo stato "i"
		x = G[i].block;

		// Se tale blocco non ha ancora un rappresentante (il campo "firstBlock" e' uguale a BIS_NIL)
		// allora lo stato "i" diventa il suo rappresentante. Ovvero scelgo come rappresentante di
		// un blocco che rappresenta l'etichetta "k", il primo stato che trovo che abbia etichetta "k"
		if (X[x].firstBlock == BIS_NIL) {
			X[x].firstBlock = i;
			G[i].prevInBlock = BIS_NIL;
			G[i].nextInBlock = BIS_NIL;
		}// Se invece tale blocco "x" ha gia' un rappresentante, lego lo stato i, attraverso i puntatori
			// "prevInBlock" e "nextInBlock" all'ultimo elemento che ho trovato con uguale etichetta.
			// L'array "curr_node" viene appunto usato per tenere traccia di quale sia l'ultimo stato che
			// ho trovato, per ogni etichetta (curr_node ha dimensione pari al numero totale di etichette)
		else {
			G[i].prevInBlock = curr_node[x];
			G[i].nextInBlock = BIS_NIL;
			G[curr_node[x]].nextInBlock = i;
		};

		// Aggiorno l'array temporaneo
		curr_node[x] = i;
	};

	// Deallocazione dell'array temporaneo
	delete[] curr_node;
}

void bisimulation::GetMinimizedAutoma(automa *A)
{
	/*
	    Questa funzione crea, a partire dal grafo minimizzato (con soli stati etichettati) memorizzato
	    in G e dal grafo iniziale A (con soli archi etichettati) passato come input, il grafo minimizzato
	    con soli archi etichettati, che verra' salvato su un file .FC2 dalla funzione "SaveToFC2"

	    INPUT:      A, l'automa di partenza con i soli archi etichettati

	    OUTPUT:     nessuno
	 */

	// Con questa funzione vengono indicati gli stati che gli algoritmi di PaigeTarjan e
	// FastBisimulation hanno eliminato (in realta' tali algoritmi non operano tale eliminazione,
	// che viene effettivamente compiuta dalle funzioni "DeleteNodes" e "SaveToFC2")
	MarkDeletedNodes();

	// Viene effettuata la vera e propria eliminazione degli stati marcati com "BIS_DA_CANCELLARE" dalla
	// funzione "MarkDeletedNodes"
	DeleteNodes(A);
}

void bisimulation::MarkDeletedNodes()
{
	/*
	    MarkDeletedNodes non fa altro che marcare gli stati che devono essere cancellati, valutando
	    il valore che alcuni campi degli array G e Q assumono al termine della computazione delle
	    funzioni che implementano gli algoritmi di PaigeTarjan e FastBisimulation. Tali stati vengono
	    effettivamente eliminati dalle funzioni "DeleteNodes" e "SaveToFC2".

	    Nota: le implementazioni degli algoritmi di minimizzazione qui proposte sono distruttive
		  rispetto agli array Q ed X, pertanto l'unico modo per sapere quali stati sono bisimili
		  tra loro, e' controllare il campo "block" di tali stati, che indica dunque il blocco
		  cui essi appartengono.

	    INPUT:      nessuno

	    OUTPUT:     nessuno
	 */

	// Variabili
	BIS_indexType i, q;

	// Al termine delle funzioni di minimizzazione, alcuni campi di Q non servono piu' e vengono
	// qui riutilizzati per altri scopi. Il primo ad essere preso in considerazione e' il campo "size"
	// che viene usato per tenere traccia del fatto che un blocco di Q abbia o meno uno stato
	// rappresentante (un elemento (blocco) di Q rappresenta un insieme di stati bisimili)
	// Inizializzazione del campo "size" a BIS_NOTUSED, per indicare che nessun elemento (blocco) di Q ha
	// ancora il proprio stato rappresentante
	for (i = 0; i < QBlockLimit; i++) //il max di Q puo' essere superiore a nON *****************************************
		Q[i].size = BIS_NOTUSED;

	// Annoto gli stati che devono essere cancellati
	for (i = 0; i < numberOfNodes; i++) {
		// Considero il blocco cui appartiene lo stato "i"
		q = G[i].block;

		// Se tale blocco non ha ancora un rappresentante (il campo "size" e' uguale a BIS_NOTUSED)
		// allora lo stato "i" diventa il suo rappresentante. Ovvero scelgo come rappresentante di
		// un blocco "k", il primo stato che trovo che abbia il campo "block" con valore "k"
		if (Q[q].size == BIS_NOTUSED) {
			Q[q].size = BIS_USED;
			Q[q].firstNode = i;
		}// Altrimenti indico che tale stato dev'essere cancellato, in quanto bisimile allo stato
			// rappresentante del blocco cui esso appartiene. Utilizzo il campo "nextInBlock" in quanto
			// non sara' piu' utilizzato da alcuna funzione
		else
			G[i].nextInBlock = BIS_DA_CANCELLARE;
	}
}

void bisimulation::DeleteNodes(automa *A)
{
	/*
	    DeleteNodes si occupa di indicare quali tra gli stati del grafo di partenza A, passato come input,
	    devono essere cancellati, considerando il campo "nextInBlock" degli elementi dell'array G, modificato
	    dalla funzione "MarkDeletedNodes".
	    L'automa A che viene modificato da questa funzione verra' successivamente passato come input alla
	    funzione "SaveToFC2" che capira' quali stati e quali archi debbano essere memorizzati nel file .FC2
	    che produrra' come output.

	    INPUT:      A, l'automa di partenza, con i soli archi etichettati

	    OUTPUT:     nessuno
	 */

	// Variabili
	BIS_indexType i, j;

	// Segno nell'automa A di partenza quali tra i suoi stati sono stati indicati come "BIS_DA_CANCELLARE"
	// dalla funzione "MarkDeletedNodes"
	for (i = 0; i < A->Nvertex; i++) {
		// Se lo stato "i" dev'essere cancellato, ovvero se la funzione "MarkDeletedNodes" ha impostato
		// il suo campo "nextInBlock" a "BIS_DA_CANCELLARE", allora indico in A che tale nodo dev'essere
		// cancellato. Utilizzo il campo "ne" di "A->Vertex", che indica il numero di archi uscenti, in
		// quanto, visto che tale stato verra' cancellato, esso non servira' piu'
		if (G[i].nextInBlock == BIS_DA_CANCELLARE)
			A->Vertex[i].ne = BIS_DELETED;

			// Se invece tale stato dev'essere conservato, modifico gli archi (se necessario) come segue
		else {
			for (j = 0; j < A->Vertex[i].ne; j++) {
				// Modifica degli archi: se un nodo di quelli che non devono essere cancellati ha un
				// arco uscente che giunge ad un nodo marcato come "BIS_DA_CANCELLARE", faccio puntare
				// l'arco verso il nodo rappresentante del blocco a cui tale nodo da cancellare appartiene.
				// Utilizzo il campo "tv" di "A->Vertex[].e[]", che indica lo stato a cui punta l'arco
				// considerato, in quanto non servira' piu'
				if (G[ A->Vertex[i].e[j].tv ].nextInBlock == BIS_DA_CANCELLARE)
					A->Vertex[i].e[j].tv = Q[ G[A->Vertex[i].e[j].tv].block ].firstNode;
			}
		}
	}
}

/*\***paigeTarjan.cpp****/
//initialise paige and Tarjan

int bisimulation::InitPaigeTarjan()
{
	BIS_indexType i, l, end, temp;
	struct counter *cxS;
	struct adjList *adj;

	//initialisation of the graph (G,Q,X)
	for (l = 0; l != BIS_NIL; l = temp) { //for each label block
		temp = X[l].nextXBlock;
		if (temp == BIS_NIL)
			end = l;
		Q[l].prevBlock = X[l].prevXBlock;
		Q[l].nextBlock = X[l].nextXBlock;
		Q[l].firstNode = X[l].firstBlock;
		Q[l].superBlock = 0;
		//compute Q[].size
		Q[l].size = 0;
		for (i = X[l].firstBlock; i != BIS_NIL; i = G[i].nextInBlock)//for each node
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
		adj = G[i].adj;
		//to avoid the creation of a counter set to zero
		if (adj == NULL)
			continue;
		cxS = new struct counter;
		cxS->value = 0;
		while (adj != NULL) {
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

void bisimulation::PaigeTarjan()
{
	BIS_indexType S, S1; //pointer to the X-Blocks S and S1
	BIS_indexType B, S_B; //pointer to the Q-Blocks B and S-B
	BIS_indexType oldD, newD; //old and new block of x belonging to E-1(B)
	struct adjList_1 *adj;
	struct counter *cxS;
	BIS_indexType x, y, d, e;

	while (C != BIS_NIL) {

		/*Step 1(select a refining block) & Step 2(update X)*/
		//select some block S from C
		S = C;
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
		S1 = freeXBlock;
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
			/*WE DO NOT FREE THE BLOCK S: the XBlock still exists but it is
			not in the chain of C*/
		}

		/*Step 3(compute E-1(B))*/
		/*by scanning the edges xEy such that y belongs to B
		and adding each element x in such an edge to E-1(B),
		if it has not already been added.
		Duplicates are suppressed by marking elements: B_1
		Side effect: copy the elements of B in B1
		During the same scan, compute count(x,B)=count(x,S1) because S1={B};
		create a new counter record and make G[x].countxB point to it*/
		y = b1List = Q[B].firstNode;
		b_1List = BIS_NIL;
		while (y != BIS_NIL) { //for each y belonging to B
			B1[y] = G[y].nextInBlock; //copy the elements of B in B1
			adj = G[y].adj_1;
			while (adj != NULL) { //for each node in the adj_1 of y
				x = adj->node;
				if (B_1[x] == numberOfNodes) {
					//node not already added to E-1(B)
					B_1[x] = b_1List;
					b_1List = x;
					//create a new counter: it is pointed by G[x].countxB    /*1*/
					cxS = new struct counter;
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
		while (x != BIS_NIL) { //for each x belonging to E-1(B)
			//determine the block D of Q containing it
			oldD = G[x].block; //index of D (old block of x)
			//and create an associated block D1 if one does not already exist
			if (splitD[oldD] == numberOfNodes) {
				//block D not already split
				splitD[oldD] = dList;
				dList = oldD;
				//create a new block D1
				if (freeQBlock == BIS_NIL) { //check for free space in memory
					if (QBlockLimit == BIS_MAXINDEX) {
						std::cout << "memory limit";
						std::cin >> x;
						exit(-1);
					}
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
		}//endwhile

		//dList points to the list of new blocks splitD
		d = dList;
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
					we did not free its space and now we can re-use it!!*/
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
		while (y != BIS_NIL) { //for each y belonging to B1
			adj = G[y].adj_1;
			while (adj != NULL) { //for each node in the adj_1 of y -> scan xEy, y in B
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
				if (freeQBlock == BIS_NIL) { //check for free space in memory
					if (QBlockLimit == BIS_MAXINDEX) {
						std::cout << "memory limit";
						std::cin >> x;
						exit(-1);
					}
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
		}//endwhile

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
					we did not free its space and now we can re-use it!!*/
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
		while (y != BIS_NIL) { //for each y belonging to B
			adj = G[y].adj_1;
			while (adj != NULL) {//for each node in the adj_1 of y -> scan xEy, y in B
				x = adj->node;
				cxS = adj->adj->countxS;
				if (cxS->value != 1) {
					(cxS->value)--;
					adj->adj->countxS = G[x].countxB;
				} else { //count(x,S) becomes zero
					//make xEy point to count(x,B)
					adj->adj->countxS = G[x].countxB;
					//delete count(x,S)
					delete cxS;
				}
				adj = adj->next;
			}
			x = y;
			y = B1[y];
			//re-initialisation of B1
			B1[x] = numberOfNodes;
		}
	}//end while
}


/*\***fastBisimulation.cpp****/

//FUNCTIONS
//construct the inverse of the graph represented by adjacency list

void bisimulation::Inverse()
{
	BIS_indexType i;
	struct adjList *adj;
	struct adjList_1 *a;

	for (i = 0; i < numberOfNodes; i++) {
		adj = G[i].adj;
		while (adj != NULL) {
			adj->countxS = NULL; //initialisation to avoid inconsistent pointers
			a = new struct adjList_1;
			a->node = i;
			a->next = G[adj->node].adj_1;
			G[adj->node].adj_1 = a;
			a->adj = adj;
			adj = adj->next;
		}
	}
}

/* modified strongly connected component;
   the first visit is for G-1 and the second for G;
   Q[].prevBlock represents the color of the nodes during the DFS visit,
   Q[].superBlock represents the forefathers in the SCC,
   Q[].firstNode represents the finishing time of the first DFS visit in SCC()*/
void bisimulation::Rank()
{
	BIS_indexType i, temp, r;

	//initialisation of the nodes
	for (i = 0; i < numberOfNodes; i++) {
		//color is BIS_WHITE
		Q[i].prevBlock = BIS_WHITE;
		//i is forefather of itself
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
	for (i = numberOfNodes - 1; i >= 0; i--) { /*(1)*/
		temp = Q[i].firstNode; //node that has to be visited
		if ((Q[temp].prevBlock) == BIS_BLACK) {
			SecondDFS_visit(temp, temp);
			//to normalize the ranks:
			r = G[temp].rank;
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
Q[i].superBlock == i => i is a forefather,
Q[i].superBlock == x != i => x is the forefather of i;
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
  i is the node being visited*/
void bisimulation::FirstDFS_visit(BIS_indexType i)
{
	BIS_indexType j;
	struct adjList_1 *adj_1;

	//visit G-1
	adj_1 = G[i].adj_1;
	Q[i].prevBlock = BIS_GRAY;
	while (adj_1 != NULL) {
		j = adj_1->node;
		if (Q[j].prevBlock == BIS_WHITE)
			FirstDFS_visit(j);
		adj_1 = adj_1->next;
	}
	Q[i].prevBlock = BIS_BLACK;
	//store the finishig time of the first DFS visit
	Q[t++].firstNode = i;
	/*the node i is stored at the index t corresponding to its finishing time
	 to avoid re-ordering*/
}

/*modified version of DFS_visit to optimise the computation of Rank();
  secondDFS_visit visits G and stores the forefather of the nodes in
  Q[].superBlock;
  i is the node being visited, ff is its forefather;
  remember that the meaning of the colors is inverted: BIS_WHITE <--> BIS_BLACK*/
void bisimulation::SecondDFS_visit(BIS_indexType i, BIS_indexType ff)
{
	BIS_indexType j;
	struct adjList *adj;
	//stores the temporary value of the rank computed from the children of a node
	BIS_indexType tempRank;

	//visit G
	Q[i].prevBlock = BIS_GRAY;
	G[i].rank = -1;
	tempRank = -1;

	/*if a node is not a forefather, it is sure that it is not well-founded;
	the forefathers are well-founded only if all their children are well-founded
	(and it will result during the visit)*/
	if (i != ff)
		G[i].WFflag = false;

	adj = G[i].adj;
	while (adj != NULL) {
		j = adj->node;
		//it is the first time we visit j that is in the same SCC of i
		if (Q[j].prevBlock == BIS_BLACK) {
			Q[j].superBlock = ff;
			SecondDFS_visit(j, ff);
		}
		if (Q[i].superBlock == Q[j].superBlock) { //SCC(i) == SCC(j)
			//the rank should not increase
			tempRank = G[j].rank;
			//i is non-well-founded
			G[i].WFflag = false;
		} else { //SCC(i) != SCC(j)
			if (G[j].WFflag == true) //j well-founded
				//the rank should increase of 1
				tempRank = G[Q[j].superBlock].rank + 1;
			else { //j not well-founded
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
	}//endwhile

	//well-founded nodes have even rank
	if (G[i].WFflag == true)
		(G[i].rank)++;
	Q[i].prevBlock = BIS_WHITE;
}

/*it return a exit code: 0 means proceed the computation with
FastBisimulationAlgorithm()*/
int bisimulation::InitFBA()
{
	BIS_indexType i, temp;

	//to normalize the ranks
	for (i = 1; i < numberOfNodes && Q[i].size != 0; i++)
		Q[i].size = Q[i].size + Q[i - 1].size;
	maxRank = Q[i - 1].size - 1;
	/*the technique used to nomalised the ranks is the well-know one,
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
		temp = Q[Q[i].superBlock].nextBlock; //is the rank of the forefather
		if (temp == -1 || temp == 0)
			G[i].rank = temp;
		else if ((temp % 2) == 0)
			G[i].rank = Q[temp / 2 - 1].size;
		else
			G[i].rank = Q[temp / 2 - 1].size + 1;
		/*the technique used to nomalised the ranks is the well-know one,
		used for example in counting sort*/

		//clearing for the next initialisation phase
		Q[i].firstNode = BIS_NIL;
		B1[i] = numberOfNodes;
		B_1[i] = numberOfNodes;
		splitD[i] = numberOfNodes;
	}

	BIS_indexType j, newBlock, tmpi;


	//initialization of the limit of the arrays
	QBlockLimit = numberOfNodes;
	freeQBlock = maxRank + 2; //one for each rank included rank -1 and 0

	//initialisation of G
	/*we are going to scan the nodes regarding the label-blocks
	so we can build the bisimulation structures that are composed by
	X-blocks and Q-blocks regarding both ranks and labels*/
	for (BIS_indexType l = 0; l != BIS_NIL; l = X[l].nextXBlock) //for each label block
		for (i = X[l].firstBlock; i != BIS_NIL; i = tmpi) {//for each node
			//the node i has to be inserted in the block j corresponding to rank j-1
			tmpi = G[i].nextInBlock;
			j = G[i].rank + 1;
			if (Q[j].firstNode == BIS_NIL) { //the block is empty: add i
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
				newBlock = freeQBlock++; //index of the new block

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
				for (BIS_indexType k = Q[newBlock].firstNode; k != BIS_NIL; k = G[k].nextInBlock)
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
	for (i = maxRank + 2; i < freeQBlock; i++) { //freeQBlock < nON
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
	struct adjList *adj;
	struct counter *cxS;
	for (i = 0; i < numberOfNodes; i++) {
		adj = G[i].adj;
		//to avoid the creation of a counter set to zero
		if (adj == NULL)
			continue;
		cxS = new struct counter;
		cxS->value = 0;
		while (adj != NULL) {
			(cxS->value)++;
			//each edge xEy contains a pointer to count(x,U);
			//remember that each edge y(E-1)x contains a pointer to the edge xEy!
			adj->countxS = cxS;
			adj = adj->next;
		}
		cxS->node = i;
	}

	/*divide the edges: borderEdges[i] stores the edges going to i from nodes
	of different rank and G[i].adj_1 stores only the edges going to i from
	nodes having the same rank*/
	struct adjList_1 *adj_1, *a, *b, *next;

	for (i = 0; i < numberOfNodes; i++) {
		j = G[i].rank;
		adj_1 = G[i].adj_1;
		a = NULL;
		b = NULL;
		while (adj_1 != NULL) {
			next = adj_1->next;
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
//It analysed only the nodes of Rank rank that are in the Xblock C.

void bisimulation::PaigeTarjan(BIS_indexType rank)
{
	BIS_indexType S, S1; //pointer to the X-Blocks S and S1
	BIS_indexType B, S_B; //pointer to the Q-Blocks B and S-B
	BIS_indexType oldD, newD; //old and new block of x belonging to E-1(B)
	struct adjList_1 *adj;
	struct counter *cxS;
	BIS_indexType x, y, d, e, super;

	//REMINDER: XBlock that are not in C but have Rank rank
	rankPartition = BIS_NIL;

	while (C != BIS_NIL && rank == G[Q[X[C].firstBlock].firstNode].rank) {

		/*Step 1(select a refining block) & Step 2(update X)*/
		//select some block S from C
		S = C;
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
		S1 = freeXBlock;
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
		if it has not already been added and if has the same rank of y.
		REMINDER: the adjacency list of G-1 has been cut from the edges to the
		nodes having different rank, hence we don't need to do anything special!
		Duplicates are suppressed by marking elements: B_1
		Side effect: copy the elements of B in B1
		During the same scan, compute count(x,B)=count(x,S1) because S1={B};
		create a new counter record and make G[x].countxB point to it*/
		y = b1List = Q[B].firstNode;
		b_1List = BIS_NIL;
		while (y != BIS_NIL) { //for each y belonging to B
			B1[y] = G[y].nextInBlock; //copy the elements of B in B1
			adj = G[y].adj_1;
			while (adj != NULL) { //for each node in the adj_1 of y
				x = adj->node;
				if (B_1[x] == numberOfNodes) {
					//node not already added to E-1(B)
					B_1[x] = b_1List;
					b_1List = x;
					//create a new counter: it is pointed by G[x].countxB    /*1*/
					cxS = new struct counter;
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
		while (x != BIS_NIL) { //for each x belonging to E-1(B)
			//determine the block D of Q containing it
			oldD = G[x].block; //index of D (old block of x)
			//and create an associated block D1 if one does not already exist
			if (splitD[oldD] == numberOfNodes) {
				//block D not already split
				splitD[oldD] = dList;
				dList = oldD;
				//create a new block D1
				if (freeQBlock == BIS_NIL) { //check for free space in memory
					if (QBlockLimit == BIS_MAXINDEX) {
						std::cout << "memory limit";
						std::cin >> x;
						exit(-1);
					}
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
		}//endwhile

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
				since we are computing within a particular rank (adjList_1 of G has been
				cut) D and hence D1 have Rank rank (and also super has) we just need to
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
		while (y != BIS_NIL) { //for each y belonging to B1
			adj = G[y].adj_1;
			while (adj != NULL) { //for each node in the adj_1 of y -> scan xEy, y in B
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
				if (freeQBlock == BIS_NIL) { //check for free space in memory
					if (QBlockLimit == BIS_MAXINDEX) {
						std::cout << "memory limit";
						std::cin >> x;
						exit(-1);
					}
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
		}//endwhile

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
				since we are computing within a particular rank (adjList_1 of G has been
				cut) D and hence D1 have Rank rank (and also super has) we just need to
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
		while (y != BIS_NIL) { //for each y belonging to B
			adj = G[y].adj_1;
			while (adj != NULL) { //for each node in the adj_1 of y -> scan xEy, y in B
				x = adj->node;
				cxS = adj->adj->countxS;
				if (cxS->value != 1) {
					(cxS->value)--;
					adj->adj->countxS = G[x].countxB;
				} else { //count(x,S) becomes zero
					//make xEy point to count(x,B)
					adj->adj->countxS = G[x].countxB;
					//delete count(x,S)
					delete cxS;
				}
				adj = adj->next;
			}
			x = y;
			y = B1[y];
			//re-initialisation of B1
			B1[x] = numberOfNodes;
		}
	}//end while
}

//split computes a single split as regards the single block B.

/*The function consists of step 3 and step 4 of PaigeTarjan, even if there are
some differences: once used for the split, B is not anymore necessary; so B1, S1
and the counters are not computed; since we are interested in the edges between
nodes of different rank we scan borderEdges[] instead of G[].adj_1*/
void bisimulation::Split(BIS_indexType B)
{
	BIS_indexType oldD, newD; //old and new block of x belonging to E-1(B)
	struct adjList_1 *adj;
	BIS_indexType x, y, d, e;

	/*Step 3(compute E-1(B))*/
	/*by scanning the edges xEy such that y belongs to B
	and adding each element x in such an edge to E-1(B),
	if it has not already been added.
	REMINDER: we have to scan only the edges between nodes
	of different rank
	Duplicates are suppressed by marking elements: B_1*/
	y = b1List = Q[B].firstNode;
	b_1List = BIS_NIL;
	while (y != BIS_NIL) { //for each y belonging to B
		adj = borderEdges[y]; //instead G[y].adj_1;
		while (adj != NULL) { //for each node in the adj_1 of y
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
	while (x != BIS_NIL) { //for each x belonging to E-1(B)
		//determine the block D of Q containing it
		oldD = G[x].block; //index of D (old block of x)
		//and create an associated block D1 if one does not already exist
		if (splitD[oldD] == numberOfNodes) {
			//block D not already split
			splitD[oldD] = dList;
			dList = oldD;
			//create a new block D1
			if (freeQBlock == BIS_NIL) { //check for free space in memory
				if (QBlockLimit == BIS_MAXINDEX) {
					std::cout << "memory limit";
					std::cin >> x;
					exit(-1);
				}
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
	}//endwhile

	//dList points to the list of new blocks splitD
	d = dList;
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

		e = d;
		d = splitD[d];
		//re-initialisation of splitD
		splitD[e] = numberOfNodes;
	}
}

//compute FastBisimulationAlgorithms

void bisimulation::FastBisimulationAlgorithm()
{
	BIS_indexType i, l, rP;

	/*VERY IMPORTANT:
	before computing the minimisation for that rank component,
	each rank component is rappresented by only ONE XBlock
	and eventually more than one QBlock*/

	/* for each Rank i: {R(ij)=P&T(Ri); for each j: split(Rij);}
	Exceptin 1 : remember that well-founded blocks can be refined only by blocks
	having lower rank (there are no edges between nodes in the same rank
	component). Hence we do not need to compute PT or PTB on this kind of blocks.
	Exception 2: also if there is only a QBlock representing a rank component it
	is not necessary to compute PT or PTB on it*/
	for (i = -1; i <= maxRank; i++) {
		/*in the case of the Exceptions above, we remove the XBlock, representing
		the rank component, from C and store it in rankPartition, ready to the
		succesive sequence of simple splits */
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

		/*the sub-graph of rank i is minimised and all the blocks are in the chain
		pointed by rankPartition*/
		if (i != maxRank) //i==maxRak means last rank and no more split
			while (rankPartition != BIS_NIL) {
				for (l = X[rankPartition].firstBlock; l != BIS_NIL; l = Q[l].nextBlock)
					Split(l);
				//free the XBlock just used in the split
				rP = X[rankPartition].nextXBlock;
				X[rankPartition].nextXBlock = freeXBlock;
				freeXBlock = rankPartition;
				X[rankPartition].prevXBlock = BIS_NIL;
				X[rankPartition].firstBlock = BIS_NIL;
				rankPartition = rP;
			}
	}
}

/*----------------------------------------------------------------------------*/

bool bisimulation::MinimizeAutomaPT(automa *A)
{
	FillStructures(A);
	Inverse();


	if (InitPaigeTarjan() == 0) {
		PaigeTarjan();


		GetMinimizedAutoma(A);
		return true;

	}
	return false;

}

bool bisimulation::MinimizeAutomaFB(automa *A)
{
	//std::cerr << "\nDEBUG: IN MINIMIZE\n" << std::flush;
	FillStructures(A);
	Inverse();

	Rank();
	if (InitFBA() == 0) {
		FastBisimulationAlgorithm();

		GetMinimizedAutoma(A);
		return true;

	}
	return false;

}

bisimulation::bisimulation()
{
	G = new graph[BIS_MAXINDEX];
	Q = new qPartition[BIS_MAXINDEX];
	X = new xPartition[BIS_MAXINDEX];

	borderEdges = new adjList_1*[BIS_MAXINDEX];

	B1 = new BIS_indexType[BIS_MAXINDEX]; //copy of B
	B_1 = new BIS_indexType[BIS_MAXINDEX]; //list to maintain E-1(B) and E-1(S-B)
	splitD = new BIS_indexType[BIS_MAXINDEX]; //list of split blocks
}

bisimulation::~bisimulation(){

	
		delete []G;
	delete []Q;
	delete []X;
//		delete []borderEdges;
//
//
//	 delete []B1;
//	 delete []B_1;
//	 delete []splitD;
}
