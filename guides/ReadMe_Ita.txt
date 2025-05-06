### Example
- Stampa dei possibili parametri
	- bin/efp.out
- Esecuzione con possibilities come e-state
	- bin/efp.out path_relativo/nome_dominio -st POSS

- Esecuzione con possibilities come e-state e controllo stati visitati (Configurazione più veloce)
	- bin/efp.out path_relativo/nome_dominio -st POSS -check_visited

- Test di una lista di azioni (-e parameter):
	- bin/efp.out path_relativo/nome_dominio -st POSS -e act1, act2, act3
	
- Test di una lista di azioni (-e parameter)poiù print grafico:
	- bin/efp.out path_relativo/nome_dominio -st POSS -debug -e act1, act2, act3

- Esecutione con Kripke Structure con controllo stati visitati
	- bin/efp.out path_relativo/nome_dominio -check_visited

- Esecuzione con Kripke Structure con controllo stati visitati e utilizzo di bisimulation per ridurre gli stati
	- bin/efp.out path_relativo/nome_dominio -check_visited -bis PT


### Make
- make -> fa semplicemente il make
- make clean -> pulisce file di compilazione
- make clean_build -> pulisce file di compilazione
- make clean_out -> pulisce file di output
- make clear -> pulisce file di compilazione e di output
- make old -> compila la versione vecchia del planner (in "EFP_ICAPS_old/")
- make doc -> genera il doxygen
- make fresh -> elimina tutti i dile generati (doxygen, output, build e i vari .tmp che possono essersi creati)