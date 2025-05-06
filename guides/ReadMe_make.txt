Generazione Progetto CLion, per debuggare durante l'esecuzione del codice a run time e compilazione attraverso
Makefile senza dover modificare il progetto per utilizzare CMake o altro.

-Copiare la cartella settingsProject nella root del progetto (../EFP)
-Sostituire il nome della cartella: mv settingsProject .idea
-Aprire la root folder del progetto con CLion.
-Modifica in alto a dx le configurazione CustomAPP e Makefile Attraverso il menu "EditConfiguration" che comparirà selezionando il menu a tendina
-Inserire le specifiche PATH (Valuterò di inserire le path dinamiche) con le path corrette.
-Eseguira un debug (icona insetto verde) e testare la buona riuscita del load del progetto.
