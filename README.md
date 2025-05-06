# EFP

## Thesis - Related
The repository, for a brief period of time, will store previous versions of the planner that match the implementation of some foundamental features.
The planner, at the moment, is not completly able to manage the various versions.
These versions will be merged in the next future and we will provide a unique solver that would allow, through parameters, to try different configurations.

As a temporary solution we store in the folder "Thesis Code" four archives:
- EFP2-0.zip: Contains the code of EFP 2.0 presented at (Fabiano et al. 2020)
- EFP2-1.zip: Contains the code of EFP 2.1 that is EFP 2.0 optimized -- this version works only with Possibilities and also allows to play with small variation of the classical actions
- EFP-att.zip: Contains the code where EFP is able to deal with agents attitudes (Fabiano et al. 2021) -- the example files contains the line to execute to generate the graphical states representation
- EFP-customEve.zip: Contains the code where it is possible to play with custom event model
- PLATO.zip: The ASP planner based on (Burigana et al. 2020)

## Goal
Realize a scalable Epistemic Forward Planner that bases its concept on different states representations such as:
- [x] *Kripke structure*
- [x] *possibilities*
- [ ] *OBDDs*

Temporally disabled Kripke and OBDD

## Current situation:
Implemented EFP v. 2.1 that uses templatic e-States representations with relative templatic transition function and data structure based on bitset.
The planner is based on several scientific publications where the different semantics and transition functions are described.
In particular we have the following configurations:
- **eState:** *Kripke Structures*; **transition function:** *event based updates---mA<sup>\*</sup>* (Baral et al. 2015, Le et al. 2018);
- **eState:** *Kripke Structures*; **transition function:** *optimized event based updates* (Fabiano et al. 2020);
- **eState:** *Possibilities*; **transition function:** *Iterative transition function---mA<sup>p</sup>* (Fabiano et al. 2019, Fabiano et al. 2020).
- **eState:** *Possibilities*; **transition function:** *Iterative transition function---mA<sup>p</sup> that consider agents attitudes for complex epistemic reasoning (e.g., lies, misconception etc.)* (Under review).


Moreover the planner admits templatic heuristics usage.
At the moment we implemented:
- a complete version of the *Epistemic planning graph* introduced in (Le et al. 2018);
- *number of satisfied goals* that possibly expands the original goal for a better scalability;
- BFS, DFS and DFS Iterative searches.

  
## Future works and some ideas
- Think about OBDDs.
- More Heuristics.
- Is announcement with false beliefs an ontic? It creates the world if it didn't exist. old semantic + ontic update for fully observant (sensing the same)
- If **ag_i** sees as partial **ag_y** announcing **phi** and **ag_i** thought that **ag_y** did not know **phi** how should **ag_i** react?
- If **ag_i** sees as partial **ag_y** announcing **f** and **ag_i** knows **f** then
	- **ag_i** believes that **ag_y** knows **f**; or
	- **ag_i** believes that **ag_y** knows (**f OR -f**).
- Objective vs subjective common knowledge;
- announcement/sensing of Belief formulae;
- ontic that remove uncertanty;
- static laws (the same as biased info?);
- intial state;
- false beliefs correction.


- ethic constraints.

### Requirements
- flex
- bison
- texlive-font-utils
- graphviz
- boost

### Usage
- **make**: to compile the planner.
- **make doxygen**: to compile the planner and the documentation (to check it open doxygen/Docs/html/index.html).
- **make clean**: removes all compilation-generated files.
- **make clean_build**: removes all compilation-generated files.
- **make clean_out**: removes all the file in out/ (the pdf visualization of the states).
- **make clear**: executes both **clean_build** and **clean_out**.
- **make fresh**: executes **clear** and also removes doxygen documentation.
- **make old**: cleans and compile the old version (1.0) of EFP.
- **make all**: executes **make doxygen** and **make_old**.
	
	
### Extras
- The repository also includes several scripts to help in the testing/debugging process. These are located inside the folder *scripts*.
- All the utilized beanchmark are preserved in the folder *exp*.
- The folder *ICAPS_EFP_OLD* contains EFP v. 1.0 (introduced in (Le et al. 2018)) for comparison and testing.
- The repository contains DockerFile for create container and images to compile and run
- The repository include project settings and various profile to works on ide (CLion), profiles available: Build, Build Clean, Execute, Execute and Build
- Now can debug at run time with debugger (Clion), profiling with CLion profiler integrated (Callstack, Memory, CPU usage and More)

#### Bibliography
Baral, C.; Gelfond, G.; Pontelli, E.; and Son, T. C. 2015.
An action language for multi-agent domains: Foundations.
CoRR abs/1511.01960.


Burigana, A., Fabiano, F., Dovier, A. & Pontelli, E. 2020.
Modelling Multi-Agent Epistemic Planning in ASP.
Theory and Practice of Logic Programming.


Fabiano, F.; Burigana, A.; Dovier, A.; and Pontelli, E. 2020.
EFP 2.0: A Multi-Agent Epistemic Solver with Multiple e-State Representations.
In Proceedings of the 30th International Conference on Automated Planning and Scheduling.


Fabiano, F., Burigana, A., Dovier, A., Pontelli, E. & Son, T. C. 2021.
Multi-agent Epistemic Planning with Inconsistent Beliefs, Trust and Lies.
In Proceedings of the 18th Pacific Rim International Conference on Artificial Intelligence.


Fabiano, F.; Riouak, I.; Dovier, A.; and Pontelli, E. 2019.
Non-well-founded set based multi-agent epistemic action language.
In Proceedings of the 34th Italian Conference on Computational Logic.


Le, T.; Fabiano, F.; Son, T. C.; and Pontelli, E. 2018.
EFP and PG-EFP: Epistemic forward search planners in multiagent domains.
In Proceedings of the Twenty-Eighth International Conference on Automated Planning and Scheduling.
