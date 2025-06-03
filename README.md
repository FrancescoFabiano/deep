# deep

## TODOs

- [ ] Create set of test to run on push to validate the push itself (use timeouts)
    - [ ] Randomly solve n problems from different domains and check if the solution is the same as the expected one
    - [ ] Do the same thing but activate bisimulation and also create a check that for each state generated after
      bisimulation a set of randomly generated belief formulae holds and also prints out the compression ratio (
      especially at the beginning to empirically evaluate bisimulation which has been adapted from hold code)
    - [ ] Do the same but now with heuristics and other planner configurations
    - [ ] Adapt hold scripts and make them more concise and directly generate the latex
- [ ] Fix the ReadME
- [ ] Check the doxygen and structure for consistency
- [ ] Create a docker version maybe
- [ ] Provide some support in the compilation tools
- [ ] Add Francois partial state
- [ ] Add EPDDL support
- [ ] Create generic update function with event model but maybe keep optimized solving strategies/heuristics for mA* (
  planning graph)
- [ ] Start to test against other planning types (classical, conformant etc.)
- [ ] Enrich the search strategies with Local Search, MonteCarlo, add Delphic from Buri's, Christian's etc.

## Goal

Realize a scalable Planner that tackles Multi-Agent Epistemic Planning exploiting the full power of Dynamic Epistemic
Logic.
This tool is able to reason over belief/knoweldge relations, i.e., over infomration flows.

## Current situation:

Moreover the planner admits templatic heuristics usage.
At the moment we implemented:

- a complete version of the *Epistemic planning graph* introduced in (Le et al. 2018);
- *number of satisfied goals* that possibly expands the original goal for a better scalability;
- BFS, DFS and DFS Iterative searches.

## Future works and some ideas

- Think about OBDDs.
- More Heuristics.
- Is announcement with false beliefs an ontic? It creates the world if it didn't exist. old semantic + ontic update for
  fully observant (sensing the same)
- If **ag_i** sees as partial **ag_y** announcing **phi** and **ag_i** thought that **ag_y** did not know **phi** how
  should **ag_i** react?
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

- The repository also includes several scripts to help in the testing/debugging process. These are located inside the
  folder *scripts*.
- All the utilized beanchmark are preserved in the folder *exp*.
- The folder *ICAPS_EFP_OLD* contains EFP v. 1.0 (introduced in (Le et al. 2018)) for comparison and testing.
- The repository contains DockerFile for create container and images to compile and run
- The repository include project settings and various profile to works on ide (CLion), profiles available: Build, Build
  Clean, Execute, Execute and Build
- Now can debug at run time with debugger (Clion), profiling with CLion profiler integrated (Callstack, Memory, CPU
  usage and More)

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
