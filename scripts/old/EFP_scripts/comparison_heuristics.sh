#!/usr/bin/env bash
#This script prints out the time for executing a plan on the same domain with all the heuristics of EFP
#USAGE EXAMPLE (from the project root folder): ". scripts/comparison_heuristics.sh"

#REMOVE OLD COPIES

file=$1

if [[ -f out/EFP_comparison/findingplan/"${file##*/}" ]] ; then
    rm out/EFP_comparison/findingplan/"${file##*/}"
fi

TIMEOUT="600";

heuristics=("NONE" "S_PG" "L_PG" "C_PG" "SUBGOALS"); 

for heur in "${heuristics[@]}"; do
  timeout $TIMEOUT bin/efp.out $@ -results_file -check_visited -st POSS -h $heur > findingplan_comparison.tmp
  exit_status=$?;

	if [[ $exit_status -eq 124 ]]; then

        TOPRINT="EFP Version 2.0 (on POSS and fast Bisimulation) with $heur and VISTED-STATE TIMED-OUT"
        echo -e "$TOPRINT" >> out/EFP_comparison/findingplan/"${file##*/}";
	else
		grep -w "Executed actions:" findingplan_comparison.tmp >> "out/EFP_comparison/findingplan/${file##*/}";
    grep -w "Plan Length:" findingplan_comparison.tmp  >> "out/EFP_comparison/findingplan/${file##*/}";
    grep -w "Expanded Nodes:" findingplan_comparison.tmp  >> "out/EFP_comparison/findingplan/${file##*/}";

	fi;

    echo $'\n' >> "out/EFP_comparison/findingplan/${file##*/}";

    sleep 2s;
done;

rm findingplan_comparison.tmp;
