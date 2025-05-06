#!/usr/bin/env bash
#This script prints out the time for executing a plan on the same domain with the different version of EFP
#USAGE EXAMPLE (from the project root folder): ". scripts/comparison_graphviz.sh exp/paper/CC/CC_2_2_3.txt"
#	where:
#			"exp/paper/CC/CC_2_2_3.txt" is the path to the domain name

representations=("KRIPKE_OPT" "KRIPKE" "POSS" "OLD");

for repr in "${representations[@]}"; do
    if [ "$repr" != "OLD" ]; then 
        bin/efp.out        $@ -debug -st $repr;
    fi;
    #sleep 5s;
done
