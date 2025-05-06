#!/usr/bin/env bash
#This script prints out the time for executing a plan, giventhe actions, on the same domain with the two version of EFP
#USAGE EXAMPLE (from the project root folder): ". scripts/givenplan_comparison.sh exp/paper/CC/CC_2_2_3.txt right_a left_b a_check_3"
#	where:
#			"exp/paper/CC/CC_2_2_3.txt" is the path to the domain name
#			"right_a left_b a_check_3" are the names of three actions (the number of actions can be any n > 0)


DOMAIN="$1" # Name of the domain
shift            # Shift all arguments to the left (original $1 gets lost)
ARGUMENTS=("$@") # Rebuild the array with rest of arguments

representations=("KRIPKE_OPT" "KRIPKE" "POSS" "OLD");
for repr in "${representations[@]}"; do
    if [ "$repr" = "OLD" ]; then 
        EFP_ICAPS_old/cpa+ $DOMAIN -results_file           -e "${ARGUMENTS[@]}";
    else
        bin/efp.out        $DOMAIN -results_file -st $repr -e "${ARGUMENTS[@]}";
    fi;
done
