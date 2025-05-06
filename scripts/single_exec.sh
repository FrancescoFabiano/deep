#!/usr/bin/env bash
#This script prints out the time for executing a plan on the same domain with the two version of EFP
#USAGE EXAMPLE (from the project root folder): ". scripts/comparison.sh exp/paper/CC/CC_2_2_3.txt"
#	where "exp/paper/CC/CC_2_2_3.txt" is the path to the domain name

#REMOVE OLD COPIES
if [ -f out/EFP_comparison/findingplan/"${@##*/}" ] ; then
    rm out/EFP_comparison/findingplan/"${@##*/}"
fi

TIMEOUT="30m";

        timeout $TIMEOUT bin/efp.out $@ -results_file -check_visited -st POSS > findingplan_comparison.tmp
        exit_status=$?;

        if [[ $exit_status -eq 124 ]]; then

            TOPRINT="EFP Version on POSS with VISTED-STATE TIMED-OUT"
            echo -e "$TOPRINT" >> out/EFP_comparison/findingplan/"${@##*/}";

	else

            grep -w "Executed actions:" findingplan_comparison.tmp >> out/EFP_comparison/findingplan/"${@##*/}";
            grep -w "Plan Length:" findingplan_comparison.tmp  >> out/EFP_comparison/findingplan/"${@##*/}";
	fi;


    echo $'\n' >> out/EFP_comparison/findingplan/"${@##*/}";

    sleep 10s;
    
rm findingplan_comparison.tmp;
