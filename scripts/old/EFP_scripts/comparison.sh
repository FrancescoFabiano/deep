#!/usr/bin/env bash
#This script prints out the time for executing a plan on the same domain with the two version of EFP
#USAGE EXAMPLE (from the project root folder): ". scripts/comparison.sh exp/paper/CC/CC_2_2_3.txt"
#	where "exp/paper/CC/CC_2_2_3.txt" is the path to the domain name


TIMEOUT="30m" 


#REMOVE OLD COPIES
if [ -f out/EFP_comparison/findingplan/"${@##*/}" ] ; then
    rm out/EFP_comparison/findingplan/"${@##*/}"
fi

representations=("KRIPKE_OPT" "KRIPKE" "POSS" "OLD");

for repr in "${representations[@]}"; do
    if [ "$repr" = "OLD" ]; then 
        timeout $TIMEOUT EFP_ICAPS_old/cpa+ $@ -results_file > findingplan_comparison.tmp
        exit_status=$?;
    else
        timeout $TIMEOUT bin/efp.out $@ -results_file -st $repr -check_visited > findingplan_comparison.tmp
        exit_status=$?;
    fi;
    if [[ $exit_status -eq 124 ]]; then
        
        TOPRINT="EFP Version ";

        if [ "$repr" = "OLD" ]; then 
            TOPRINT+="1.0 (on KRIPKE with STRANDARD transition function)"
        fi;
        if [ "$repr" = "KRIPKE_OPT" ]; then
            TOPRINT+="2.0 (on KRIPKE with OPTIMIZED transition function)"
        fi;
        if [ "$repr" = "KRIPKE" ]; then
            TOPRINT+="2.0 (on KRIPKE with STANDARD transition function)"
        fi;
        if [ "$repr" = "POSS" ]; then
            TOPRINT+="2.0 (on POSS with STANDARD transition function)"
        fi;
   
   		if [ "$repr" != "OLD" ]; then
            TOPRINT+=" with VISTED-STATE"
        fi;
        TOPRINT+=" TIMED-OUT"
        echo -e "$TOPRINT" >> out/EFP_comparison/findingplan/"${@##*/}";

    else

        grep -w "Executed actions:" findingplan_comparison.tmp >> out/EFP_comparison/findingplan/"${@##*/}";
        grep -w "Plan Length:" findingplan_comparison.tmp  >> out/EFP_comparison/findingplan/"${@##*/}";

    fi;

    echo $'\n' >> out/EFP_comparison/findingplan/"${@##*/}";
    
    sleep 5s;
done;

rm findingplan_comparison.tmp;