#!/usr/bin/env bash
#Just Run
RESPATH="ICAPS_res_bis/";
DOMAIN_EXT=".txt"; # Name of the domain

#mv $OUTPATH_FINDINGPLAN $OUTPATH$DOMAIN_NAME$DOMAIN_EXT;

echo -e "\nExecution of efp.out in all the folders in" $RESPATH "\n"
#echo "NAME, INDEX, KRIPKE STANDARD, K-STD V,  KRIPKE OPT, K-OPT V, POSS, POSS V, OLD" > "$RESPATH/graphres.csv";
INDEX=1;
for dir in $RESPATH*; do
    if [ -d "$dir" ]; then

        echo -e "Currently working on the $(basename $dir) directory";
        #mkdir -p $OUTPATH"$(basename $dir)"

		for dirNest in $dir"/"*; do
			if [ -d "$dirNest" ]; then
			    echo -e "Currently working on the $(basename $dirNest) directory";
				ITERATION=0;
				AVG_PT=0;
				AVG_FB=0;

				for file in $(find "$dirNest"/ -type f -name *$DOMAIN_EXT); do
					if [[ "$(basename $file)" == *"exec_"* ]]; then
						#mv $OUTPATH_FINDINGPLAN"$(basename $file)" $OUTPATH"$(basename $dir)"/"$(basename $file)"
						while read line; do
							#if [[$line = "EFP"*]]; then
							   [ -z "$line" ] && continue;
							   if [[ $line == *"TIMED-OUT"* ]]; then
									num="9999.99";
								else
									num="${line##* }";
								fi
								
							   if [[ $line == *"Paige-Tarjan"* ]]; then
									AVG_PT="$( bc <<<"$AVG_PT + $num" )"
								elif [[ $line == *"Fast"* ]]; then
									AVG_FB="$( bc <<<"$AVG_FB + $num" )"
								fi;
						done < $file;
						ITERATION=$(( $ITERATION + 1 ));
					fi
				done;
				
				echo -n "AVG_PT  " > "$dirNest/results_average.txt";
				echo "$AVG_PT / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
				echo " " >> "$dirNest/results_average.txt";

				echo -n "AVG_FB   " >> "$dirNest/results_average.txt";
				echo "$AVG_FB / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
				echo " " >> "$dirNest/results_average.txt";
			fi;
		done;
	fi;
    echo -e "Done with $(basename $dir)\n"
done
echo -e "The end :)"