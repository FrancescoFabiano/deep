#!/usr/bin/env bash
#Just Run
RESPATH="ICAPS_res/";
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
				AVG_KSTD_V=0;
				AVG_KSTD=0;
				AVG_KOPT_V=0;
				AVG_KOPT=0;
				AVG_POSS_V=0;
				AVG_POSS=0;
				AVG_KOLD=0;

				for file in $(find "$dirNest"/ -type f -name *$DOMAIN_EXT); do
					if [[ "$(basename $file)" == *"exec_"* ]]; then

					#echo $file;
					#mv $OUTPATH_FINDINGPLAN"$(basename $file)" $OUTPATH"$(basename $dir)"/"$(basename $file)"
					while read line; do
					    #if [[$line = "EFP"*]]; then
					       [ -z "$line" ] && continue;
						   if [[ $line == *"TIMED-OUT"* ]]; then
									num="9999.99";
								else
									num="${line##* }";
								fi
						   if [[ $line == *"1.0"* ]]; then
								AVG_KOLD="$( bc <<<"$AVG_KOLD + $num" )"
							else
								if [[ $line == *"VISITED-STATE"* ]]; then
									if [[ $line == *"KRIPKE"* ]]; then
										if [[ $line == *"STANDARD"* ]]; then
											AVG_KSTD_V="$( bc <<<"$AVG_KSTD_V + $num" )"
										else
											if [[ $line == *"OPTIMIZED"* ]]; then
												AVG_KOPT_V="$( bc <<<"$AVG_KOPT_V + $num" )"
											fi
										fi
									else
										if [[ $line == *"POSS"* ]]; then
											if [[ $line == *"STANDARD"* ]]; then
												AVG_POSS_V="$( bc <<<"$AVG_POSS_V + $num" )"
											fi
										fi
									fi
								else
									if [[ $line == *"KRIPKE"* ]]; then
										if [[ $line == *"STANDARD"* ]]; then
											AVG_KSTD="$( bc <<<"$AVG_KSTD + $num" )"
										else
											if [[ $line == *"OPTIMIZED"* ]]; then
												AVG_KOPT="$( bc <<<"$AVG_KOPT + $num" )"
											fi
										fi
									else
										if [[ $line == *"POSS"* ]]; then
											if [[ $line == *"STANDARD"* ]]; then
												AVG_POSS="$( bc <<<"$AVG_POSS + $num" )"
											fi
										fi
									fi
								fi
							fi;
					done < $file;
					ITERATION=$(( $ITERATION + 1 ));
					fi
				done;
				
				echo -n "K-OLD  " > "$dirNest/results_average.txt";
				echo "$AVG_KOLD / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
				echo " " >> "$dirNest/results_average.txt";

				
				echo -n "K-STD_V  " >> "$dirNest/results_average.txt";
				echo "$AVG_KSTD_V / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
								echo -n "K-STD  " >> "$dirNest/results_average.txt";

				echo "$AVG_KSTD / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
				echo " " >> "$dirNest/results_average.txt";
				
				
				echo -n "K-OPT_V  " >> "$dirNest/results_average.txt";
				echo "$AVG_KOPT_V / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
								echo -n "K-OPT  " >> "$dirNest/results_average.txt";

				echo "$AVG_KOPT / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
				echo " " >> "$dirNest/results_average.txt";
				
				
				echo -n "POSS_V   " >> "$dirNest/results_average.txt";
				echo "$AVG_POSS_V / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
				echo -n "POSS   " >> "$dirNest/results_average.txt";
				echo "$AVG_POSS / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
				echo " " >> "$dirNest/results_average.txt";
				# echo -n "$(basename $dir),$INDEX," >> "$RESPATH/graphres.csv";
				# INDEX=$(( $INDEX + 1 ));
				# echo "$AVG_KSTD / $ITERATION" | bc -l >> "$RESPATH/graphres.csv";
				# echo -n "," >> "$RESPATH/graphres.csv";
				# echo "$AVG_KSTD_V / $ITERATION" | bc -l >> "$RESPATH/graphres.csv";
				# echo -n "," >> "$RESPATH/graphres.csv";

				
				# echo "$AVG_KOPT / $ITERATION" | bc -l >> "$RESPATH/graphres.csv";
				# echo -n "," >> "$RESPATH/graphres.csv";
				# echo "$AVG_KOPT_V / $ITERATION" | bc -l >> "$RESPATH/graphres.csv";
				# echo -n "," >> "$RESPATH/graphres.csv";

				# echo "$AVG_POSS / $ITERATION" | bc -l >> "$RESPATH/graphres.csv";
				# echo -n "," >> "$RESPATH/graphres.csv";
				# echo "$AVG_POSS_V / $ITERATION" | bc -l >> "$RESPATH/graphres.csv";
				# echo -n "," >> "$RESPATH/graphres.csv";

				# echo "$AVG_KOLD / $ITERATION" | bc -l >> "$RESPATH/graphres.csv";
			fi;
		done;
	fi;
    echo -e "Done with $(basename $dir)\n"
done
echo -e "The end :)"