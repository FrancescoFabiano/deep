#!/usr/bin/env bash
#Just Run
RESPATH1="results/ICLP_res/";
DOMAIN_EXT=".out"; # Name of the domain
configuration=("frumpy" "many");
planner_type=("multi_bis" "multi" "single");
SCALE=3;


#mv $OUTPATH_FINDINGPLAN $OUTPATH$DOMAIN_NAME$DOMAIN_EXT;

echo -e "\nExecution of avg_maker in all the folders in" $RESPATH "\n"
#echo "NAME, INDEX, KRIPKE STANDARD, K-STD V,  KRIPKE OPT, K-OPT V, POSS, POSS V, OLD" > "$RESPATH/graphres.csv";
INDEX=1;

for type in "${planner_type[@]}"; do
	RESPATH="$RESPATH1$type/";
	echo "$RESPATH";
	for dir in $RESPATH*; do
		if [ -d "$dir" ]; then

			echo -e "\tCurrently working on the $(basename $dir) directory";
			#mkdir -p $OUTPATH"$(basename $dir)"

			for dirNest in $dir"/"*; do
				if [ -d "$dirNest" ]; then
					echo -e "\t\tCurrently working on the $(basename $dirNest) directory";
					ITERATION=0;
					AVG_many1=0;
					AVG_many2=0;
					AVG_many3=0;

					AVG_frumpy1=0;
					AVG_frumpy2=0;
					AVG_frumpy3=0;


					for file in $(find "$dirNest"/ -type f -name *$DOMAIN_EXT); do
						if [[ "$(basename $file)" == *"exec_"* ]]; then
							#mv $OUTPATH_FINDINGPLAN"$(basename $file)" $OUTPATH"$(basename $dir)"/"$(basename $file)"
							while read line; do
								#if [[$line = "EFP"*]]; then
								   [ -z "$line" ] && continue;
								   if [[ $line == *"TIMED-OUT"* ]]; then
										num1="9999.99";
										num2="9999.99";
										num3="9999.99";
									else
										echo -e $line | tr '\n' ' ' | sed -e 's/[^0-9|\.]/ /g' -e 's/^ *//g' -e 's/ *$//g' | tr -s ' ' | sed 's/ /\n/g' > num.tmp;
										mapfile -t myArray < num.tmp;
										num1="${myArray[1]}";
										num2="${myArray[2]}";
										num3="${myArray[6]}";
									fi

									if [[ $line == *"many"* ]]; then
										AVG_many1="$( bc <<<"$AVG_many1 + $num1" )";
										AVG_many2="$( bc <<<"$AVG_many2 + $num2" )";
										AVG_many3="$( bc <<<"$AVG_many3 + $num3" )";

									elif [[ $line == *"frumpy"* ]]; then
										AVG_frumpy1="$( bc <<<"$AVG_frumpy1 + $num1" )";
										AVG_frumpy2="$( bc <<<"$AVG_frumpy2 + $num2" )";
										AVG_frumpy3="$( bc <<<"$AVG_frumpy3 + $num3" )";
									fi;

							done < $file;
							ITERATION=$(( $ITERATION + 1 ));
						fi
					done;

					echo "MANY  " > "$dirNest/results_average.txt";
					echo -n "Total:		" >> "$dirNest/results_average.txt";
					echo "scale = $SCALE; $AVG_many1 / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
					echo -n "Solving:	" >> "$dirNest/results_average.txt";
					echo "scale = $SCALE; $AVG_many2 / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
					echo -n "CPU:		" >> "$dirNest/results_average.txt";
					echo "scale = $SCALE; $AVG_many3 / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
					echo " " >> "$dirNest/results_average.txt";

					echo -e "\n\nFRUMPY   " >> "$dirNest/results_average.txt";
					echo -n "Total:		" >> "$dirNest/results_average.txt";
					echo "scale = $SCALE; $AVG_frumpy1 / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
					echo -n "Solving:	" >> "$dirNest/results_average.txt";
					echo "scale = $SCALE; $AVG_frumpy2 / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
					echo -n "CPU:		" >> "$dirNest/results_average.txt";
					echo "scale = $SCALE; $AVG_frumpy3 / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
					echo " " >> "$dirNest/results_average.txt";
				fi;
			done;
		fi;
		echo -e "\tDone with $(basename $dir)\n"
	done
done

rm num.tmp;

echo -e "The end :)"
