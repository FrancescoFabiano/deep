#!/usr/bin/env bash
#Just Run
RESPATH="results/ICLP_res/multi_inc/";
DOMAIN_EXT=".out"; # Name of the domain
configuration=("frumpy" "many");
SCALE=3;

#mv $OUTPATH_FINDINGPLAN $OUTPATH$DOMAIN_NAME$DOMAIN_EXT;

echo -e "\nExecution of avg_maker_splitted in all the folders in" $RESPATH "\n"
#echo "NAME, INDEX, KRIPKE STANDARD, K-STD V,  KRIPKE OPT, K-OPT V, POSS, POSS V, OLD" > "$RESPATH/graphres.csv";
INDEX=1;

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
				AVG_many4=0;
				AVG_many5=0;

				AVG_frumpy1=0;
				AVG_frumpy2=0;
				AVG_frumpy3=0;
				AVG_frumpy4=0;
				AVG_frumpy5=0;


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
								num4="9999.99";
								num5="9999.99";

							else
								echo -e $line | tr '\n' ' ' | sed -e 's/[^0-9|\.]/ /g' -e 's/^ *//g' -e 's/ *$//g' | tr -s ' ' | sed 's/ /\n/g' > num.tmp;
								mapfile -t myArray < num.tmp;
								num1="${myArray[1]}";
								num2="${myArray[2]}";
								num3="${myArray[6]}";
								num4="${myArray[7]}";
								num5="${myArray[8]}";

							fi

							if [[ $line == *"many"* ]]; then
								AVG_many1="$( bc <<<"$AVG_many1 + $num1" )";
								AVG_many2="$( bc <<<"$AVG_many2 + $num2" )";
								AVG_many3="$( bc <<<"$AVG_many3 + $num3" )";
								AVG_many4="$( bc <<<"$AVG_many4 + $num4" )";
								AVG_many5="$( bc <<<"$AVG_many5 + $num5" )";


							elif [[ $line == *"frumpy"* ]]; then
								AVG_frumpy1="$( bc <<<"$AVG_frumpy1 + $num1" )";
								AVG_frumpy2="$( bc <<<"$AVG_frumpy2 + $num2" )";
								AVG_frumpy3="$( bc <<<"$AVG_frumpy3 + $num3" )";
								AVG_frumpy4="$( bc <<<"$AVG_frumpy4 + $num4" )";
								AVG_frumpy5="$( bc <<<"$AVG_frumpy5 + $num5" )";
							fi;

						done < $file;
						ITERATION=$(( $ITERATION + 1 ));
					fi
				done;

				echo "MANY  " > "$dirNest/results_average.txt";
				echo -n -e "Total: \t\t" >> "$dirNest/results_average.txt";
				echo "scale = $SCALE; $AVG_many1 / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
				echo -n -e "Solving: \t" >> "$dirNest/results_average.txt";
				echo "scale = $SCALE; $AVG_many2 / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
				echo -n -e "CPU: \t\t" >> "$dirNest/results_average.txt";
				echo "scale = $SCALE; $AVG_many3 / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
				echo -n -e "\nGrounding: \t" >> "$dirNest/results_average.txt";
				echo "scale = $SCALE; ($AVG_many1 - $AVG_many5) / $ITERATION" | bc -l | tr '\n' ' ' >> "$dirNest/results_average.txt";
				echo -n -e "\t(Ground: " >> "$dirNest/results_average.txt";
				echo "scale = $SCALE; $AVG_many4 / $ITERATION" | bc -l | tr '\n' ' ' >> "$dirNest/results_average.txt";
			  echo -n ", " >> "$dirNest/results_average.txt";
			  echo -n -e "PreProces: " >> "$dirNest/results_average.txt";
			  echo "scale = $SCALE; ($AVG_many1 - ($AVG_many5+$AVG_many4)) / $ITERATION" | bc -l | tr '\n' ' ' >> "$dirNest/results_average.txt";
			  echo ")" >> "$dirNest/results_average.txt";
				echo -n -e "Reasoning: \t" >> "$dirNest/results_average.txt";
				echo "scale = $SCALE; $AVG_many5 / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
				echo " " >> "$dirNest/results_average.txt";

				echo -e "\n\nFRUMPY   " >> "$dirNest/results_average.txt";
				echo -n -e "Total: \t\t" >> "$dirNest/results_average.txt";
				echo "scale = $SCALE; $AVG_frumpy1 / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
				echo -n -e "Solving: \t" >> "$dirNest/results_average.txt";
				echo "scale = $SCALE; $AVG_frumpy2 / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
				echo -n -e "CPU: \t\t" >> "$dirNest/results_average.txt";
				echo "scale = $SCALE; $AVG_frumpy3 / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
				echo -n -e "\nGrounding: \t" >> "$dirNest/results_average.txt";
				echo "scale = $SCALE; ($AVG_frumpy1 - $AVG_frumpy5) / $ITERATION" | bc -l | tr '\n' ' ' >> "$dirNest/results_average.txt";
				echo -n -e "\t(Ground: " >> "$dirNest/results_average.txt";
				echo "scale = $SCALE; $AVG_frumpy4 / $ITERATION" | bc -l | tr '\n' ' ' >> "$dirNest/results_average.txt";
			  echo -n ", " >> "$dirNest/results_average.txt";
			  echo -n -e "PreProces: " >> "$dirNest/results_average.txt";
			  echo "scale = $SCALE; ($AVG_frumpy1 - ($AVG_frumpy5+$AVG_frumpy4)) / $ITERATION" | bc -l | tr '\n' ' ' >> "$dirNest/results_average.txt";
			  echo ")" >> "$dirNest/results_average.txt";
				echo -n -e "Reasoning: \t" >> "$dirNest/results_average.txt";
				echo "scale = $SCALE; $AVG_frumpy5 / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
				echo " " >> "$dirNest/results_average.txt";
			fi;
		done;
	fi;
	echo -e "\tDone with $(basename $dir)\n"
done

rm num.tmp;

echo -e "The end :)"
