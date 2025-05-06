#!/usr/bin/env bash
#Just Run
RESPATH="/mnt/c/Users/Francesco/Desktop/out/Doxastic/";
DOMAIN_EXT=".txt"; # Name of the domain
SCALE=2;


#mv $OUTPATH_FINDINGPLAN $OUTPATH$DOMAIN_NAME$DOMAIN_EXT;

echo -e "\nExecution of avg_maker in all the folders in" $RESPATH "\n"
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
					AVG_1=0;
					for file in $(find "$dirNest"/ -type f); do
						if [[ "$(basename $file)" == *"exec_"* ]]; then
							#mv $OUTPATH_FINDINGPLAN"$(basename $file)" $OUTPATH"$(basename $dir)"/"$(basename $file)"
							while read line; do

								#if [[$line = "EFP"*]]; then
								   [ -z "$line" ] && continue;
								   if [[ $line == *"TIMED-OUT"* ]]; then
										num1="9999.99";
									else
										echo -e $line | tr '\n' ' ' | sed -e 's/[^0-9|\.]/ /g' -e 's/^ *//g' -e 's/ *$//g' | tr -s ' ' | sed 's/ /\n/g' > num.tmp;
										mapfile -t myArray < num.tmp;
										num1="${myArray[1]}";
									fi
									AVG_1="$( bc <<<"$AVG_1 + $num1" )";


							done < $file;
							ITERATION=$(( $ITERATION + 1 ));
						fi
					done;

					echo -n "Average Time:  " > "$dirNest/results_average.txt";
					echo "scale = $SCALE; $AVG_1 / $ITERATION" | bc -l >> "$dirNest/results_average.txt";
				fi;
			done;
		fi;
		echo -e "\tDone with $(basename $dir)\n"
	done


rm num.tmp;

echo -e "The end :)"
