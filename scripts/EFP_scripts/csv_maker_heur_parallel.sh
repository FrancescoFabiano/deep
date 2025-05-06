#!/usr/bin/env bash
#Just Run
RESPATH="out/Heur-tests_parallel/";
DOMAIN_EXT=".txt"; # Name of the domain

#mv $OUTPATH_FINDINGPLAN $OUTPATH$DOMAIN_NAME$DOMAIN_EXT;

heuristics=('BFS' 'SUM_PG' 'LENGTH_PG' 'CLASSI_PG' 'SUBGOALS')
heuristics_TO=('NONE' 'S_PG' 'L_PG' 'C_PG' 'SUBGOALS')
posHeur=$(( ${#heuristics[*]} - 1 ))
lastHeur="${heuristics[$posHeur]}"


fields=('Time' 'Nodes' 'Length')
posFields=$(( ${#fields[*]} - 1 ))
lastFields="${fields[$posFields]}"

echo -e "\nExecution of csv_maker in all the folders in" $RESPATH "\n"

echo -n "Domain,Instance," > "$RESPATH/graphres.csv";
for heur in "${heuristics[@]}"; do
	for field in "${fields[@]}"; do
		if [[ $field == $lastFields && $heur == $lastHeur ]]; then
     		echo "$field-$heur" >> "$RESPATH/graphres.csv";
		else 
     		echo -n "$field-$heur," >> "$RESPATH/graphres.csv";
  		fi 
	done
done

for dir in $RESPATH*; do
	if [ -d "$dir" ]; then

       	echo -e "Currently working on the $(basename $dir) directory";
       	#mkdir -p $OUTPATH"$(basename $dir)"

		#Replacing faulty files

		for dirNest in $dir"/"*; do
			if [ -d "$dirNest" ]; then
				for file in $(find "$dirNest"/ -type f -name results.txt); do

					counter=0
					for heur in "${heuristics[@]}"; do
						sed -i "s/${heuristics_TO[$counter]}/$heur/g" "$file"
						counter=$((counter+1))
					done
				
					domain=$(basename $dirNest)
					instance_num="${domain#*_}";
					echo -e "Currently working on the $(basename $dirNest) directory";
					echo -n "${domain%"_$instance_num"},$domain," >> "$RESPATH/graphres.csv";

					
					for heur in "${heuristics[@]}"; do
					
						while read line; do					
							if [[ $line == *"EFP Version 2.0"*"$heur"* ]]; then

								if [[ $line == *"TIMED-OUT"* ]]; then
									num="TO";
								else
									num="${line##* }";
								fi
								

								if  [[ $num == "TO"* ]]; then
									
									if [[ $heur == $lastHeur ]]; then
										echo "TO,-,-" >> "$RESPATH/graphres.csv";
									else 
										echo -n "TO,-,-," >> "$RESPATH/graphres.csv";
									fi 

								else
									length="-1"
									nodes="-1"
									while read line_nested; do
										if [[ $line_nested == *"[$heur]Plan Length"* ]]; then
											length=$( echo $line_nested | tr -d -c 0-9 )
										fi

										if [[ $line_nested == *"[$heur]Expanded Nodes"* ]]; then
											nodes=$( echo $line_nested | tr -d -c 0-9 )
										fi
									done < $file;        							

									if [[ $heur == $lastHeur ]]; then
										echo "$num,$nodes,$length" >> "$RESPATH/graphres.csv";
									else 
										echo -n "$num,$nodes,$length," >> "$RESPATH/graphres.csv";
									fi
								fi
							fi
						done < $file;
					done
				done
			fi
		done
	fi
done

python3 scripts/EFP_scripts/gen_table.py

echo -e "The end :)"