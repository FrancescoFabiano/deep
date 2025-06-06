#!/usr/bin/env bash
#Just Run
RESPATH="/mnt/c/Users/Francesco/Desktop/ICLP_res/";
DOMAIN_EXT=".out"; # Name of the domain

#mv $OUTPATH_FINDINGPLAN $OUTPATH$DOMAIN_NAME$DOMAIN_EXT;

echo -e "\nExecution of efp.out in all the folders in" $RESPATH "\n"

COnfig_RESPATH=$RESPATH"ICLP20/configs/";
planner_type=("multi" "single");

echo -n "Domain ,auto, frumpy, jumpy, tweety, handy, crafty, trendy, many" > "$RESPATH/graphres.csv";
INDEX=1;
for type in "${planner_type[@]}"; do
	for dir in $RESPATH"ICLP20/configs/$type/"*; do
		if [ -d "$dir" ]; then

			echo -e "Currently working on the $(basename $dir) directory";
			#mkdir -p $OUTPATH"$(basename $dir)"

			for dirNest in $dir"/"*; do
				if [ -d "$dirNest" ]; then
					echo -e "Currently working on the $(basename $dirNest) directory";

					for file in $(find "$dirNest"/ -type f -name results$DOMAIN_EXT); do
						echo -n -e '\n'$(basename $dirNest)"-$type" >> "$RESPATH/graphres.csv";
						#mv $OUTPATH_FINDINGPLAN"$(basename $file)" $OUTPATH"$(basename $dir)"/"$(basename $file)"
						while read line; do
							#if [[$line = "EFP"*]]; then
							[ -z "$line" ] && continue;
							if [[ $line == *"TIMED-OUT"* ]]; then
								num=",9999.99";
							else
								num=",${line##* }";
								num=${num%?};
							fi
							echo -n $num >> "$RESPATH/graphres.csv";
						done < $file;
					done;
				fi;
			done;
		fi;
		echo -e "Done with $(basename $dir)\n"
	done
done
echo -e "The end :)"
