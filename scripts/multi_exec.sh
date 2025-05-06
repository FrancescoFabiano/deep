#!/usr/bin/env bash
#Just Run
INPATH="exp/Doxastic/"; #Path of the domain
OUTPATH="out/Doxastic/"; #Path of the output

OUTPATH_FINDINGPLAN='out/EFP_comparison/findingplan/'

mkdir -p $OUTPATH
DOMAIN_EXT=".txt"; # Name of the domain

ITERATIONS=3;
MAX_TIME_PER_ITERATION=3600;



#mv $OUTPATH_FINDINGPLAN $OUTPATH$DOMAIN_NAME$DOMAIN_EXT;

echo -e "\nExecution of efp.out in all the folders in" $INPATH "\n"
for dir in $INPATH*; do
    if [ -d "$dir" ]; then

        echo -e "Currently working on the $(basename $dir) directory";
        #mkdir -p $OUTPATH"$(basename $dir)"

        for file in $(find "$dir"/ -type f); do
            mkdir -p $OUTPATH"$(basename $dir)"/"$(basename ${file%%.*})"/"for_average"

            echo -e "    Solving the instance $(basename ${file%%.*})";
            i="1";
            while [[ $i -le $ITERATIONS ]]; do
                start=$SECONDS;
					. scripts/single_exec.sh "$file";

				duration=$(( SECONDS - start ));
                #mv $OUTPATH_FINDINGPLAN"$(basename $file)" $OUTPATH"$(basename $dir)"/"$(basename $file)"
                grep -w "EFP" $OUTPATH_FINDINGPLAN"$(basename $file)" >> $OUTPATH"$(basename $dir)"/"$(basename ${file%%.*})"/"for_average"/"exec_$i.txt";
               # while read line; do
               #    if ["$line" = "EFP"*]; then
               #       [ -z "$line" ] && continue;
               #      echo ${line##* };
               # fi;
               #done < $OUTPATH_FINDINGPLAN"$(basename $file)";
				if [[ $duration -gt $MAX_TIME_PER_ITERATION ]]; then
					i=$(( $ITERATIONS + 1 ));
					#echo $i;
				else
					i=$(( $i + 1 ));
					#echo $i;
				fi
            done
            mv $OUTPATH_FINDINGPLAN"$(basename $file)" $OUTPATH"$(basename $dir)"/"$(basename ${file%%.*})"/"results.txt";
        done
    fi
    echo -e "Done with $(basename $dir)\n"
done
echo -e "The end :)"
