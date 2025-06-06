#!/usr/bin/env bash
#Just Run
INPATH="exp/ICLP20/Bisimulation/"; #Path of the domain
OUTPATH="out/ICLP20/configs/"; #Path of the output

OUTPATH_FINDINGPLAN='out/ASP_comparison/findingplan/'

mkdir -p $OUTPATH
DOMAIN_EXT=".lp"; # Name of the domain
OUT_EXT=".out";

ITERATIONS=3;
MAX_TIME_PER_ITERATION=3600*1;


#mv $OUTPATH_FINDINGPLAN $OUTPATH$DOMAIN_NAME$DOMAIN_EXT;

echo -e "\nExecution of planner_asp in all the folders in" $INPATH "\n"
for dir in $INPATH*; do
  if [ -d "$dir" ]; then

    echo -e "Currently working on the $(basename $dir) directory";
    #mkdir -p $OUTPATH"$(basename $dir)"

    for file in $(find "$dir"/ -type f -name *$DOMAIN_EXT); do
      mkdir -p $OUTPATH"multi_bis"/"$(basename $dir)"/"$(basename ${file%%.*})"/"for_average"


      echo -e "    Solving the instance $(basename ${file%%.*})";
      i="1";
      while [[ $i -le $ITERATIONS ]]; do
        start=$SECONDS;
        . scripts/ASP_scripts/bis_execution.sh "$file";
        duration=$(( SECONDS - start ));

        grep -w "Configuration" $OUTPATH_FINDINGPLAN"multi_bis"/"$(basename ${file%%.*})$OUT_EXT" > $OUTPATH"multi_bis"/"$(basename $dir)"/"$(basename ${file%%.*})"/"for_average"/"exec_$i$OUT_EXT";

        if [[ $duration -gt $MAX_TIME_PER_ITERATION ]]; then
          i=$(( $ITERATIONS + 1 ));
          #echo $i;
        else
          i=$(( $i + 1 ));
          #echo $i;
        fi
      done
      mv $OUTPATH_FINDINGPLAN"multi_bis"/"$(basename ${file%%.*})$OUT_EXT" $OUTPATH"multi_bis"/"$(basename $dir)"/"$(basename ${file%%.*})"/"results$OUT_EXT";

      mv $OUTPATH_FINDINGPLAN"multi_bis"/"plans$OUT_EXT" $OUTPATH"multi_bis"/"$(basename $dir)"/"$(basename ${file%%.*})"/"plans$OUT_EXT";

    done
  fi
  echo -e "Done with $(basename $dir)\n"
done
echo -e "The end :)"
