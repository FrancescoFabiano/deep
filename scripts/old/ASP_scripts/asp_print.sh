#!/usr/bin/env bash

DOMAIN="$1"      # Name of the domain
shift            # Shift all arguments to the left (original $1 gets lost)
ARGUMENTS=("$@") # Rebuild the array with rest of arguments
ASP_PATH="../../asp"
OUT_PATH="../../out/asp"
DOMAIN_PATH="../../exp/ICLP20/"

clingo $ASP_PATH/planner_multi.lp $DOMAIN_PATH/$DOMAIN > $OUT_PATH/output.txt;
python3 $ASP_PATH/output_render/out_reader.py $OUT_PATH/output;
dot -Tpdf $OUT_PATH/output.dot > $OUT_PATH/output.pdf;

#rm $OUT_PATH/output.txt;
#rm $OUT_PATH/output.dot;
