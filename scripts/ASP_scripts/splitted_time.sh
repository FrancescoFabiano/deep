#!/usr/bin/env bash
#This script prints out the time for executing a plan on the same domain with the two version of EFP
#USAGE EXAMPLE (from the project root folder): ". scripts/comparison.sh exp/paper/CC/CC_2_2_3.txt"
#	where "exp/paper/CC/CC_2_2_3.txt" is the path to the domain name


TIMEOUT="30m"
OUT_EXT=".out";

THREADS=4;


#configuration=("auto" "frumpy" "jumpy" "tweety" "handy" "crafty" "trendy" "many");
configuration=("frumpy" "many");
mkdir -p out/ASP_comparison/findingplan/multi_inc/

#REMOVE OLD COPIES
if [ -f out/ASP_comparison/findingplan/multi_inc/"$(basename ${@%%.*})$OUT_EXT" ] ; then
	rm out/ASP_comparison/findingplan/multi_inc/"$(basename ${@%%.*})$OUT_EXT"
fi
echo -e "    	Using planner_multi_inc"

for config in "${configuration[@]}"; do

	echo -e "    		Solving with configuration $config";
	timeout $TIMEOUT clingo asp/testing/inc.lp asp/testing/planner_multi_inc.lp $@ -t $THREADS --configuration=$config > findingplan_comparison.tmp 2> err_file.tmp
	exit_status=$?;
	TOPRINT="Configuration $config ";
	TOPRINT+="with $THREADS Threads ";
	#TOPRINT+="took "


	if [[ $exit_status -eq 124 ]]; then
		TOPRINT+="TIMED-OUT."
		echo -e "$TOPRINT" >> out/ASP_comparison/findingplan/multi_inc/"$(basename ${@%%.*})$OUT_EXT";
	else
		echo -e "$TOPRINT" >> out/ASP_comparison/findingplan/multi_inc/"$(basename ${@%%.*})$OUT_EXT";
		grep -w "Time" findingplan_comparison.tmp >> out/ASP_comparison/findingplan/multi_inc/"$(basename ${@%%.*})$OUT_EXT";
		grep -w "Grounding" findingplan_comparison.tmp >> out/ASP_comparison/findingplan/multi_inc/"$(basename ${@%%.*})$OUT_EXT";
		grep -w "Reasoning:" findingplan_comparison.tmp >> out/ASP_comparison/findingplan/multi_inc/"$(basename ${@%%.*})$OUT_EXT";

	fi;

	echo $'\n' >> out/ASP_comparison/findingplan/multi_inc/"$(basename ${@%%.*})$OUT_EXT";
	awk -v RS="\0" -v ORS="" '{gsub(/\s+Time\s+/,"")}7' out/ASP_comparison/findingplan/multi_inc/"$(basename ${@%%.*})$OUT_EXT" > out/ASP_comparison/findingplan/multi_inc/"$(basename ${@%%.*})-1.tmp";
	awk -v RS="\0" -v ORS="" '{gsub(/\s*CPU/,", cpu")}7' out/ASP_comparison/findingplan/multi_inc/"$(basename ${@%%.*})-1.tmp" > out/ASP_comparison/findingplan/multi_inc/"$(basename ${@%%.*})-2.tmp";
	awk -v RS="\0" -v ORS="" '{gsub(/\s*Grounding/,", grounding")}7' out/ASP_comparison/findingplan/multi_inc/"$(basename ${@%%.*})-2.tmp" > out/ASP_comparison/findingplan/multi_inc/"$(basename ${@%%.*})-3.tmp";
	awk -v RS="\0" -v ORS="" '{gsub(/\s*Reasoning/,", reasoning")}7' out/ASP_comparison/findingplan/multi_inc/"$(basename ${@%%.*})-3.tmp" > out/ASP_comparison/findingplan/multi_inc/"$(basename ${@%%.*}).tmp";

	rm out/ASP_comparison/findingplan/multi_inc/"$(basename ${@%%.*})$OUT_EXT";

	rm out/ASP_comparison/findingplan/multi_inc/"$(basename ${@%%.*})-1.tmp";
	rm out/ASP_comparison/findingplan/multi_inc/"$(basename ${@%%.*})-2.tmp";
	rm out/ASP_comparison/findingplan/multi_inc/"$(basename ${@%%.*})-3.tmp";

	mv out/ASP_comparison/findingplan/multi_inc/"$(basename ${@%%.*}).tmp" out/ASP_comparison/findingplan/multi_inc/"$(basename ${@%%.*})$OUT_EXT"

	mv findingplan_comparison.tmp out/ASP_comparison/findingplan/multi_inc/"plans$OUT_EXT"

	sleep 10s;

done;

rm err_file.tmp;
