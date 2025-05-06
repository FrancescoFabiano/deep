#!/usr/bin/env bash
#This script prints out the time for executing a plan on the same domain with all the heuristics of EFP
#USAGE EXAMPLE (from the project root folder): ". scripts/comparison_heuristics.sh"

#REMOVE OLD COPIES

file=$1

if [[ -f out/EFP_comparison/findingplan/"${file##*/}" ]] ; then
    rm out/EFP_comparison/findingplan/"${file##*/}"
fi

TIMEOUT="120";

heuristics=("NONE" "S_PG" "L_PG" "C_PG" "SUBGOALS"); 

cleanup(){
  myPID=$BASHPID
  # use mkdir as a mutex between the child processes
  if mkdir mutex_lock_directory
  then
    # when the mutex is aquired (this process finished first) kill all other processes doing heuristics
    echo "$myPID has aquired the lock"
    while IFS= read -r line; do
      if [ $myPID -ne $line ]
        then
          echo "[$myPID] killing $line"
          kill "$line" 
      fi
    done < pids.tmp
    # unlock the mutex by destroying lock.tmp
    rm -r mutex_lock_directory
  # process finished, but not first. wait to be killed. (wait instead of self terminating so that the pid remains taken and another process is not assigned it only to be killed by the first heuristic to complete.)
  else  
    echo "$BASHPID cannot aquire the lock."
    sleep 5
    echo "Process took to long to be killed, self terminating."
    exit 0
  fi
  echo "$myPID has finished cleanup."
}

func(){
  # echo "1 is ${1}"
  # echo "1##*/ is ${1##*/}"
  # echo "-------------------"

  heur=$2
  cleaner=$3
  if [ $cleaner = "f" ]; then
    tmp_output="findingplan_comparison_${heur}.tmp"
  else
    tmp_output="findingplan_comparison.tmp"
  fi
  header=""
  body=""
  # echo "Working with file: $tmp_output"
  

  timeout $TIMEOUT bin/efp.out $file -results_file -check_visited -st POSS -h $heur > "$tmp_output"
  exit_status=$?;
  if [ $cleaner = "t" ];then
    cleanup &>/dev/null #silenced output
  fi
  

	if [[ $exit_status -eq 124 ]]; then
        header="EFP Version 2.0 (on POSS and fast Bisimulation) with $heur and VISTED-STATE TIMED-OUT \n"
        echo -e "$header" >> out/EFP_comparison/findingplan/"${file##*/}";
	else
		body=" [$heur]$(grep -w "Executed actions:" "$tmp_output")"
    body="$body \n [$heur]$(grep -w "Plan Length:" "$tmp_output")"
    body="$body \n [$heur]$(grep -w "Expanded Nodes:" "$tmp_output") \n"
    echo -e "$body" >> out/EFP_comparison/findingplan/"${file##*/}";
	fi;
  # sleep 2s;
  # rm "$tmp_output"
}


declare -a pids_array=()
rm -r mutex_lock_directory &>/dev/null #silenced output
rm -rf pids.tmp &>/dev/null #silenced output
for heur in "${heuristics[@]}"; do
  func $1 $heur $2 &
  echo "Spawned thread [$!]: $heur"
  echo "$!" >> pids.tmp
  # sleep 5
done;
wait

rm findingplan_comparison_*.tmp &>/dev/null #silenced output
rm pids.tmp &>/dev/null #silenced output
rm -r mutex_lock_directory &>/dev/null #silenced output


# cd /mnt/d/Documents/NMSU/Research/Parallel\ Programming/Codebase/EFP/