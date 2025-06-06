#!/usr/bin/env bash
#####Used to transform each .dot file inside a folder (argument) in a .pdf
#This script is used inside the code
for file in "$@"/*;
do
   if [ "${file##*.}" = "dot" ]; then
	  dot -Tpdf "$file" > "${file%.*}".pdf
	  dot -Tpng "$file" > "${file%.*}".png
	  rm "${file%.*}".dot
   fi
done
