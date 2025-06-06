#!/usr/bin/env bash
#Run with the folder (considered already to in "exp/") that contains the domains in .txt as argument: example for ICAPS20 folder run "scripts/folder_asp_encoder.sh ICAPS20/"
INPATH="exp/"$@; #Path of the domain
OUTPATH="out/ASP_encoding/"; #Path of the output
FOLDER_OUT=$@

mkdir -p $OUTPATH$FOLDER_OUT
DOMAIN_EXT=".txt"; # Name of the domain
DOMAIN_ASP_EXT=".lp"; # Name of the domain



echo -e "\nExecution of efp.out -generate_asp in all the folders in" $INPATH "\n"
for dir in $INPATH*; do
  if [ -d "$dir" ]; then

    echo -e "Currently working on the $(basename $dir) directory";

    for file in $(find "$dir"/ -type f -name *$DOMAIN_EXT); do


      echo -e "    Converting the instance $(basename ${file%%.*})";

      bin/efp.out "$file" -generate_asp > /dev/null;

      file_txt="$(basename $file)"
      nested_dir=${file_txt//__?*/}
      file_asp=${file_txt%.*}$DOMAIN_ASP_EXT

      mkdir -p $OUTPATH$FOLDER_OUT"$(basename $dir)"/"$nested_dir"

      mv $OUTPATH"$file_asp" $OUTPATH$FOLDER_OUT"$(basename $dir)"/"$nested_dir"/"$file_asp";
    done
  else
    echo -e "Currently working on the $INPATH directory";

    for file in $(find "$INPATH"/ -type f -name *$DOMAIN_EXT); do


      echo -e "    Converting the instance $(basename ${file%%.*})";

      bin/efp.out "$file" -generate_asp > /dev/null;

      file_txt="$(basename $file)"
      nested_dir=${file_txt//__?*/}
      file_asp=${file_txt%.*}$DOMAIN_ASP_EXT

      mkdir -p $OUTPATH$FOLDER_OUT"$(basename $dir)"/"$nested_dir"

      mv $OUTPATH"$file_asp" $OUTPATH$FOLDER_OUT/"$nested_dir"/"$file_asp";
    done
  fi
  echo -e "Done with $(basename $dir)\n"
done
echo -e "The end :)"
