#!/usr/bin/env bash
#Just Run
INPATH='exp/ICAPS20/correctness/newma_example/'; #Path of the domain
OUTPATH='out/ICAPS20/correctness/newma_example/'; #Path of the output
DOMAIN_NAME='example10';
DOMAIN_EXT='.txt'; # Name of the domain


OUTPATH_FINDINGPLAN='out/EFP_comparison/findingplan/'$DOMAIN_NAME$DOMAIN_EXT
OUTPATH_GIVENPLAN='out/EFP_comparison/givenplan/'$DOMAIN_NAME$DOMAIN_EXT
OUTPATH_GRAPHVIZ='out/state/'$DOMAIN_NAME'_'

DOMAIN=$INPATH$DOMAIN_NAME$DOMAIN_EXT;
ARGUMENTS='distract_a_c open_a peek_a';

scripts/comparison.sh $DOMAIN;
scripts/givenplan_comparison.sh $DOMAIN $ARGUMENTS > temp.tmp;
scripts/givenplan_comparison_graphviz.sh $DOMAIN $ARGUMENTS> temp.tmp;

mkdir -p $OUTPATH'/'

mv $OUTPATH_FINDINGPLAN $OUTPATH'/'$DOMAIN_NAME'_finding.txt';
mv $OUTPATH_GIVENPLAN $OUTPATH'/'$DOMAIN_NAME'_givenplan.txt';

mkdir -p $OUTPATH'/'$DOMAIN_NAME'_state/kripke';
mkdir -p $OUTPATH'/'$DOMAIN_NAME'_state/kripke_opt';
mkdir -p $OUTPATH'/'$DOMAIN_NAME'_state/poss';

mv $OUTPATH_GRAPHVIZ'kripke'/* $OUTPATH'/'$DOMAIN_NAME'_state/kripke/'
mv $OUTPATH_GRAPHVIZ'kripke_opt'/* $OUTPATH'/'$DOMAIN_NAME'_state/kripke_opt/'
mv $OUTPATH_GRAPHVIZ'poss'/* $OUTPATH'/'$DOMAIN_NAME'_state/poss/'

rm -r $OUTPATH_GRAPHVIZ*

############################The second one
DOMAIN_NAME='example7';

OUTPATH_FINDINGPLAN='out/EFP_comparison/findingplan/'$DOMAIN_NAME$DOMAIN_EXT
OUTPATH_GIVENPLAN='out/EFP_comparison/givenplan/'$DOMAIN_NAME$DOMAIN_EXT
OUTPATH_GRAPHVIZ='out/state/'$DOMAIN_NAME'_'

DOMAIN=$INPATH$DOMAIN_NAME$DOMAIN_EXT;
ARGUMENTS='open_a';

mkdir -p $OUTPATH'/'


scripts/comparison.sh $DOMAIN;
scripts/givenplan_comparison.sh $DOMAIN $ARGUMENTS > temp.tmp;
scripts/givenplan_comparison_graphviz.sh $DOMAIN $ARGUMENTS> temp.tmp;


mv $OUTPATH_FINDINGPLAN $OUTPATH'/'$DOMAIN_NAME'_finding.txt';
mv $OUTPATH_GIVENPLAN $OUTPATH'/'$DOMAIN_NAME'_givenplan.txt';

mkdir -p $OUTPATH'/'$DOMAIN_NAME'_state/kripke';
mkdir -p $OUTPATH'/'$DOMAIN_NAME'_state/kripke_opt';
mkdir -p $OUTPATH'/'$DOMAIN_NAME'_state/poss';

mv $OUTPATH_GRAPHVIZ'kripke'/* $OUTPATH'/'$DOMAIN_NAME'_state/kripke/'
mv $OUTPATH_GRAPHVIZ'kripke_opt'/* $OUTPATH'/'$DOMAIN_NAME'_state/kripke_opt/'
mv $OUTPATH_GRAPHVIZ'poss'/* $OUTPATH'/'$DOMAIN_NAME'_state/poss/'

rm -r $OUTPATH_GRAPHVIZ*

############################The third one
DOMAIN_NAME='example8';

OUTPATH_FINDINGPLAN='out/EFP_comparison/findingplan/'$DOMAIN_NAME$DOMAIN_EXT
OUTPATH_GIVENPLAN='out/EFP_comparison/givenplan/'$DOMAIN_NAME$DOMAIN_EXT
OUTPATH_GRAPHVIZ='out/state/'$DOMAIN_NAME'_'

DOMAIN=$INPATH$DOMAIN_NAME$DOMAIN_EXT;
ARGUMENTS='peek_a';

mkdir -p $OUTPATH'/'


scripts/comparison.sh $DOMAIN;
scripts/givenplan_comparison.sh $DOMAIN $ARGUMENTS > temp.tmp;
scripts/givenplan_comparison_graphviz.sh $DOMAIN $ARGUMENTS> temp.tmp;

rm temp.tmp;

mv $OUTPATH_FINDINGPLAN $OUTPATH'/'$DOMAIN_NAME'_finding.txt';
mv $OUTPATH_GIVENPLAN $OUTPATH'/'$DOMAIN_NAME'_givenplan.txt';

mkdir -p $OUTPATH'/'$DOMAIN_NAME'_state/kripke';
mkdir -p $OUTPATH'/'$DOMAIN_NAME'_state/kripke_opt';
mkdir -p $OUTPATH'/'$DOMAIN_NAME'_state/poss';

mv $OUTPATH_GRAPHVIZ'kripke'/* $OUTPATH'/'$DOMAIN_NAME'_state/kripke/'
mv $OUTPATH_GRAPHVIZ'kripke_opt'/* $OUTPATH'/'$DOMAIN_NAME'_state/kripke_opt/'
mv $OUTPATH_GRAPHVIZ'poss'/* $OUTPATH'/'$DOMAIN_NAME'_state/poss/'

rm -r $OUTPATH_GRAPHVIZ*


############################WRONG one
INPATH='exp/paper/Coin_In_The_Box/'; #Path of the domain
OUTPATH='out/ICAPS20/correctness/wrong/'; #Path of the output
DOMAIN_NAME='Coin_in_the_Box';

OUTPATH_FINDINGPLAN='out/EFP_comparison/findingplan/'$DOMAIN_NAME$DOMAIN_EXT
OUTPATH_GIVENPLAN='out/EFP_comparison/givenplan/'$DOMAIN_NAME$DOMAIN_EXT
OUTPATH_GRAPHVIZ='out/state/'$DOMAIN_NAME'_'

DOMAIN=$INPATH$DOMAIN_NAME$DOMAIN_EXT;

mkdir -p $OUTPATH'/'

scripts/comparison.sh $DOMAIN;

mv $OUTPATH_FINDINGPLAN $OUTPATH'/'$DOMAIN_NAME'_finding.txt';