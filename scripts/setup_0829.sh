#!/bin/bash
set -e
BASE=/root/autodl-tmp
SRC=$BASE/dchord_v22_torch_repro_code_v2
DST=$BASE/0829

mkdir -p $DST/dchord_v22_repro $DST/five_stage $DST/reports $DST/scripts

# --- dchord_v22 repro code ---
cp -r $SRC/patches $DST/dchord_v22_repro/
cp -r $SRC/overlay $DST/dchord_v22_repro/
cp -r $SRC/scripts $DST/dchord_v22_repro/
cp $SRC/README.md $DST/dchord_v22_repro/
cp $SRC/expected_torch_results.json $DST/dchord_v22_repro/
cp $SRC/VERSION $DST/dchord_v22_repro/
cp $SRC/apply.sh $DST/dchord_v22_repro/

# --- acceptance results ---
cp -r $BASE/reports/dchord_repro_test $DST/reports/
cp -r $BASE/reports/dchord_repro_test2 $DST/reports/

echo SETUP_DONE
