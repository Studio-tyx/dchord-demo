#!/bin/bash
CD=/root/autodl-tmp

# test_*.py probes
rm -f $CD/test_partial.py $CD/test_img.py $CD/test_ts_dflash.py $CD/test_ts_schema.py \
      $CD/test_dchord_k3.py $CD/test_dchord_model.py $CD/test_dchord_bm.py \
      $CD/test_keyprefix.py $CD/test_bidir.py

# inspect_*.py probes
rm -f $CD/inspect_dag.py $CD/inspect_dflash.py $CD/inspect_sample.py $CD/inspect_celeba.py \
      $CD/inspect_qwen35.py $CD/inspect_tok.py $CD/inspect_dchord_ckpt.py

# dump / convert / old dchord
rm -f $CD/dump_dflash_src.py $CD/convert_dflash.py $CD/dchord.py $CD/dchord_0ft.py

# all *.b64
rm -f $CD/*.b64

# *.tgz (already extracted)
rm -f $CD/dchord_v22.tgz $CD/scripts.tgz

# inspect_*.out + dchord_*.out + dflash src dumps
rm -f $CD/inspect_*.out $CD/dchord_*.out $CD/dflash_model_src.py $CD/dflash_gen_src.py

# converted temp model + synth parquet
rm -rf $CD/models/Qwen3.5-4B-DFlash-TS
rm -rf $CD/dchord_celeba_synth

echo CLEANUP_DONE
