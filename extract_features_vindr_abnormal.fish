#!/usr/bin/env fish

python ./src/codebase/extract_features.py \
  --data-dir "$HOME/.code/datasets/vindr-mammo" \
  --img-dir "images_png" \
  --csv-file "vindr_detection_v1_folds_abnormal.csv" \
  --clip_chk_pt_path "$HOME/.code/model_weights/mammoclip/mammoclip-b5-model-best-epoch-7.tar" \
  --dataset "ViNDr" \
  --split "all" \
  --label "abnormal" \
  --arch "upmc_breast_clip_det_b5_period_n_lp" \
  --output-file "out/features/vindr_abnormal_all_features.pt" \
  --batch-size 16 \
  --num-workers 4 \
  --mean 0.3089279 \
  --std 0.25053555408335154
