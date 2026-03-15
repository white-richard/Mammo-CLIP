#!/usr/bin/fish
set -x base_folder = "/home/richiewhite/.code/datasets/rsna"

python ./src/preprocessing/preprocess_image_to_png_kaggle.py \
    --phase="train" \
    --base_folder=$base_folder \
    --zoom=2
    # --num-images=20
  
and python ./src/preprocessing/preprocess_image_to_png_kaggle.py \
  --phase="test" \
  --base_folder=$base_folder \
  --zoom=2
  # --num-images=20

and python ./src/preprocessing/preprocess_image_to_png_kaggle.py \
    --phase="train" \
    --base_folder=$base_folder \
    --zoom=3
    # --num-images=20

and python ./src/preprocessing/preprocess_image_to_png_kaggle.py \
    --phase="test" \
    --base_folder=$base_folder \
    --zoom=3
    # --num-images=20