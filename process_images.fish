#!/usr/bin/fish
set -x base_folder = "/home/richiewhite/.code/datasets/rsna"

python ./src/preprocessing/preprocess_image_to_png_kaggle.py \
    --phase="train" \
    --base_folder=$base_folder \
    # --num-images=20 \
    # --zoom=2
  
and python ./src/preprocessing/preprocess_image_to_png_kaggle.py \
  --phase="test" \
  --base_folder=$base_folder \
  # --num-images=20 \
  # --zoom=2

and python ./src/preprocessing/preprocess_image_to_png_kaggle.py \
    --phase="train" \
    --base_folder=$base_folder \
    # --num-images=20 \
    # --zoom=3

and python ./src/preprocessing/preprocess_image_to_png_kaggle.py \
    --phase="test" \
    --base_folder=$base_folder \
    # --num-images=20 \
    # --zoom=3