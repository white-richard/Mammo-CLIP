#!/bin/sh
#SBATCH --output=src/psc_logsmisc/back_translation_wo_period_%j.out

pwd
hostname
date

CURRENT=$(date +"%Y-%m-%d_%T")
echo $CURRENT

slurm_output_train1=src/psc_logsmisc/back_translation_wo_period_$CURRENT.out

source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh

conda activate breast_clip_rtx_6000


python src/codebase/augment_text.py \
  --dataset-path="src/codebase/data_csv" \
  --csv-path="upmc_dicom_consolidated_final_folds_BIRADS_num_1_report.csv" \
  --dataset="upmc"