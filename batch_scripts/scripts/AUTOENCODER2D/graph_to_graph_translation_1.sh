#!/bin/bash -l

#SBATCH --job-name autoencoder2D      # Job name
#SBATCH --account rad
#SBATCH --partition rad
#SBATCH --array=0      # Replace with the number of lines in your parameter file - 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:rtx6000:1  #(The name of the GPU)
##SBATCH --nodelist=gpunode02
#SBATCH --time=1-23:59:00       #(MAXIMAL TIME OF EXECUTION) hrs:min:sec
#SBATCH --chdir=/data/hagmann_group/harmonization/graph_harmonization_final/batch_scripts/log_files/AUTOENCODER2D/training/ #(CHANGE DIRECTORY TO THIS ONE)
#SBATCH -o %N.%j.%a.out #(AUTOMATICALLY CREATED FOLDER WHERE THE OUTPUTS OF YOUR SCRIPT WILL BE STORED)
#SBATCH -e %N.%j.%a.err  #(SAME BUT FOR ERRORS)

## This is the code that the selected node will run
echo "Job executed on " $(hostname)

MAINDIR="AUTOENCODER2D"

# Set the path to the parameter combinations file
PARAM_FILE="/data/hagmann_group/harmonization/graph_harmonization_final/batch_scripts/parameters/$MAINDIR/parameters_combinations_1.txt"

echo "$PARAM_FILE"

echo "$SLURM_ARRAY_TASK_ID"

# Read the specific line from the parameter file based on SLURM_ARRAY_TASK_ID
PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $PARAM_FILE)

#echo "$PARAMS"

trainSet="train"
valSet="val"

# Add extra parameters directly in the SLURM script
EXTRA_PARAMS="--epochs 1000 --maindir $MAINDIR --train_set $trainSet --val_set $valSet --best latest"

# Combine PARAMS from the file and EXTRA_PARAMS
ALL_PARAMS="$PARAMS $EXTRA_PARAMS"

echo "$ALL_PARAMS"

source activate /cluster/home/ja3098/.conda/envs/harmonization_final #(ACTIVATE THE ENVIRONMENT YOU WOULD HAVE CREATED BEFORE)

echo "Environment activated"

# Run the Python script with the combined parameter options
python /data/hagmann_group/harmonization/graph_harmonization_final/src/image_image_translation_data_augmentation_autoencoder_2d.py $ALL_PARAMS # (PATH to the python file you want to execute)
