#!/bin/bash -l

#SBATCH --job-name evalLinear      # Job name
#SBATCH --account rad
#SBATCH --partition rad
#SBATCH --array=0     # Replace with the number of lines in your parameter file - 1
#SBATCH --mem-per-cpu=60G
#SBATCH --time=1-23:59:00       #(MAXIMAL TIME OF EXECUTION) hrs:min:sec
#SBATCH --nodelist=gpunode03
#SBATCH --chdir=/data/hagmann_group/harmonization/graph_harmonization_final/batch_scripts/log_files/EVALUATION/training/ #(CHANGE DIRECTORY TO THIS ONE)
#SBATCH -o %N.%j.%a.out #(AUTOMATICALLY CREATED FOLDER WHERE THE OUTPUTS OF YOUR SCRIPT WILL BE STORED)
#SBATCH -e %N.%j.%a.err  #(SAME BUT FOR ERRORS)

## This is the code that the selected node will run
echo "Job executed on " $(hostname)

MAINDIR="EVALUATION"
CKDIR="checkpoints1"
SEED="42"

echo "$SLURM_ARRAY_TASK_ID"

# Add extra parameters directly in the SLURM script
EXTRA_PARAMS="--maindir $MAINDIR --ckdir $CKDIR --seed $SEED"

# Combine PARAMS from the file and EXTRA_PARAMS
ALL_PARAMS="$EXTRA_PARAMS"

echo "$ALL_PARAMS"

source activate /cluster/home/ja3098/.conda/envs/harmonization_final #(ACTIVATE THE ENVIRONMENT YOU WOULD HAVE CREATED BEFORE)

echo "Environment activated"

# Run the Python script with the combined parameter options
python /data/hagmann_group/harmonization/graph_harmonization_final/src/evaluation_training_linear.py $ALL_PARAMS # (PATH to the python file you want to execute)
