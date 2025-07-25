#!/bin/bash -l

#SBATCH --job-name graphmetrics      # Job name
#SBATCH --account rad
#SBATCH --partition rad
#SBATCH --array=0      # Replace with the number of lines in your parameter file - 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50G
##SBATCH --gres=gpu:rtx3090:1  #(The name of the GPU)
#SBATCH --nodelist=gpunode02
#SBATCH --time=1-23:59:00       #(MAXIMAL TIME OF EXECUTION) hrs:min:sec
#SBATCH --chdir=/data/hagmann_group/harmonization/graph_harmonization_final/batch_scripts/log_files/GRAPHMETRICS/ #(CHANGE DIRECTORY TO THIS ONE)
#SBATCH -o %N.%j.%a.out #(AUTOMATICALLY CREATED FOLDER WHERE THE OUTPUTS OF YOUR SCRIPT WILL BE STORED)
#SBATCH -e %N.%j.%a.err  #(SAME BUT FOR ERRORS)

## This is the code that the selected node will run
echo "Job executed on " $(hostname)

MAINDIR="GRAPHMETRICS"

# Set the path to the parameter combinations file
PARAM_FILE="/data/hagmann_group/harmonization/graph_harmonization_final/batch_scripts/parameters/$MAINDIR/parameters_combinations_1.txt"

echo "$PARAM_FILE"

echo "$SLURM_ARRAY_TASK_ID"

# Read the specific line from the parameter file based on SLURM_ARRAY_TASK_ID
PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID+4))p" $PARAM_FILE)

#echo "$PARAMS"

# Add extra parameters directly in the SLURM script
EXTRA_PARAMS="--maindir $MAINDIR"

# Combine PARAMS from the file and EXTRA_PARAMS
ALL_PARAMS="$PARAMS $EXTRA_PARAMS"

echo "$ALL_PARAMS"

source activate /cluster/home/ja3098/.conda/envs/harmonization_final #(ACTIVATE THE ENVIRONMENT YOU WOULD HAVE CREATED BEFORE)

echo "Environment activated"

# Run the Python script with the combined parameter options
python /data/hagmann_group/harmonization/graph_harmonization_final/src/calculate_graph_metrics_actual_matrices.py $ALL_PARAMS # (PATH to the python file you want to execute)
