# import the packages

print("graph_evaluation_all_metrics_main.py starting")

import os
import torch

from utils.config_evaluation import read_configuration_param
from utils.functions_basic import set_seed
from graph_evaluation_all_metrics import graph_evaluation_function

print("All packages imported")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print("PATH:", os.environ.get('PATH'))
print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH'))
if torch.cuda.is_available():
    print("Available GPUs:", torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))

output_path = "/data/hagmann_group/harmonization/graph_harmonization_final/outputs"
parameter_file_path = "/data/hagmann_group/harmonization/graph_harmonization_final/batch_scripts/parameters"

def main():

    # Parse arguments
    config = read_configuration_param()

    # Print the argument parameters and their values
    for arg, value in vars(config).items():
        if isinstance(value, bool):
            # If the argument is boolean, check its value
            if value:
                print(f'{arg} is set to True')
            else:
                print(f'{arg} is set to False')
        else:
            # Print non-boolean argument values
            print(f'{arg}: {value}')

    # Setup seeds
    seed = config.seed
    set_seed(seed)

    parameter_file_combination_path = os.path.join(parameter_file_path, config.maindir,
                                                   f"parameters_combinations_{config.parameter_combination_number}.txt")
    graph_evaluation_function(output_path, parameter_file_combination_path,
                              config.line_number, config.model_epoch_chosen, config.maindir, config.evaluation_set)

    print("Finished")

if __name__ == "__main__":
    main()


