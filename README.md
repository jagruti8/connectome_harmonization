# Structural Connectome Harmonization Using Deep Learning

This repository contains the code for the paper  
**"Structural Connectome Harmonization Using Deep Learning: The Strength of Graph Neural Networks."**

The project focuses on harmonizing structural connectivity matrices acquired using different diffusion imaging parameters.

## Directory Structure

Inside the `batch_scripts` directory, there are two subfolders:

- `parameters/`: Contains text files for setting argument parameter options.
- `scripts/`: Contains script files for submitting jobs to a high-performance computing (HPC) environment.

Inside the `dataset_creation` directory, you will find several CSV files listing subjects used for various stages of the pipeline:

- `example_subjects.csv` – A small subset of 6 subjects from the training set, used to perform a quick test and verify that the code runs correctly.
- `test10_subjects.csv` – A list of 10 subjects from the test set, used to generate predictions for visual comparison of connectivity matrices across different methods.
- `test_retest_subjects.csv` – Contains the list of test and retest subjects.
- `test_subjects.csv` – The complete list of subjects used for final model testing.
- `train_subjects.csv` – Contains the subjects used for training the model.
- `val_subjects.csv` – Contains the subjects used for validating the model during training.

Inside the `src` directory, you will find the main Python files that are executed by the scripts in `batch_scripts/scripts/`.

Additionally, the `src` directory includes a `utils/` subdirectory, which contains modules for:

- Model definitions  
- Data loading functions  
- Graph metric calculations
- Configuration files
- Other core utility functions used throughout the `src` codebase





---

For questions or issues, feel free to open an issue or contact the repository maintainer.










