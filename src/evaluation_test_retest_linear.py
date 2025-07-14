# import the packages

# To predict age and gender on test-retest matrices for the highest b-value and highest resolution
# The restricted csv file for retest data has no age information, meaning age is assumed to be same for test-retest subjects irrespective of the test-retest time interval

print("evaluation_test_retest_linear.py starting")

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr

from utils.data import load_graphs_dict, graph_matrix_vector
from utils.config_evaluation import read_configuration_param
from utils.functions_basic import set_seed

print("All packages imported")

data_behaviour_path = "/data/hagmann_group/behaviour_prediction/data_Urblauna/HCP_behavioral"
data_SC_path = "/data/hagmann_group/jagruti/dataset_1065"
output_path = "/data/hagmann_group/harmonization/graph_harmonization_final/outputs"
file_path = "/data/hagmann_group/harmonization/graph_harmonization_final/dataset_creation"

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

    saved_models_path = os.path.join(output_path, "models", config.maindir, "LINEAR",
                             f"SEED_{config.seed}", config.ckdir)

    if not os.path.exists(saved_models_path):
        print(f"{saved_models_path} not present")
        return False

    behavioral_restricted_dataset = pd.read_csv(os.path.join(data_behaviour_path, "hcp_behavioral_RESTRICTED.csv"),
                                                index_col=0)
    behavioral_dataset = pd.read_csv(os.path.join(data_behaviour_path, "hcp_behavioral.csv"), index_col=0)

    # adding the 'Gender' and 'Age' columns of the behavioral dataset and 'Age_in_Yrs' column of the restricted behavioral dataset
    behavioral_all_dataset = pd.concat(
        (behavioral_dataset[['Gender', 'Age']], behavioral_restricted_dataset['Age_in_Yrs']), axis=1)

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Fit and transform the 'Gender' column
    behavioral_all_dataset['Gender_Numeric'] = label_encoder.fit_transform(behavioral_all_dataset['Gender'])

    vectors_numbers_of_fibers = {}
    y_label_site = {}
    y_label_age = {}
    y_label_gender = {}
    set_x = {}
    sets = ['test_retest']
    datasets = ['HCP_DWI', 'HCP_DWI_Retest']

    for set1 in sets:
        vectors_numbers_of_fibers[set1], y_label_site[set1], y_label_age[set1], y_label_gender[set1], \
        set_x[set1]= create_dataset(config, set1, behavioral_all_dataset, datasets)

        print(f"Number of samples in the {set1} dataset : {len(vectors_numbers_of_fibers[set1])}")

    models = {}
    tasks = ['age', 'gender']
    models['age'] = joblib.load(os.path.join(saved_models_path, "model_age_Regression.pkl"))
    #print("SVM model for age prediction loaded - 'model_age_Regression.pkl'")
    models['gender'] = joblib.load(os.path.join(saved_models_path, "model_gender_SVM.pkl"))
    #print("SVM model for gender prediction loaded - 'model_gender_SVM.pkl'")

    for task in tasks:
        print(task)
        for set1 in sets:
            print(set1)
            if task == "age":
                predict_age(models[task], vectors_numbers_of_fibers[set1], y_label_site[set1], y_label_age[set1], set_x[set1], datasets)
            elif task == "gender":
                predict_gender(models[task], vectors_numbers_of_fibers[set1], y_label_site[set1], y_label_gender[set1], set_x[set1], datasets)

    print("Model prediction over")

def create_dataset(config, set1, behavioral_all_dataset, datasets):

    bvalue = 3000
    resolution_path = "1_25"
    subject_IDs = list(pd.read_csv(os.path.join(file_path, set1 + "_subjects.csv"), header=None, index_col=None)[0])

    track_subjects_IDs = []
    y_label_site_all = []
    y_label_age_all = []
    y_label_gender_all = []
    vectors_number_of_fibers_all = []
    set_x_all = []

    for i, dataset_x in enumerate(datasets):
        graphs_dict, subject_IDs_valid = load_graphs_dict(subject_IDs,
                                                          os.path.join(data_SC_path, dataset_x, "ds_HCP_bval_" + str(
                                                              bvalue) + "_anat_0_7_dwi_" + resolution_path),
                                                          config.scale, 0)
        _, vectors_number_of_fibers_dict = graph_matrix_vector(subject_IDs_valid, graphs_dict,
                                                               weight_value='number_of_fibers')

        for j, subject in enumerate(subject_IDs_valid):
            track_subjects_IDs.append(subject)
            vectors_number_of_fibers_all.append(vectors_number_of_fibers_dict[subject])
            y_label_age = behavioral_all_dataset.loc[subject]['Age_in_Yrs']
            y_label_gender = behavioral_all_dataset.loc[subject]['Gender_Numeric']
            y_label_site_all.append(3)
            y_label_age_all.append(y_label_age)
            y_label_gender_all.append(y_label_gender)
            set_x_all.append(i)

    print(f"No of subjects in {set1} is {len(track_subjects_IDs)}")
    print(f"No of unique subjects in {set1} is {len(set(track_subjects_IDs))}")

    vectors_number_of_fibers_all = np.stack(vectors_number_of_fibers_all)
    y_label_site_all = np.stack(y_label_site_all)
    y_label_age_all = np.stack(y_label_age_all)
    y_label_gender_all = np.stack(y_label_gender_all)
    set_x_all = np.stack(set_x_all)

    return vectors_number_of_fibers_all, y_label_site_all, y_label_age_all, y_label_gender_all, set_x_all

def predict_age(model, vectors_number_of_fibers, y_label_site, y_label_age, set_x, datasets):
    X_test = vectors_number_of_fibers
    y_test = y_label_age
    y_pred = model.predict(X_test)
    y_pred_integer = y_pred.copy()
    y_pred_integer = np.round(y_pred_integer)
    print(f"Lengths : {len(y_test)}, {len(y_pred)}")
    # Evaluate the model
    for i in np.unique(set_x):
        print(f"Dataset : {datasets[i]}")
        # Evaluate the model
        y_test_i = y_test[set_x==i]
        y_pred_i = y_pred[set_x==i]
        y_pred_integer_i = y_pred_integer[set_x==i]
        y_label_site_i = y_label_site[set_x==i]
        validation_loss = np.mean(np.abs(y_test_i - y_pred_i))
        validation_loss_integer = np.mean(np.abs(y_test_i - y_pred_integer_i))
        cod_score = r2_score(y_test_i, y_pred_i)
        cod_score_integer = r2_score(y_test_i, y_pred_integer_i)
        correlation = pearsonr(y_test_i, y_pred_i)
        correlation_integer = pearsonr(y_test_i, y_pred_integer_i)
        print(f"MAE is {validation_loss}, r2 score is {cod_score} and pearson correlation is {correlation}")
        print(f"For predicted values rounded to nearest integer")
        print(f"MAE is {validation_loss_integer}, r2 score is {cod_score_integer} and "
              f"pearson correlation is {correlation_integer}")
        for j in np.unique(y_label_site_i):
            print(f"Site : {j}")
            y_test_i_j = y_test_i[y_label_site_i == j]
            y_pred_i_j = y_pred_i[y_label_site_i == j]
            y_pred_integer_i_j = y_pred_integer_i[y_label_site_i == j]
            validation_loss = np.mean(np.abs(y_test_i_j - y_pred_i_j))
            validation_loss_integer = np.mean(np.abs(y_test_i_j - y_pred_integer_i_j))
            cod_score = r2_score(y_test_i_j, y_pred_i_j)
            cod_score_integer = r2_score(y_test_i_j, y_pred_integer_i_j)
            correlation = pearsonr(y_test_i_j, y_pred_i_j)
            correlation_integer = pearsonr(y_test_i_j, y_pred_integer_i_j)
            print(f"MAE is {validation_loss}, r2 score is {cod_score} and pearson correlation is {correlation}")
            print(f"For predicted values rounded to nearest integer")
            print(f"MAE is {validation_loss_integer}, r2 score is {cod_score_integer} and "
                  f"pearson correlation is {correlation_integer}")

def predict_gender(model, vectors_number_of_fibers, y_label_site, y_label_gender, set_x, datasets):
    X_test = vectors_number_of_fibers
    y_test = y_label_gender
    y_pred = model.predict(X_test)
    print(f"Lengths : {len(y_test)}, {len(y_pred)}")
    # Evaluate the model
    for i in np.unique(set_x):
        print(f"Dataset : {datasets[i]}")
        y_test_i = y_test[set_x == i]
        y_pred_i = y_pred[set_x == i]
        y_label_site_i = y_label_site[set_x == i]
        accuracy = accuracy_score(y_test_i, y_pred_i)
        print(f"Accuracy: {accuracy * 100}")
        #print("Classification Report:\n", classification_report(y_test_i, y_pred_i))
        for j in np.unique(y_label_site_i):
            print(f"Site : {j}")
            y_test_i_j = y_test_i[y_label_site_i == j]
            y_pred_i_j = y_pred_i[y_label_site_i == j]
            # Evaluate the model
            accuracy = accuracy_score(y_test_i_j, y_pred_i_j)
            print(f"Accuracy: {accuracy * 100}")
            #print("Classification Report:\n", classification_report(y_test_i_j, y_pred_i_j))

if __name__ == "__main__":
    main()


