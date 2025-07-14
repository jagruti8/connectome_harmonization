# import the packages

print("evaluation_training_linear.py starting")

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr

from utils.data import load_graphs_dict, graph_matrix_vector
from utils.config_evaluation import read_configuration_param
from utils.functions_basic import set_seed

print("All packages imported")

data_SC_path = "/data/hagmann_group/jagruti/dataset_1065/HCP_DWI"
data_behaviour_path = "/data/hagmann_group/behaviour_prediction/data_Urblauna/HCP_behavioral"
file_path = "/data/hagmann_group/harmonization/graph_harmonization_final/dataset_creation"
output_path = "/data/hagmann_group/harmonization/graph_harmonization_final/outputs"

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

    save_path = os.path.join(output_path, "models", config.maindir, "LINEAR",
                             f"SEED_{config.seed}", config.ckdir)

    os.makedirs(save_path, exist_ok=True)

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
    sets = ['train','val', 'test']

    for set1 in sets:
        vectors_numbers_of_fibers[set1], y_label_site[set1], y_label_age[set1], y_label_gender[set1] = \
        create_dataset(set1, config, behavioral_all_dataset)

        print(f"Number of samples in the {set1} dataset : {len(vectors_numbers_of_fibers[set1])}" )

    models = {}
    tasks = ['age', 'gender']

    models['age'] = age_training(vectors_numbers_of_fibers['train'], y_label_site['train'], y_label_age['train'])
    models['gender'] = gender_training(vectors_numbers_of_fibers['train'], y_label_site['train'], y_label_gender['train'])

    for task in tasks:
        print(task)
        for set1 in sets:
            print(set1)
            if task == "age":
                predict_age(models[task], vectors_numbers_of_fibers[set1], y_label_site[set1], y_label_age[set1])
            elif task == "gender":
                predict_gender(models[task], vectors_numbers_of_fibers[set1], y_label_site[set1], y_label_gender[set1])

    joblib.dump(models['age'], os.path.join(save_path, "model_age_Regression.pkl"))
    print("SVM model for age prediction saved as 'model_age_Regression.pkl'")
    joblib.dump(models['gender'], os.path.join(save_path, "model_gender_SVM.pkl"))
    print("SVM model for gender prediction saved as 'model_gender_SVM.pkl'")

    print("Model prediction over")

def create_dataset(set1, config, behavioral_all_dataset):

    """load the graphs, the matrices and the sites for the subject set"""

    subject_IDs = list(pd.read_csv(os.path.join(file_path, set1 + "_subjects.csv"), header=None, index_col=None)[0])

    bvalues = [1000, 3000]
    resolutions_paths = ["2_3", "1_25"]

    track_subjects_IDs = []
    y_label_site_all = []
    y_label_age_all = []
    y_label_gender_all = []
    vectors_number_of_fibers_all = []

    for j, bval in enumerate(bvalues):
        for k, res in enumerate(resolutions_paths):
            # loading the .gpickle files for the chosen subjects

            graphs_dict, subject_IDs_valid = load_graphs_dict(subject_IDs,
                                                                  os.path.join(data_SC_path, "ds_HCP_bval_" + str(bval) + "_anat_0_7_dwi_" + res),
                                                                  config.scale, 0)

            print(f"Length of valid subject IDs : {len(subject_IDs_valid)}")

            # getting the matrices for edge attribute "normalized_fiber_density"
            _, vectors_number_of_fibers_input = graph_matrix_vector(subject_IDs_valid, graphs_dict,
                                                                     weight_value='number_of_fibers')

            for subject_x, vectors_number_of_fibers_x in vectors_number_of_fibers_input.items():
                track_subjects_IDs.append(subject_x)
                vectors_number_of_fibers_all.append(vectors_number_of_fibers_x)
                y_label_site = j*2+k
                y_label_age = behavioral_all_dataset.loc[subject_x]['Age_in_Yrs']
                y_label_gender = behavioral_all_dataset.loc[subject_x]['Gender_Numeric']
                y_label_site_all.append(y_label_site)
                y_label_age_all.append(y_label_age)
                y_label_gender_all.append(y_label_gender)

    print(f"No of subjects in {set1} is {len(track_subjects_IDs)}")
    print(f"No of unique subjects in {set1} is {len(set(track_subjects_IDs))}")

    vectors_number_of_fibers_all = np.stack(vectors_number_of_fibers_all)
    y_label_site_all = np.stack(y_label_site_all)
    y_label_age_all = np.stack(y_label_age_all)
    y_label_gender_all = np.stack(y_label_gender_all)

    return vectors_number_of_fibers_all, y_label_site_all, y_label_age_all, y_label_gender_all

def age_training(vectors_number_of_fibers, y_label_site, y_label_age):
    ID_x = np.where(y_label_site == 3)
    X_train = vectors_number_of_fibers[ID_x]
    y_train = y_label_age[ID_x]
    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def gender_training(vectors_number_of_fibers, y_label_site, y_label_gender):
    ID_x = np.where(y_label_site == 3)
    X_train = vectors_number_of_fibers[ID_x]
    y_train = y_label_gender[ID_x]
    # Train an SVM classifier
    model = SVC(kernel='rbf', C=10.0)  # You can change kernel to 'rbf', 'poly', etc.
    model.fit(X_train, y_train)
    return model

def predict_age(model, vectors_number_of_fibers, y_label_site, y_label_age):
    X_test = vectors_number_of_fibers
    y_test = y_label_age
    y_pred = model.predict(X_test)
    y_pred_integer = y_pred.copy()
    y_pred_integer = np.round(y_pred_integer)
    print(f"Lengths : {len(y_test)}, {len(y_pred)}")
    # Evaluate the model
    validation_loss = np.mean(np.abs(y_test - y_pred))
    validation_loss_integer = np.mean(np.abs(y_test - y_pred_integer))
    cod_score = r2_score(y_test, y_pred)
    cod_score_integer = r2_score(y_test, y_pred_integer)
    correlation = pearsonr(y_test, y_pred)
    correlation_integer = pearsonr(y_test, y_pred_integer)
    print(f"MAE is {validation_loss}, r2 score is {cod_score} and pearson correlation is {correlation}")
    print(f"For predicted values rounded to nearest integer")
    print(f"MAE is {validation_loss_integer}, r2 score is {cod_score_integer} and "
          f"pearson correlation is {correlation_integer}")
    for i in np.unique(y_label_site):
        print(f"Site : {i}")
        y_test_i = y_test[y_label_site==i]
        y_pred_i = y_pred[y_label_site==i]
        y_pred_integer_i = y_pred_integer[y_label_site==i]
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

def predict_gender(model, vectors_number_of_fibers, y_label_site, y_label_gender):
    X_test = vectors_number_of_fibers
    y_test = y_label_gender
    y_pred = model.predict(X_test)
    print(f"Lengths : {len(y_test)}, {len(y_pred)}")
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    for i in np.unique(y_label_site):
        print(f"Site : {i}")
        y_test_i = y_test[y_label_site==i]
        y_pred_i = y_pred[y_label_site==i]
        # Evaluate the model
        accuracy = accuracy_score(y_test_i, y_pred_i)
        print(f"Accuracy: {accuracy * 100}")
        print("Classification Report:\n", classification_report(y_test_i, y_pred_i))

if __name__ == "__main__":
    main()
