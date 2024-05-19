#IMPORTS
import subprocess
import argparse
import os

# Data handling and processing
import pandas as pd
import numpy as np

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from sklearn import set_config

# Model selection and evaluation
from sklearn.model_selection import train_test_split, cross_val_score, KFold

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Utilities
from sklearn.metrics import accuracy_score


#SCRIPTING
def get_args():
    """
    Parse and return command line arguments required for script execution.
    
    Returns:
        Namespace: A Namespace object containing the arguments as attributes.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate a water pump status prediction model.")
    parser.add_argument('train_input_file', type=str, help='Path to the training input file')
    parser.add_argument('train_labels_file', type=str, help='Path to the training labels file')
    parser.add_argument('test_input_file', type=str, help='Path to the test input file')
    parser.add_argument('numerical_preprocessing', type=str, help='Numerical preprocessing method')
    parser.add_argument('categorical_preprocessing', type=str, help='Categorical preprocessing method')
    parser.add_argument('model_type', type=str, help='Machine learning model type')
    parser.add_argument('test_prediction_output_file', type=str, help='Path to the output prediction file')

    args = parser.parse_args()
    return args
    
    
def read_and_engineer_data():
    """
    Read the dataset CSV files and perform initial data engineering.
    
    The function assumes specific paths for the dataset files and 
    performs basic data cleaning and feature engineering tasks such 
    as date conversion and age calculation.

    Returns:
        tuple: Returns a tuple containing the training values DataFrame, 
        test values DataFrame, and training labels Series.   
    """    
    # Paths to the datasets
    test_set_path = "702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv"
    training_labels_path = "0bf8bc6e-30d0-4c50-956a-603fc693d966.csv"
    training_values_path = "4910797b-ee55-40a7-8668-10efd5c1b960.csv"

    # Load the datasets
    raw_test_values = pd.read_csv(test_set_path)
    raw_training_labels = pd.read_csv(training_labels_path)
    raw_training_values = pd.read_csv(training_values_path)

    #CLEANING DATA
    # Conversion of 'date_recorded' to datetime
    raw_training_values['date_recorded'] = pd.to_datetime(raw_training_values['date_recorded'])
    raw_test_values['date_recorded'] = pd.to_datetime(raw_test_values['date_recorded'])

    # Simple feature engineering: calculating waterpoint age
    current_year = pd.to_datetime('now').year
    raw_training_values['construction_year'] = raw_training_values['construction_year'].replace(0, np.nan)
    raw_training_values['age'] = current_year - raw_training_values['construction_year']
    raw_test_values['construction_year'] = raw_test_values['construction_year'].replace(0, np.nan)
    raw_test_values['age'] = current_year - raw_test_values['construction_year']

    y_train = raw_training_labels['status_group']  
    
    return raw_training_values, raw_test_values, y_train


def identify_cols(raw_training_values):
    """
    Identify and separate the categorical and numerical columns.
    
    Args:
        raw_training_values (DataFrame): The training data before preprocessing.
        
    Returns:
        tuple: A tuple containing the list of categorical column names and 
        the list of numerical column names.
    """
# Identify categorical and numerical columns
    categorical_columns = [col for col in raw_training_values.columns if raw_training_values[col].dtype == 'object' and col not in ['date_recorded']]
    numerical_columns = ["amount_tsh", "gps_height", "longitude", "latitude", "num_private", "region_code", "district_code", "population", "age"]
    
    return categorical_columns, numerical_columns
      
      
def get_preprocessor(categorical_preprocessing, numerical_preprocessing, raw_training_values):
    """
    Create a preprocessing pipeline based on specified methods for 
    categorical and numerical data.
    
    Args:
        categorical_preprocessing (str): The chosen method for preprocessing 
                                         categorical features.
        numerical_preprocessing (str): The chosen method for preprocessing 
                                       numerical features.
        raw_training_values (DataFrame): The training data used to identify 
                                         column datatypes.
    
    Returns:
        ColumnTransformer: A configured ColumnTransformer object with separate 
        pipelines for numerical and categorical data.
    """
    # Get col variables
    categorical_columns, numerical_columns = identify_cols(raw_training_values)
    
    # Numerical transformer with imputation and optional scaling
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()) if numerical_preprocessing == '_Scale_' else ('passthrough', 'passthrough')
    ])
    
    # Categorical transformer with imputation and encoding
    if categorical_preprocessing == '_OneHot_':
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=100))
        ])
    elif categorical_preprocessing == '_Ordinal_':
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
    elif categorical_preprocessing == '_Target_':
        categorical_transformer = TargetEncoder()
        
    # Combine into a single preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])
    preprocessor.set_output(transform="pandas")
    return preprocessor



def process_data(raw_training_values, raw_test_values, y_train, numerical_preprocessing, categorical_preprocessing):
    """
    Process the training and test data using the defined preprocessing pipelines.
    
    Args:
        raw_training_values (DataFrame): The raw training data.
        raw_test_values (DataFrame): The raw test data.
        y_train (Series): The training data labels.
        numerical_preprocessing (str): The chosen method for preprocessing 
                                       numerical features.
        categorical_preprocessing (str): The chosen method for preprocessing 
                                         categorical features.
    
    Returns:
        tuple: A tuple containing the preprocessed training data and 
        preprocessed test data.
    """
    preprocessor = get_preprocessor(categorical_preprocessing, numerical_preprocessing, raw_training_values)
    X_train, X_val, y_train, y_val = train_test_split(raw_training_values, y_train, test_size=0.2, random_state=42)
    X_train_preprocessed = preprocessor.fit_transform(X_train, y_train)
    X_val_preprocessed = preprocessor.transform(X_val)
    X_test_preprocessed = preprocessor.transform(raw_test_values)
    return X_train_preprocessed, X_val_preprocessed, X_test_preprocessed, y_train, y_val


#MODEL TRAINING
def train_and_evaluate_model(model_type, X_train_preprocessed, y_train, X_test_preprocessed):
    """
    Train and evaluate a machine learning model based on the specified type.
    
    Args:
        model_type (str): The type of machine learning model to train.
        X_train_preprocessed (DataFrame): The preprocessed training data features.
        y_train (Series): The training data labels.
        X_test_preprocessed (DataFrame): The preprocessed test data features.
    
    Returns:
        ndarray: The test predictions made by the trained model on the test dataset.
    """
    args = get_args()
    
    # Map model type argument to actual classifier
    models = {
        'LogisticRegression': LogisticRegression(),
        'RandomForestClassifier': RandomForestClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'HistGradientBoostingClassifier': HistGradientBoostingClassifier(),
        'MLPClassifier': MLPClassifier(max_iter=10000)
    }

    classifier = models.get(model_type)
        
    if classifier is None:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    preprocessor = get_preprocessor(args.categorical_preprocessing, args.numerical_preprocessing, raw_training_values)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    
    kf = KFold(n_splits=5)
    cv_scores = []
    for train_index, test_index in kf.split(X_train_preprocessed):
        X_train_fold, X_test_fold = X_train_preprocessed.iloc[train_index], X_train_preprocessed.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        pipeline.fit(X_train_fold, y_train_fold)
        score = pipeline.score(X_test_fold, y_test_fold)
        cv_scores.append(score)
    print(np.mean(cv_scores))
    
    #Outfile CV Score
    output_folder1 = f"Model Accuracy Part 1/{model_type}"
    os.makedirs(output_folder1, exist_ok=True)
    results_file_path = os.path.join(output_folder1, f"{model_type}{args.numerical_preprocessing}{args.categorical_preprocessing}Results.txt")

    with open(results_file_path, 'w') as f:
        print(f"{model_type} {args.numerical_preprocessing} {args.categorical_preprocessing} CV training accuracy scores:", cv_scores, file=f)
        print(f"{model_type} {args.numerical_preprocessing} {args.categorical_preprocessing} CV training average accuracy:", np.mean(cv_scores), "\n", file=f)
    
    pipeline.fit(X_train_preprocessed, y_train)
    test_predictions = classifier.predict(X_test_preprocessed)

    print("finished", model_type, "\n")
    return test_predictions



# Main execution logic 
if __name__ == "__main__":
    """
    Main execution function of the script.
    
    Parses arguments, reads and processes data, trains and evaluates a model,
    and saves the prediction results to a file. This function orchestrates the
    execution of the script when called directly from the command line.
    """
    args = get_args()
    
    # Get raw data
    raw_training_values, raw_test_values, y_train = read_and_engineer_data()
    
    #Preprocess data
    X_train_preprocessed, X_val_preprocessed, X_test_preprocessed, y_train, y_val = process_data(raw_training_values, raw_test_values, y_train, args.numerical_preprocessing, args.categorical_preprocessing)
    
    # test_predictions, test_score = train_and_evaluate_model(args.model_type, raw_training_values, y_train, raw_test_values, preprocessor)
    test_predictions = train_and_evaluate_model(args.model_type, X_train_preprocessed, y_train, X_val_preprocessed, y_val, X_test_preprocessed)
    print(test_predictions.shape)
    print(raw_test_values.shape)
    print(X_train_preprocessed.shape)
    print(y_train.shape)

    # Save the predictions to a CSV file-- this is not working for some reason
    output_folder = f"DF predictions/{args.model_type}"
    os.makedirs(output_folder, exist_ok=True)
    output_file_name  = args.test_prediction_output_file
    output_path = os.path.join(output_folder, output_file_name)
    output_df = pd.DataFrame({'id': raw_test_values['id'], 'status_group': test_predictions})
    output_df.to_csv(output_path, index=False)
