# Water-Pump-Status-Prediction

This project implements machine learning models to predict the status of water pumps in Tanzania using data from DrivenData's competition. The project includes preprocessing steps, model evaluation using cross-validation, and hyperparameter optimization with Optuna.

## Libraries/Imports Used
- optuna
- scikitlearn
- pandas
- NumPy
- subprocess
- argparse
- os

## Project Structure
- **part1.py**: This script trains and evaluates various machine learning models with different preprocessing techniques using cross-validation.
- **part2.py**: This script performs hyperparameter optimization using Optuna.
- **TestAll.py**: This script automates the process of running `part1.py` with different combinations of preprocessing and model types.

## Usage

### Running Part 1
To evaluate the models with different preprocessing techniques and model types, you can manually run `part1.py` with the desired arguments. However, it's more efficient to use `TestAll.py` to automate this process.

**Example Command:**
python part1.py `<train-input-file> <train-labels-file> <test-input-file> <numerical-preprocessing> <categorical-preprocessing> <model-type> <test-prediction-output-file>`

**Automate Part 1 with `TestAll.py`:**
The `TestAll.py` script loops through all the command-line argument options instead of manually writing the command in the terminal for each variation of a model’s training. Run this file in your IDE.

### Running Part 2
For hyperparameter optimization, run the `part2.py` script. It uses Optuna to find the best hyperparameters for different classifiers.

## Data
The dataset for this project is sourced from the DrivenData competition Pump it Up: Data Mining the Water Table. It consists of training data (input features and labels) and test data (input features only).

## Preprocessing
Various preprocessing steps are implemented, including handling categorical features, dealing with missing values, scaling numerical values, and dealing with datetime features. For categorical features, I use three types of encoding: O
- OneHotEncoder
- OrdinalEncoder
- TargetEncoder

For numerical features, I consider two options: no scaling and StandardScaler.

## Machine Learning Models
I evaluate the performance of five families of machine learning models:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Histogram-based Gradient Boosting Classifier
- Multi-layer Perceptron Classifier

## Hyperparameter Optimization
The project uses Optuna for hyperparameter optimization. The optimization process includes defining the configuration space and evaluating the performance using cross-validation.
