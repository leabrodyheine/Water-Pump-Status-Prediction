# Following code outline is from Optuna website example with changes for my own code.

"""
This script demonstrates the use of Optuna for hyperparameter optimization on 
various classifiers. It utilizes Optuna to find the best hyperparameters for 
Random Forest, Gradient Boosting, and Histogram Gradient Boosting classifiers 
based on a dataset processed by custom functions defined in 'part1' module.

The script reads and preprocesses data using functions from 'part1', then 
defines an objective function for Optuna to optimize. Optuna's study object 
is used to conduct trials to maximize cross-validated accuracy of the classifiers.
"""

# Imports
import optuna
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.linear_model
import sklearn.neural_network

from part1 import read_and_engineer_data
from part1 import process_data

# Get Data
raw_training_values, raw_test_values, y = read_and_engineer_data()

# Preprocess Data
preprocessor = get_preprocessor("_Target_", "_Scale_", raw_training_values)
X, X_test_preprocessed, y = (
    preprocessor.fit_transform(raw_training_values),
    preprocessor.transform(raw_test_values),
    y,
)


# Objective Function
def objective(trial):
    """
    Defines the objective function to optimize classifier parameters.

    This function is called by Optuna to evaluate a given set of hyperparameters.
    It suggests hyperparameters for RandomForestClassifier,
    GradientBoostingClassifier, and HistGradientBoostingClassifier, then
    evaluates their performance using cross-validation.

    Args:
        trial (optuna.trial.Trial): An Optuna trial object that suggests hyperparameters.

    Returns:
        float: The mean accuracy score from cross-validation, which Optuna attempts to maximize.
    """
    class_name = trial.suggest_categorical(
        "classifier",
        [
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "HistGradientBoostingClassifier",
        ],
    )

    # Random Forest
    if class_name == "RandomForestClassifier":
        # max features (cat), n_estimators (int)
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        class_obj = sklearn.ensemble.RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=10
        )
    # Gradient Boosting Classifier
    elif class_name == "GradientBoostingClassifier":
        gb_max_depth = trial.suggest_int("gb_max_depth", 2, 32, log=True)
        gb_learn_rate = trial.suggest_float("gb_learn_rate", 0.00001, 4, log=True)
        class_obj = sklearn.ensemble.GradientBoostingClassifier(
            max_depth=gb_max_depth, learning_rate=gb_learn_rate
        )
    # Hist Gradient Boosting Classifier
    elif class_name == "HistGradientBoostingClassifier":
        hg_max_iter = trial.suggest_int("gb_max_iter", 100, 100000, log=True)
        hg_max_depth = trial.suggest_int("gb_max_depth", 2, 32, log=True)
        hg_learn_rate = trial.suggest_float("gb_learn_rate", 0.00001, 4, log=True)
        class_obj = sklearn.ensemble.HistGradientBoostingClassifier(
            max_iter=hg_max_iter, max_depth=hg_max_depth, learning_rate=hg_learn_rate
        )

    # Create a pipeline with the preprocessor and the classifier
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", class_obj)]
    )

    # Use cross-validation with the pipeline
    score = sklearn.model_selection.cross_val_score(pipeline, X, y, n_jobs=-1, cv=3)
    accuracy_mean = score.mean()
    return accuracy_mean


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print(study.best_trial)
