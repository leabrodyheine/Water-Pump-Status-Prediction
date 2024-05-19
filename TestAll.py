import subprocess
"""
This script automates the process of running 'part1.py' with different combinations 
of preprocessing and model types for a machine learning task. It iterates over 
specified scaling options, categorical encoding techniques, and machine learning 
model types, executing 'part1.py' for each combination.

For each combination, 'part1.py' is called with the respective command-line arguments 
to train a model, and predictions are saved to uniquely named CSV files indicating 
the preprocessing and model type used.

Usage:
    Ensure 'part1.py' and the necessary CSV files ('train_input_file.csv', 
    'train_labels_file.csv', 'test_input_file.csv') are in the same directory. 
    Run this script without any arguments. 
"""
for s in ["_Scale_", "_NoScale_"]:
    for t in ["_Target_", "_Ordinal_", "_OneHot_"]:
        for m in ['LogisticRegression', 'RandomForestClassifier', 'GradientBoostingClassifier', 'HistGradientBoostingClassifier', 'MLPClassifier']:
            subprocess.run(["python", "part1.py", "train_input_file.csv", "train_labels_file.csv", "test_input_file.csv", s, t, m, f"test_predictions{s}{t}{m}.csv"])
            
