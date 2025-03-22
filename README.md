# Course Project for Applied Machine Learning

Spring 2025

Author: Yuchen Wang

## Project Structure

- `data`
    - `data/interim`: intermediate results
    - `data/processed`:
        - `data/processed/bank_g1.csv`: Group 1's training set (processed)
        - `data/processed/bank_g2.csv`: Group 2's training set (processed)
        - `data/processed/bank_g3.csv`: Group 3's training set (processed)
        - `data/processed/bank_g4.csv`: Group 4's training set (processed)
        - `data/processed/bank_g4_raw.csv`: Group 4's data augmented by additional real data and has not been processed by feature engineering work yet.
    - `data/raw`: raw csv's
        - `bank_full_filtered.csv`: records in bank-full.csv but not in bank.csv
    - `data/test`:
        - `test_1000_stratified.csv`: the test set with 1000 records (500 yes + 500 no)
- `results`
    - `results/data_overview`: charts that show the basic information of `bank_g1.csv`/`bank_g2.csv`/`bank_g3.csv`/`bank_g4.csv`
    - `results/xgboost_evaluation`: charts that show the evaluation results of the four groups by XGBoost
    - `results/xgboost_recall_focused`: results from the tuned XGBoost model with a particular emphasis on recall improvement.
- `utils`: a python package
    - `utils/feature_engineering_utils.py`: the feature engineering work mentioned in the report. It accepts a csv file as input and do feature transformation on it and export the result csv file.
        ``` python
        # use case

        from utils.feature_engineering_utils.py import transform_bank_features
        from utils.path_utils import get_path_from_project_root

        input_file_path = get_path_from_project_root("data", "raw", "bank.csv")
        output_file_path = get_path_from_project_root("data", "processed", "bank_feature_transformed.csv")
        
        transform_bank_features(input_file_path, output_file_path)

        ```
    - `utils/path_utils`: a tool function that deals with the relative path of this project.
        ``` python
        # use case
        from utils.path_utils import get_path_from_project_root
        g1_data_path = get_path_from_project_root('data', 'bank_g1.csv')
        ```
- `group1.py`: script used to generate the training set for group1
- `group2.py`: script used to generate the training set for group2
- `group3.py`: script used to generate the training set for group3
- `group4.py`: script used to generate the training set for group4

- `chart_data_overview.py`: script used to draw data overview plots
- `Exp_XGBoost`: script used to train models over traing sets of group 1-4 (original/tuned)

## Reference

- Dataset: https://archive.ics.uci.edu/dataset/222/bank+marketing