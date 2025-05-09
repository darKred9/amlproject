# Group 1:
# 
# This group serves as the baseline. The original
# bank.csv file was directly subjected to feature 
# engineering to generate the training set for 
# Group 1.

from utils.feature_engineering_utils import transform_bank_features
from utils.path_utils import get_path_from_project_root

def group_1():
    input_file_path = get_path_from_project_root("data", "raw", "bank.csv")
    output_file_path = get_path_from_project_root("data", "processed", "bank_g1.csv")
    
    transform_bank_features(input_file_path, output_file_path)

def main():
    group_1()

if __name__ == "__main__":
    main()