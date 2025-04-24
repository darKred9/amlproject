# Group 4 read additional data & Split the testset
# 
# A total of 3,700 instances were randomly selected 
# from bank_full.csv and added to bank.csv as 
# additional records. The resulting dataset 
# augmented by real additional records was then 
# used as the training set for Group 4.

import pandas as pd
from utils.feature_engineering_utils import transform_bank_features
from utils.path_utils import get_path_from_project_root
from sklearn.model_selection import train_test_split


def create_test_set_and_enrich_training(filtered_data_path, 
                                        original_data_path,
                                        test_output_path, 
                                        enriched_output_path,
                                        test_size=1000,
                                        yes_samples_to_add=3700,
                                        separator=';'):


    filtered_df = pd.read_csv(filtered_data_path, sep=separator)
    
    min_yes = min(sum(filtered_df['y'] == 'yes'), test_size // 2)
    min_no = min(sum(filtered_df['y'] == 'no'), test_size // 2)
    actual_test_size = min_yes + min_no
    
    stratify_col = filtered_df['y']
    
    remaining_df, test_df = train_test_split(
        filtered_df,
        test_size=actual_test_size,
        stratify=stratify_col,
        random_state=42
    )
    
    test_raw_path = test_output_path.replace('.csv', '_raw.csv')
    test_df.to_csv(test_raw_path, index=False, sep=separator)
    
    transform_bank_features(test_raw_path, test_output_path, separator=separator)

    remaining_yes_df = remaining_df[remaining_df['y'] == 'yes']
    selected_yes_df = remaining_yes_df.sample(n=yes_samples_to_add, random_state=42)
    
    original_df = pd.read_csv(original_data_path, sep=separator)
    enriched_df = pd.concat([original_df, selected_yes_df], ignore_index=True)

    enriched_raw_path = enriched_output_path.replace('.csv', '_raw.csv')
    enriched_df.to_csv(enriched_raw_path, index=False, sep=separator)

    transform_bank_features(enriched_raw_path, enriched_output_path, separator=separator)

    return test_output_path, enriched_output_path


def main():

    filtered_data_path = get_path_from_project_root("data", "raw", "bank_full_filtered.csv")
    original_data_path = get_path_from_project_root("data", "raw", "bank.csv")
    test_output_path = get_path_from_project_root("data", "test", "test_1000_stratified.csv")
    enriched_output_path = get_path_from_project_root("data", "processed", "bank_g4.csv")
    
    create_test_set_and_enrich_training(
        filtered_data_path=filtered_data_path,
        original_data_path=original_data_path,
        test_output_path=test_output_path,
        enriched_output_path=enriched_output_path,
        test_size=1000,  # 500 yes + 500 no
        yes_samples_to_add=3700
    )


if __name__ == "__main__":
    main()
