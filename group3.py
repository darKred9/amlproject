# Group 3 random oversampling
# 
# Oversampling (Random Oversampling) was also applied 
# to the training set in this group. Specifically, 
# random oversampling was performed directly on the 
# original bank.csv dataset. The resulting data were 
# then passed through the feature engineering work 
# to generate the training set for Group 3.

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from utils.feature_engineering_utils import transform_bank_features
from utils.path_utils import get_path_from_project_root


def perform_random_oversampling(input_file, intermediate_file, output_file, separator=';'):

    df = pd.read_csv(input_file, sep=separator)

    X = df.drop(columns=['y'])
    y = df['y']

    target_minority = 3700
    sampling_strategy = {'yes': target_minority}
    
    random_oversampler = RandomOverSampler(
        sampling_strategy=sampling_strategy,
        random_state=42
    )
    
    X_resampled, y_resampled = random_oversampler.fit_resample(X, y)
    
    resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_df['y'] = y_resampled
    
    resampled_df.to_csv(intermediate_file, index=False, sep=separator)

    transform_bank_features(intermediate_file, output_file, separator=separator)

    return output_file


def main():

    input_file = get_path_from_project_root("data", "raw", "bank.csv")
    intermediate_file = get_path_from_project_root("data", "interim", "bank_random_oversampled.csv")
    output_file = get_path_from_project_root("data", "processed", "bank_g3.csv")
    
    perform_random_oversampling(input_file, intermediate_file, output_file)


if __name__ == "__main__":
    main()