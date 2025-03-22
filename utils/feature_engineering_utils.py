import pandas as pd
import numpy as np



def transform_bank_features(input_file_path, output_file_path, separator=';'):

    df = pd.read_csv(input_file_path, sep=separator)

    # 1. no changes applied
    numeric_features = ['age', 'balance', 'campaign', 'pdays', 'previous']

    # 2. log transformation to numerical values
    df['log_duration'] = np.log1p(df['duration'])

    # 3. Cyclical encoding
    month_mapping = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df['month_num'] = df['month'].map(month_mapping)

    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

    # 4. poutcome
    poutcome_mapping = {'success': 1, 'failure': -1, 'unknown': 0, 'other': 0}
    df['poutcome_num'] = df['poutcome'].map(poutcome_mapping)

    # 5. Binary encoding
    binary_features = ['default', 'housing', 'loan']
    for feature in binary_features:
        df[f'{feature}_bin'] = df[feature].map({'yes': 1, 'no': 0})

    # 6. one-hot
    categorical_features = ['job', 'marital', 'education', 'contact']
    df_encoded = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features)

    features_to_keep = (
        numeric_features + 
        ['log_duration', 'month_sin', 'month_cos',
         'poutcome_num', 'default_bin', 'housing_bin', 'loan_bin'] +
        [col for col in df_encoded.columns if any(
            col.startswith(f'{feat}_') for feat in categorical_features
        )] +
        ['y']  
    )

    features_to_keep = [col for col in features_to_keep if col in df_encoded.columns]

    final_df = df_encoded[features_to_keep].copy()

    # 7. yes -> 1; no -> 0
    if 'y' in final_df.columns:
        final_df.loc[:, 'y'] = final_df['y'].map({'yes': 1, 'no': 0})

    final_df.to_csv(output_file_path, index=False)
    
    return final_df


if __name__ == "__main__":
    # use case
    from utils.path_utils import get_path_from_project_root
    
    input_file_path = get_path_from_project_root("data", "raw", "bank.csv")
    output_file_path = get_path_from_project_root("data", "processed", "bank_feature_transformed.csv")
    
    transform_bank_features(input_file_path, output_file_path)