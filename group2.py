# Group 2:
# 
# Oversampling (SMOTENC) was applied to the training set in this group. 
# Specifically, all categorical attributes intended for one-hot encoding 
# were first label encoded. Oversampling was then performed using SMOTENC, 
# which is suitable for handling nominal attributes without introducing 
# unintended ordinal relationships that may result from label encoding. 
# After oversampling, the data were transformed back to their original 
# format and subsequently passed through the feature engineering work 
# to produce the training set for Group 2.

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC

from utils.feature_engineering_utils import transform_bank_features
from utils.path_utils import get_path_from_project_root


def preprocess_with_smotenc(input_file, intermediate_file, output_file, separator=';'):
    df = pd.read_csv(input_file, sep=separator)
    
    categorical_features = [
        'job', 'marital', 'education', 'default', 
        'housing', 'loan', 'contact', 'month', 'poutcome'
    ]
    
    # Label Encoding
    encoders = {}
    encoded_df = df.copy()
    
    for feature in categorical_features:
        le = LabelEncoder()
        encoded_df[feature] = le.fit_transform(df[feature])
        encoders[feature] = {
            'encoder': le,
            'categories': le.classes_,
            'mapping': dict(zip(le.classes_, le.transform(le.classes_)))
        }
    
    # oversample by SMOTENC
    X = encoded_df.drop(columns=['y'])
    y = (encoded_df['y'] == 'yes').astype(int)
    
    categorical_indices = [X.columns.get_loc(feature) for feature in categorical_features]
        
    target_yes_count = 3700
    sampling_strategy = target_yes_count / sum(y == 0)
    
    smote_nc = SMOTENC(
        categorical_features=categorical_indices,
        sampling_strategy=sampling_strategy,
        random_state=42,
        k_neighbors=5
    )
    
    X_resampled, y_resampled = smote_nc.fit_resample(X, y)
    
    resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_df['y'] = y_resampled

    # transform them back to original format
    for feature in categorical_features:
        inverse_mapping = {v: k for k, v in encoders[feature]['mapping'].items()}
        resampled_df[feature] = resampled_df[feature].map(inverse_mapping)
    
    resampled_df['y'] = resampled_df['y'].map({1: 'yes', 0: 'no'})
    
    resampled_df.to_csv(intermediate_file, index=False, sep=separator)
    
    # do feature engineering
    transform_bank_features(intermediate_file, output_file, separator=separator)

    return output_file

def main():    
    input_file = get_path_from_project_root("data", "raw", "bank.csv")
    intermediate_file = get_path_from_project_root("data", "interim", "bank_resampled.csv")
    output_file = get_path_from_project_root("data", "processed", "bank_g2.csv")

    preprocess_with_smotenc(input_file, intermediate_file, output_file)

if __name__ == "__main__":
    main()
