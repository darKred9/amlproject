import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils.path_utils import get_path_from_project_root

def train_and_evaluate_xgboost(train_file, test_file, model_params=None):

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    X_train = train_data.drop('y', axis=1)
    y_train = train_data['y']
    
    X_test = test_data.drop('y', axis=1)
    y_test = test_data['y']
    
    if model_params is None:
        model_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'verbosity': 0,
            'random_state': 42
        }
    
    model = xgb.XGBClassifier(**model_params)
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # will be shown in the table
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # heat map
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        'model': model,
        'train_file': train_file,
        'test_file': test_file,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'feature_importance': model.feature_importances_,
        'feature_names': X_train.columns.tolist()
    }
    return results


def plot_results(results_list, output_dir=None):
    model_names = [os.path.basename(r['train_file']).replace('.csv', '') for r in results_list]
    
    plt.figure(figsize=(12, 8))
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    metric_values = {metric: [r[metric] for r in results_list] for metric in metrics}
    
    bar_width = 0.15
    index = np.arange(len(model_names))
    
    for i, metric in enumerate(metrics):
        plt.bar(index + i * bar_width, metric_values[metric], bar_width, 
                label=metric.replace('_', ' ').title())
    
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Model Evaluation Metrics Comparison')
    plt.xticks(index + bar_width * 2, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
    
    # confusion matrix heatmaps
    plt.figure(figsize=(15, 10))
    for i, result in enumerate(results_list):
        plt.subplot(2, 2, i+1)
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'], 
                   yticklabels=['Negative', 'Positive'])
        plt.title(f"{model_names[i]}\nConfusion Matrix")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
    
    # feature importance
    plt.figure(figsize=(15, 10))
    for i, result in enumerate(results_list):
        # top 15
        importance = result['feature_importance']
        feature_names = result['feature_names']
        indices = np.argsort(importance)[-15:]
        
        plt.subplot(2, 2, i+1)
        plt.barh(range(len(indices)), importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.title(f"{model_names[i]}\nFeature Importance (Top 15)")
        plt.xlabel('Importance')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    
    # result table
    results_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': [r['accuracy'] for r in results_list],
        'Precision': [r['precision'] for r in results_list],
        'Recall': [r['recall'] for r in results_list],
        'F1 Score': [r['f1_score'] for r in results_list],
        'AUC': [r['auc'] for r in results_list],
    })
    
    if output_dir:
        results_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)


def main():
    train_files = [
        get_path_from_project_root("data", "processed", "bank_g1.csv"),
        get_path_from_project_root("data", "processed", "bank_g2.csv"),
        get_path_from_project_root("data", "processed", "bank_g3.csv"),
        get_path_from_project_root("data", "processed", "bank_g4.csv")
    ]
    
    test_file = get_path_from_project_root("data", "test", "test_1000_stratified.csv")
    output_dir = get_path_from_project_root("results", "xgboost_evaluation")

    # # output path of the tuned model
    # output_dir = get_path_from_project_root("results", "xgboost_recall_focused")

    model_params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'verbosity': 0,
        'random_state': 42,
        'scale_pos_weight': 1
    }

    # # tuned parameters
    # model_params = {
    #     'max_depth': 6, 
    #     'learning_rate': 0.08,
    #     'n_estimators': 200, 
    #     'objective': 'binary:logistic',
    #     'eval_metric': 'logloss',
    #     'subsample': 0.9,
    #     'colsample_bytree': 0.9,    
    #     'reg_lambda': 0.8,           
    #     'min_child_weight': 1, 
    #     'scale_pos_weight': 1.2, 
    #     'verbosity': 0,
    #     'random_state': 42,
    #     'prediction_threshold': 0.35  
    # }
    
    
    results_list = []
    for train_file in train_files:
        results = train_and_evaluate_xgboost(train_file, test_file, model_params)
        if results:
            results_list.append(results)

    plot_results(results_list, output_dir)


if __name__ == "__main__":
    main()
