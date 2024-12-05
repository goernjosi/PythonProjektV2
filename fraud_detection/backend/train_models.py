# Import required libraries for data processing, modeling, and evaluation
import pandas as pd  # For data manipulation and reading CSV
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from imblearn.over_sampling import SMOTE  # For handling class imbalance
# Import different model types for comparison
from sklearn.ensemble import RandomForestClassifier  # Ensemble learning method
from sklearn.tree import DecisionTreeClassifier  # Basic tree-based model
from sklearn.linear_model import LogisticRegression  # Linear model for binary classification
from xgboost import XGBClassifier  # Advanced gradient boosting model
# Import metrics for model evaluation
from sklearn.metrics import (classification_report, roc_auc_score, accuracy_score, 
                           precision_score, recall_score, f1_score, confusion_matrix)
import joblib  # For saving trained models

# Utility Functions
def prepare_data(df, target_column='Fraud', test_size=0.3, random_state=42):
    """
    Prepare and split data for modeling.
    This function separates features from the target variable and creates train/test splits.
    
    Parameters:
    - df: DataFrame containing all data
    - target_column: Name of the column we want to predict (default: 'Fraud')
    - test_size: Proportion of data to use for testing (default: 30%)
    - random_state: Seed for reproducibility
    
    Returns:
    - Training and test sets for both features (X) and target (y)
    """
    X = df.drop(columns=[target_column])  # Remove target column from features
    y = df[target_column]  # Isolate target variable
    # Split data while maintaining the same ratio of classes in both sets
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE to handle class imbalance in the training data.
    SMOTE creates synthetic examples of the minority class (fraud cases)
    to balance the dataset and prevent bias towards the majority class.
    
    Parameters:
    - X_train: Training features
    - y_train: Training target values
    - random_state: Seed for reproducibility
    
    Returns:
    - Balanced feature and target sets
    """
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)

def quick_evaluate_model(y_test, y_pred, y_pred_prob, model_name="Model"):
    """
    Evaluate model performance using multiple metrics.
    Prints classification report and ROC AUC score for comprehensive evaluation.
    
    Parameters:
    - y_test: True target values
    - y_pred: Model's predictions
    - y_pred_prob: Probability predictions for positive class
    - model_name: Name of the model being evaluated
    """
    print(f"{model_name} Results:")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("\nROC AUC Score:", roc_auc_score(y_test, y_pred_prob))

def train_models(data_path):
    """
    Main function to train and save multiple fraud detection models.
    Handles the entire pipeline from data loading to model saving.
    
    Parameters:
    - data_path: Path to the CSV file containing the data
    """
    # Load the dataset
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Prepare data for modeling
    print("Preparing data...")
    df = df.drop(columns=['Index'])  # Remove Index column as it's not useful for prediction
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Apply SMOTE to balance the training data
    print("Applying SMOTE to handle class imbalance...")
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    # Define all models with their specific parameters
    # Each model has different strengths and approaches to classification
    models = {
        'Random Forest': RandomForestClassifier(
            random_state=42  # For reproducibility
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42  # For reproducibility
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,      # For reproducibility
            solver='liblinear',   # Effective for small datasets
            max_iter=1000         # Maximum iterations for convergence
        ),
        'XGBoost': XGBClassifier(
            n_estimators=500,     # Number of boosting rounds
            learning_rate=0.1,    # Step size shrinkage to prevent overfitting
            max_depth=4,          # Maximum depth of each tree
            min_child_weight=3,   # Minimum sum of instance weight in child
            subsample=0.7,        # Fraction of samples used for tree building
            colsample_bytree=0.7, # Fraction of features used for tree building
            gamma=1,              # Minimum loss reduction for split
            reg_alpha=2,          # L1 regularization term
            reg_lambda=2,         # L2 regularization term
            random_state=42       # For reproducibility
        )
    }
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model on the balanced dataset
        model.fit(X_train_resampled, y_train_resampled)
        
        # Generate predictions
        y_pred = model.predict(X_test)  # Class predictions
        y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probability scores for fraud class
        
        # Evaluate model performance
        quick_evaluate_model(y_test, y_pred, y_pred_prob, name)
        
        # Save the trained model for later use
        save_path = f'models/{name.lower().replace(" ", "_")}.joblib'
        joblib.dump(model, save_path)
        print(f"{name} model saved to {save_path}")

# Entry point of the script
if __name__ == '__main__':
    train_models('creditcard.csv')