import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n{model_name} Performance:")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print(f"ROC AUC Score: {roc_auc_score(y_test, proba):.3f}")

def train_models(data_path):
    # Load data
    df = pd.read_csv(data_path)
    
    # Split features and target
    X = df.drop(columns=['Fraud', 'Index'])
    y = df['Fraud']
    
    # Train-Test-Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Train and evaluate Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_resampled, y_train_resampled)
    evaluate_model(rf, X_test, y_test, "Random Forest")
    
    # Train and evaluate Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_resampled, y_train_resampled)
    evaluate_model(dt, X_test, y_test, "Decision Tree")
    
    # Train and evaluate Logistic Regression
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_resampled, y_train_resampled)
    evaluate_model(lr, X_test, y_test, "Logistic Regression")
    
    # Train and evaluate XGBoost
    xgb = XGBClassifier(random_state=42)
    xgb.fit(X_train_resampled, y_train_resampled)
    evaluate_model(xgb, X_test, y_test, "XGBoost")
    
    # Save models
    joblib.dump(rf, 'models/random_forest.joblib')
    joblib.dump(dt, 'models/decision_tree.joblib')
    joblib.dump(lr, 'models/logistic_regression.joblib')
    joblib.dump(xgb, 'models/xgboost.joblib')

if __name__ == '__main__':
    train_models('creditcard.csv')