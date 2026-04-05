import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def run_churn_model(input_path="data/features.csv", output_path="data/churn_risks.csv", metrics_path="data/model_metrics.csv", model_path="data/churn_model.pkl"):
    print(f"Loading features from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Select features
    features = [
        'total_lifetime_orders',
        'total_spent',
        'total_items_bought',
        'average_order_value',
        'average_days_between_orders'
    ]
    
    X = df[features]
    y = df['Churn']
    
    print(f"Dataset Shape: {X.shape}")
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred))
    
    # Export Metrics for Dashboard Evaluation
    metrics_df = pd.DataFrame({'y_true': y_test, 'y_prob': y_pred_proba})
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Exported raw test metrics to {metrics_path}")
    
    # Export Model for Dashboard SHAP & Live Scoring
    joblib.dump(model, model_path)
    print(f"Saved fitted model to {model_path}")
    
    # Score the userbase
    print("Scoring all users for Churn Risk...")
    df['Churn_Probability'] = model.predict_proba(X)[:, 1]
    
    # Priority logic: Filter active users (Churn == 0), then sort by Probability -> Descending, then total_spent (CLV proxy) -> Descending
    active_users = df[df['Churn'] == 0].copy()
    active_users = active_users.sort_values(by=['Churn_Probability', 'total_spent'], ascending=[False, False])
    
    active_users.to_csv(output_path, index=False)
    print(f"Saved prioritized At-Risk users to {output_path}")

if __name__ == "__main__":
    run_churn_model()
