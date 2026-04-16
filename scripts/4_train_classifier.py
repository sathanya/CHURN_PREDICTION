import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

def train_classification_layer(data_dir="data", model_dir="models"):
    print("Loading final archetypes data...")
    df = pd.read_csv(f"{data_dir}/final_archetypes.csv", index_col='customer_id')

    # Separate features and target
    drop_cols = ['Cluster', 'Archetype', 'umap_x', 'umap_y']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['Archetype']

    # Load scaler used for clustering, to ensure inputs are scaled similarly
    # If the clusters were created on scaled data, we train Random Forest on scaled data too 
    # (though Random Forests do not strictly require scaling, SHAP values are cleaner without it, 
    # but for simplicity we'll train on the original features).
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nTraining RandomForestClassifier...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Evaluate
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(rf, f"{model_dir}/rf_classifier.pkl")
    print(f"Model saved to {model_dir}/rf_classifier.pkl")

    print("\nFitting SHAP Explainer (Explainability & Regulatory Compliance)...")
    # Tree explainer is excellent for Random Forests
    explainer = shap.TreeExplainer(rf)
    joblib.dump(explainer, f"{model_dir}/shap_explainer.pkl")
    
    # Pre-calculate a small background dataset for Streamlit if needed, or just save the explainer
    # We will use the explainer on test dataset instances in the portal
    print("Pipeline Phase 5 and 6 Complete.")

if __name__ == "__main__":
    train_classification_layer()
