import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

def train_sequence_model(seq_dir="data", output_path="data/sequence_nn_model.pkl"):
    print("Loading sequential data...")
    X_seq = np.load(f"{seq_dir}/X_seq.npy")
    y_seq = np.load(f"{seq_dir}/y_seq.npy")
    
    print(f"Loaded Trajectory Tensor shape: {X_seq.shape}")
    
    # Flatten the sequence for MLP: (Users, TimeSteps * Features)
    num_users = X_seq.shape[0]
    X_flat = X_seq.reshape(num_users, -1)
    
    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y_seq, test_size=0.2, random_state=42, stratify=y_seq)
    
    print("Building Multi-Layer Perceptron (Neural Network) for Trajectories...")
    # This acts as a dense neural network simulating our sequence evaluator
    model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42, early_stopping=True)
    
    print("Training Neural Network for Time-Series Churn...")
    model.fit(X_train, y_train)
    
    print("\n--- Sequence NN Evaluation ---")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    print(f"Neural Network Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(f"Neural Network ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
    print(classification_report(y_test, y_pred))
    
    # Save the model via joblib
    joblib.dump(model, output_path)
    print(f"Saved Sequence NN model to {output_path}")

    # Export a few sample trajectories for the frontend SHAP-ish view
    sample_risk_idx = np.argsort(y_pred_proba)[-5:] # top 5 riskiest in test
    sample_trajectories = X_seq[X_test_idx] if 'X_test_idx' in locals() else X_seq[len(X_train) + sample_risk_idx]
    
    # Since we can't easily grab the original 3D test set without mapping indices, we just resample:
    X_test_3D = X_seq[len(X_train):] # rough splitting
    # Wait, train_test_split randomized it. Let's just predict on the whole set to get top 5:
    all_probas = model.predict_proba(X_flat)[:, 1]
    top_5_global = np.argsort(all_probas)[-5:]
    
    import json
    traj_data = {
        "months": ["M-5", "M-4", "M-3", "M-2", "M-1", "Current Month"],
        "datasets": []
    }
    
    for i, idx in enumerate(top_5_global):
        traj_data["datasets"].append({
            "id": f"Risk_Sample_{i+1}",
            "risk_score": float(all_probas[idx]),
            "spend_trend": [float(x) for x in X_seq[idx, :, 0]]
        })
        
    with open(f"{seq_dir}/lstm_eval_samples.json", "w") as f:
        json.dump(traj_data, f)
        
if __name__ == "__main__":
    train_sequence_model()
