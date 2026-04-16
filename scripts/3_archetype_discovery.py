import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap.umap_ as umap
import joblib

def discover_archetypes(data_dir="data", model_dir="models"):
    print("Loading features...")
    pipeline_a = pd.read_csv(f"{data_dir}/pipeline_a_features.csv").set_index('customer_id')
    pipeline_b = pd.read_csv(f"{data_dir}/pipeline_b_features.csv").set_index('customer_id')

    os.makedirs(model_dir, exist_ok=True)

    # Standardize
    scaler_A = StandardScaler()
    scaled_A = scaler_A.fit_transform(pipeline_a)
    
    scaler_B = StandardScaler()
    scaled_B = scaler_B.fit_transform(pipeline_b)

    # Dictionary to store results
    results_A = {}
    results_B = {}

    print("Evaluating K-Means (K=3 to 7)...")
    for k in range(3, 8):
        kmeans_a = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels_a = kmeans_a.fit_predict(scaled_A)
        score_a = silhouette_score(scaled_A, labels_a)
        results_A[k] = {'score': score_a, 'labels': labels_a, 'model': kmeans_a}

        kmeans_b = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels_b = kmeans_b.fit_predict(scaled_B)
        # Avoid issue if B has no variation or only 1 cluster
        if len(set(labels_b)) > 1:
            score_b = silhouette_score(scaled_B, labels_b)
        else:
            score_b = -1
        results_B[k] = {'score': score_b, 'labels': labels_b, 'model': kmeans_b}

    # Find best K for each pipeline
    best_k_A = max(results_A, key=lambda k: results_A[k]['score'])
    best_k_B = max(results_B, key=lambda k: results_B[k]['score'])

    print("-" * 50)
    print("HYPOTHESIS TEST RESULTS")
    print("-" * 50)
    print(f"Pipeline A (Manual) best K: {best_k_A} (Silhouette: {results_A[best_k_A]['score']:.4f})")
    print(f"Pipeline B (Mined)  best K: {best_k_B} (Silhouette: {results_B[best_k_B]['score']:.4f})")
    
    # Determine winner
    if results_B[best_k_B]['score'] > results_A[best_k_A]['score']:
        print("\n=> Hypothesis H1 ACCEPTED: Mined features outperformed manual features.")
        winning_pipeline = 'B'
        best_features = pipeline_b
        best_scaled = scaled_B
        best_labels = results_B[best_k_B]['labels']
        best_k = best_k_B
        scaler = scaler_B
        
        # Save feature columns for later
        joblib.dump(pipeline_b.columns.tolist(), f"{model_dir}/feature_columns.pkl")
    else:
        print("\n=> Null Hypothesis H0 holds: Manual features outperformed/equal to mined features.")
        winning_pipeline = 'A'
        best_features = pipeline_a
        best_scaled = scaled_A
        best_labels = results_A[best_k_A]['labels']
        best_k = best_k_A
        scaler = scaler_A
        
        # Save feature columns
        joblib.dump(pipeline_a.columns.tolist(), f"{model_dir}/feature_columns.pkl")

    joblib.dump(scaler, f"{model_dir}/scaler.pkl")

    print("\nRunning UMAP for 2D visualization on winning pipeline...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(best_scaled)

    # Save final dataset
    final_df = best_features.copy()
    final_df['Cluster'] = best_labels
    final_df['umap_x'] = embedding[:, 0]
    final_df['umap_y'] = embedding[:, 1]
    
    # Assign archetype names (we'll just use mapped names based on K)
    archetype_names = ["Anxious Hoarder", "Lifestyle Maximizer", "Disciplined Builder", "Chaotic Survivor", "Balanced User", "Unknown A", "Unknown B"]
    final_df['Archetype'] = final_df['Cluster'].apply(lambda x: archetype_names[x] if x < len(archetype_names) else f"Type {x}")
    
    final_df.to_csv(f"{data_dir}/final_archetypes.csv")
    print(f"Saved {len(final_df)} records to {data_dir}/final_archetypes.csv")

if __name__ == "__main__":
    discover_archetypes()
