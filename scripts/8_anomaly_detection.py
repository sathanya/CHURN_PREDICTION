import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest

def run_anomaly_detection(data_dir="data"):
    print("Initializing Security/Threat Detection Pipeline...")
    
    features_path = os.path.join(data_dir, "features.csv")
    if not os.path.exists(features_path):
        print("ERROR: features.csv not found.")
        return

    # 1. Load Data
    print("Loading entity feature matrix...")
    df = pd.read_csv(features_path)
    
    # 2. Select behavioral features for isolation
    print("Isolating behavioral vectors...")
    tracking_features = ['total_lifetime_orders', 'total_spent', 'total_items_bought', 'average_order_value', 'average_days_between_orders']
    X = df[tracking_features].fillna(0)

    # 3. Train Isolation Forest
    # contamination=0.01 means we assume 1% of the database is wildly anomalous
    print("Training Isolation Forest Anomaly Detector...")
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    
    # -1 for outliers, 1 for inliers
    df['Anomaly_Flag'] = iso_forest.fit_predict(X)
    df['Anomaly_Score'] = iso_forest.decision_function(X) # lower is more abnormal
    
    # 4. Filter Anomalies and Generate "Reasoning"
    anomalies = df[df['Anomaly_Flag'] == -1].copy()
    print(f"Detected {len(anomalies)} Critical Threat Anomalies out of {len(df)} nodes.")
    
    # Determine why they were flagged (simplistic heuristic explaining the outlier)
    def determine_reason(row):
        reasons = []
        if row['average_order_value'] > df['average_order_value'].quantile(0.99):
            reasons.append("Ultra-High AOV (Possible Corporate Bot/Fraud)")
        if row['total_lifetime_orders'] > df['total_lifetime_orders'].quantile(0.99):
            reasons.append("Extreme Order Frequency (Account Sharing/Scraping)")
        if row['average_days_between_orders'] < 1.0:
            reasons.append("Sub-24Hr Velocity (Bot Behavior)")
            
        if not reasons:
            reasons.append("Complex Multi-Dimensional Outlier")
            
        return " | ".join(reasons)

    anomalies['Flag_Reason'] = anomalies.apply(determine_reason, axis=1)
    
    # 5. Export
    output_path = os.path.join(data_dir, "anomalies.csv")
    export_df = anomalies[['CustomerID', 'total_spent', 'total_lifetime_orders', 'Anomaly_Score', 'Flag_Reason']].sort_values('Anomaly_Score')
    export_df.to_csv(output_path, index=False)
    print(f"Exported detailed threats to {output_path}")

if __name__ == "__main__":
    run_anomaly_detection()
