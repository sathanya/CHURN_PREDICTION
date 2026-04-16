import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor

def train_causal_nudge(input_path="data/features.csv", output_path="data/causal_model.pkl"):
    print("Loading features for Causal Nudge modeling...")
    df = pd.read_csv(input_path)
    
    # We will synthetically augment the dataset with historical 'campaigns'
    # Treatment (T) = 1 if they received a discount, 0 if not
    np.random.seed(42)
    df['Treatment'] = np.random.binomial(1, 0.3, size=len(df))
    
    # Baseline spend
    base_spend = df['total_spent']
    
    # Simulate causal effect: A discount increases future spend (survival), but costs money.
    # The effect is heterogeneous: people with high average_days_between_orders are less responsive.
    uplift = 50 + (100 * (df['Treatment'])) - (2 * df['average_days_between_orders'])
    
    # Target variable: Future Spend
    df['Future_Spend'] = np.maximum(0, df['total_spent'] * 0.1 + uplift + np.random.normal(0, 20, size=len(df)))
    
    print("Training Single-Learner (S-Learner) Causal Model using Random Forest...")
    # Features for the model MUST include the Treatment variable
    features = [
        'total_lifetime_orders',
        'total_spent',
        'total_items_bought',
        'average_order_value',
        'average_days_between_orders',
        'Treatment'
    ]
    
    X = df[features]
    y = df['Future_Spend']
    
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, output_path)
    print(f"Saved Causal S-Learner to {output_path}")

if __name__ == "__main__":
    train_causal_nudge()
