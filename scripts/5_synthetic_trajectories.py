import pandas as pd
import numpy as np
import os
import joblib

def create_synthetic_trajectories(input_path="data/features.csv", output_dir="data", seq_length=6):
    print("Loading base features for trajectory simulation...")
    df = pd.read_csv(input_path)
    
    # We will simulate `seq_length` months of data for each user.
    # Features per time step: [monthly_spent, monthly_orders, days_since_purchase]
    num_features = 3
    num_users = len(df)
    
    # Pre-allocate array: (Users, TimeSteps, Features)
    X_seq = np.zeros((num_users, seq_length, num_features))
    y_seq = df['Churn'].values
    customer_ids = []
    
    print(f"Simulating {seq_length}-month trajectories for {num_users} users...")
    
    for i, row in df.iterrows():
        churned = row['Churn'] == 1
        base_monthly_spent = row['total_spent'] / max(1, (row['total_lifetime_orders'])) * 2  # Loose proxy for avg monthly
        base_monthly_orders = max(1, row['total_lifetime_orders'] / 12)
        
        # We add some Gaussian noise to simulate real fluctuations
        for t in range(seq_length):
            # Normal user behavior fluctuates around the mean
            noise_spent = np.random.normal(0, base_monthly_spent * 0.2)
            noise_orders = np.random.normal(0, base_monthly_orders * 0.2)
            
            m_spent = max(0, base_monthly_spent + noise_spent)
            m_orders = max(0, int(base_monthly_orders + noise_orders))
            
            # If churned, we simulate a 'drop-off' trajectory in the last 3 months
            if churned and t >= (seq_length - 3):
                decay = np.power(0.5, (t - (seq_length - 4)))
                m_spent *= decay
                m_orders = max(0, int(m_orders * decay))
                
            # Simulate "Days since last purchase for this month"
            if m_orders == 0:
                days_since = 30 + (t * 5) # Increasing gap if no orders
            else:
                days_since = np.random.randint(2, 15)
                
            X_seq[i, t, 0] = m_spent
            X_seq[i, t, 1] = m_orders
            X_seq[i, t, 2] = days_since
            
        # Optional: track customer ID if it existed in the dataframe
        if 'CustomerID' in df.columns:
            customer_ids.append(row['CustomerID'])
        else:
            customer_ids.append(f"C_{i}")
            
    # Normalize the 3D tensor
    print("Normalizing sequential features...")
    # Global normalization per feature axis
    for f in range(num_features):
        max_val = np.max(X_seq[:, :, f])
        if max_val > 0:
            X_seq[:, :, f] = X_seq[:, :, f] / max_val
            
    # Save the arrays
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/X_seq.npy", X_seq)
    np.save(f"{output_dir}/y_seq.npy", y_seq)
    joblib.dump(customer_ids, f"{output_dir}/seq_customers.pkl")
    
    print(f"Saved sequential dataset with shape {X_seq.shape} to {output_dir}")

if __name__ == "__main__":
    create_synthetic_trajectories()
