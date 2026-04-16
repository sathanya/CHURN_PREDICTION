import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(num_customers=5000, days=180, output_dir="data"):
    """
    Generates synthetic banking transactions incorporating hidden psychological money archetypes
    so that clustering algorithms can identify meaningful real-world financial behaviors.
    """
    print(f"Generating data for {num_customers} customers over {days} days...")
    os.makedirs(output_dir, exist_ok=True)
    
    start_date = datetime.now() - timedelta(days=days)
    records = []

    # Hidden archetypes that our clustering will discover
    ARCHETYPES = ['Anxious Hoarder', 'Lifestyle Maximizer', 'Disciplined Builder', 'Chaotic Survivor']
    
    # Merchant mapping to psychological buckets
    MERCHANTS = {
        'survival': ['Groceries', 'Utilities', 'Rent', 'Pharmacy'],
        'social': ['Dining', 'Entertainment', 'Pubs/Bars', 'Coffee Shops'],
        'anxiety': ['ATM Small', 'ATM Medium'],
        'aspirational': ['Travel', 'Luxury', 'Electronics', 'Spa'],
        'income': ['Salary', 'Freelance'],
        'savings': ['Transfer to Savings', 'Investment']
    }

    # Helper function to get random date
    def random_date(start, end):
        return start + timedelta(seconds=np.random.randint(0, int((end - start).total_seconds())))

    for i in range(num_customers):
        if i % 500 == 0 and i > 0:
            print(f"Processed {i} customers...")
            
        cust_id = f"CUST_{str(i).zfill(5)}"
        archetype = np.random.choice(ARCHETYPES, p=[0.25, 0.3, 0.25, 0.2])
        
        balance = np.random.uniform(1000, 5000) # Initial balance
        salary = np.random.uniform(3000, 8000)
        
        # Monthly loop
        for month in range(1, (days // 30) + 1):
            month_start = start_date + timedelta(days=(month-1)*30)
            
            # --- Income Event ---
            salary_date = month_start + timedelta(days=np.random.randint(0, 3))
            balance += salary
            records.append([cust_id, salary_date, salary, 'Income', 'Branch', balance])
            
            # --- Archetype Behaviors ---
            if archetype == 'Anxious Hoarder':
                # Hoards cash, lots of small ATM trips, minimal social/aspirational
                num_atm = np.random.randint(5, 15)
                for _ in range(num_atm):
                    date = random_date(month_start, month_start + timedelta(days=28))
                    amt = np.random.uniform(50, 200)
                    balance -= amt
                    records.append([cust_id, date, -amt, 'ATM Small', 'ATM', balance])
                
                # Basic survival
                for _ in range(np.random.randint(10, 15)):
                    date = random_date(month_start, month_start + timedelta(days=28))
                    amt = np.random.uniform(20, 150)
                    balance -= amt
                    merch = np.random.choice(MERCHANTS['survival'])
                    records.append([cust_id, date, -amt, merch, 'Online', balance])
                    
            elif archetype == 'Lifestyle Maximizer':
                # High social/aspirational, weekend splurges
                # High weekend ratio
                for _ in range(np.random.randint(15, 30)):
                    # Skew towards weekends
                    day_offset = np.random.choice([0,1,2,3,4,5,6], p=[0.05, 0.05, 0.05, 0.05, 0.2, 0.3, 0.3])
                    week_offset = np.random.randint(0, 4) * 7
                    date = month_start + timedelta(days=day_offset + week_offset, hours=np.random.randint(18, 23))
                    
                    amt = np.random.uniform(50, 500)
                    balance -= amt
                    cat = np.random.choice(['social', 'aspirational'], p=[0.7, 0.3])
                    merch = np.random.choice(MERCHANTS[cat])
                    records.append([cust_id, date, -amt, merch, 'Online', balance])
                    
                # Basic survival
                for _ in range(np.random.randint(5, 10)):
                    date = random_date(month_start, month_start + timedelta(days=28))
                    amt = np.random.uniform(50, 200)
                    balance -= amt
                    merch = np.random.choice(MERCHANTS['survival'])
                    records.append([cust_id, date, -amt, merch, 'Online', balance])

            elif archetype == 'Disciplined Builder':
                # Quick transfer to savings after salary
                savings_date = salary_date + timedelta(days=np.random.randint(1, 3))
                savings_amt = salary * np.random.uniform(0.2, 0.4)
                balance -= savings_amt
                records.append([cust_id, savings_date, -savings_amt, 'Transfer to Savings', 'Online', balance])
                
                # Controlled survival/social
                for _ in range(np.random.randint(15, 25)):
                    date = random_date(month_start, month_start + timedelta(days=28))
                    amt = np.random.uniform(30, 250)
                    balance -= amt
                    cat = np.random.choice(['survival', 'social', 'aspirational'], p=[0.7, 0.2, 0.1])
                    merch = np.random.choice(MERCHANTS[cat])
                    records.append([cust_id, date, -amt, merch, 'Online', balance])

            elif archetype == 'Chaotic Survivor':
                # Erratic patterns, high drift, salary fragmented out quickly
                num_transfers = np.random.randint(3, 8)
                for i in range(num_transfers):
                    t_date = salary_date + timedelta(hours=np.random.randint(1, 48))
                    amt = (salary * np.random.uniform(0.4, 0.8)) / num_transfers
                    balance -= amt
                    records.append([cust_id, t_date, -amt, 'Transfer Out', 'Online', balance])
                    
                # High frequency, low amount erratic spending
                for _ in range(np.random.randint(20, 40)):
                    date = random_date(month_start, month_start + timedelta(days=28))
                    amt = np.random.uniform(10, 80)
                    balance -= amt
                    cat = np.random.choice(['survival', 'social'], p=[0.6, 0.4])
                    merch = np.random.choice(MERCHANTS[cat])
                    records.append([cust_id, date, -amt, merch, 'Online', balance])

    df = pd.DataFrame(records, columns=['customer_id', 'transaction_date', 'amount', 'merchant_category', 'channel', 'balance'])
    
    # Sort and slightly perturb balances to represent real-life drift
    df = df.sort_values(by=['customer_id', 'transaction_date'])
    
    print(f"Generated {len(df)} transactions.")
    
    file_path = os.path.join(output_dir, "synthetic_transactions.csv")
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    # Ensure reproducibility
    np.random.seed(42)
    # 5000 customers over 6 months generates about 700k-1M rows. 
    # Let's use 3000 to keep generation and FP-Growth fast enough for local runs while still proving the hypothesis.
    generate_synthetic_data(num_customers=3000, days=180)
