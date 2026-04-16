import os
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
import warnings
warnings.filterwarnings('ignore')

def process_pipeline(input_path="data/synthetic_transactions.csv", output_dir="data"):
    print("Loading data...")
    df = pd.read_csv(input_path, parse_dates=['transaction_date'])
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Extracting time features...")
    df['day_of_week'] = df['transaction_date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    df['month'] = df['transaction_date'].dt.month
    df['year_month'] = df['transaction_date'].dt.to_period('M')

    # =========================================================
    # PIPELINE A: MANUAL FORENSIC FEATURE ENGINEERING
    # =========================================================
    print("Running Pipeline A (Manual Feature Engineering)...")
    
    customers = pd.DataFrame({'customer_id': df['customer_id'].unique()})
    
    # 1. Impulse Ratio (weekend spend ÷ weekday spend)
    # We only sum negative amounts (outflows)
    outflows = df[df['amount'] < 0]
    weekend_spend = outflows[outflows['is_weekend']].groupby('customer_id')['amount'].sum().abs()
    weekday_spend = outflows[~outflows['is_weekend']].groupby('customer_id')['amount'].sum().abs()
    
    impulse = (weekend_spend / (weekday_spend + 1e-5)).rename('impulse_ratio')
    
    # 2. Anxiety Score (ATM withdrawals under 500)
    # Our data uses negative for outflow
    anxiety = outflows[(outflows['merchant_category'].str.startswith('ATM')) & (outflows['amount'].abs() < 500)].groupby('customer_id').size().rename('anxiety_score')
    
    # 3. Social Spend %
    social_merchants = ['Dining', 'Entertainment', 'Pubs/Bars', 'Coffee Shops']
    social_spend = outflows[outflows['merchant_category'].isin(social_merchants)].groupby('customer_id')['amount'].sum().abs()
    total_spend = outflows.groupby('customer_id')['amount'].sum().abs()
    social_pct = (social_spend / (total_spend + 1e-5)).rename('social_spend_pct')
    
    # 4. Fear Buffer (mean balance / monthly income)
    mean_balance = df.groupby('customer_id')['balance'].mean()
    monthly_income = df[df['merchant_category'] == 'Income'].groupby(['customer_id', 'year_month'])['amount'].sum().groupby('customer_id').mean()
    fear_buffer = (mean_balance / (monthly_income + 1e-5)).rename('fear_buffer')
    
    # 5. Drift Score (std dev of balance over time)
    drift_score = df.groupby('customer_id')['balance'].std().rename('drift_score')

    # 6. Gratification Index & 7. Salary Fragmentation
    # We'll calculate these via a groupby apply for simpler logic, or just a vectorized approximation
    # For speed, we'll approximate: 
    # Savings total / Income total
    savings_total = outflows[outflows['merchant_category'] == 'Transfer to Savings'].groupby('customer_id')['amount'].sum().abs()
    gratification = (savings_total / (monthly_income + 1e-5)).rename('gratification_index')
    
    # Salary fragmentation: number of 'Transfer Out'
    frag = outflows[outflows['merchant_category'] == 'Transfer Out'].groupby('customer_id').size().rename('salary_fragmentation')
    
    # Combine Pipeline A features
    pipeline_a = customers.join(impulse, on='customer_id') \
                          .join(anxiety, on='customer_id') \
                          .join(social_pct, on='customer_id') \
                          .join(fear_buffer, on='customer_id') \
                          .join(drift_score, on='customer_id') \
                          .join(gratification, on='customer_id') \
                          .join(frag, on='customer_id')
    
    pipeline_a.fillna(0, inplace=True)
    pipeline_a.to_csv(f"{output_dir}/pipeline_a_features.csv", index=False)
    print(f"Pipeline A complete: {pipeline_a.shape[1]-1} features.")

    # =========================================================
    # PIPELINE B: FP-GROWTH ASSOCIATION MINING
    # =========================================================
    print("Running Pipeline B (FP-Growth)...")
    
    # Discretize transactions into binary flags per customer
    # We'll create flags like 'High_ATM', 'High_Social', 'Has_Savings'
    
    # Group by customer and compute aggregates
    flags_df = pd.DataFrame({'customer_id': df['customer_id'].unique()}).set_index('customer_id')
    
    # High ATM usage
    atm_counts = outflows[outflows['merchant_category'].str.startswith('ATM')].groupby('customer_id').size()
    flags_df['High_ATM'] = atm_counts > atm_counts.median()
    
    # High Social Spend
    soc = outflows[outflows['merchant_category'].isin(social_merchants)].groupby('customer_id').size()
    flags_df['High_Social'] = soc > soc.median()
    
    # High Survival Spend
    surv_merchants = ['Groceries', 'Utilities', 'Rent', 'Pharmacy']
    surv = outflows[outflows['merchant_category'].isin(surv_merchants)].groupby('customer_id').size()
    flags_df['High_Survival'] = surv > surv.median()
    
    # Has Savings
    sav = outflows[outflows['merchant_category'] == 'Transfer to Savings'].groupby('customer_id').size()
    flags_df['Has_Savings'] = sav > 0
    
    # Has erratic transfers out
    erratic = outflows[outflows['merchant_category'] == 'Transfer Out'].groupby('customer_id').size()
    flags_df['Erratic_Transfers'] = erratic > 0
    
    # Weekend Splurger
    flags_df['Weekend_Splurger'] = pipeline_a.set_index('customer_id')['impulse_ratio'] > 1.5
    
    # High Balance
    flags_df['High_Balance'] = pipeline_a.set_index('customer_id')['fear_buffer'] > 2.0
    
    flags_df.fillna(False, inplace=True)
    
    # FP-GROWTH
    # We supply boolean matrix to fpgrowth
    print("   Running FP-Growth min_support=0.05...")
    frequent_itemsets = fpgrowth(flags_df, min_support=0.05, use_colnames=True)
    
    print("   Extracting rules min_confidence=0.6...")
    try:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)
        # Filter for strong rules
        rules = rules[rules['confidence'] >= 0.6]
        print(f"   Found {len(rules)} significant rules.")
        
        # Take top 15 rules by lift to use as features
        top_rules = rules.sort_values('lift', ascending=False).head(15)
        
        # Convert rules to features
        pipeline_b = pd.DataFrame({'customer_id': flags_df.index}).set_index('customer_id')
        
        # Add basic one-hot encoded flags + rule interactions
        for c in flags_df.columns:
            pipeline_b[f'flag_{c}'] = flags_df[c].astype(int)
            
        for i, row in top_rules.iterrows():
            antecedents = list(row['antecedents'])
            consequents = list(row['consequents'])
            rule_name = f"rule_{'_AND_'.join(antecedents)}_IMPLIES_{'_AND_'.join(consequents)}"
            # Rule is True if all antecedents and all consequents are True
            items = antecedents + consequents
            pipeline_b[rule_name] = flags_df[list(items)].all(axis=1).astype(int)
            
    except Exception as e:
        print(f"   Rule mining yielded insufficient patterns: {e}")
        pipeline_b = flags_df.astype(int)
        
    pipeline_b.reset_index().to_csv(f"{output_dir}/pipeline_b_features.csv", index=False)
    print(f"Pipeline B complete: {pipeline_b.shape[1]} features.")
    
    # Save rules for dashboard
    if 'rules' in locals() and len(rules) > 0:
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        rules.to_csv(f"{output_dir}/fpgrowth_rules.csv", index=False)

if __name__ == "__main__":
    process_pipeline()
