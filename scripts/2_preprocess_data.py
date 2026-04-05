import os
import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    print("Loading raw data... (this might take a minute for a ~23MB Excel file)")
    df = pd.read_excel(file_path)
    print(f"Raw dataset shape: {df.shape}")
    
    # Clean data: drop rows without a CustomerID
    df = df.dropna(subset=['CustomerID'])
    
    # Remove cancelled orders (InvoiceNo starts with 'C')
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df = df[~df['InvoiceNo'].str.startswith('C')]
    
    # Focus only on positive quantities and positive prices
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    # Create the total amount spent per item line
    df['Total_Price'] = df['Quantity'] * df['UnitPrice']
    print(f"Clean numeric dataset shape: {df.shape}")
    return df

def create_ml_features(df, output_path="data/features.csv"):
    print("Engineering ML features for Churn Modeling...")
    
    # Calculate the max date in the dataset to define "Today"
    analysis_date = df['InvoiceDate'].max()
    
    # Aggregate data per Customer
    customer_df = df.groupby('CustomerID').agg(
        total_lifetime_orders=('InvoiceNo', 'nunique'),
        first_order_date=('InvoiceDate', 'min'),
        last_order_date=('InvoiceDate', 'max'),
        total_spent=('Total_Price', 'sum'),
        total_items_bought=('Quantity', 'sum')
    ).reset_index()
    
    # Calculate Average Order Value
    customer_df['average_order_value'] = customer_df['total_spent'] / customer_df['total_lifetime_orders']
    
    # Calculate Days Since Last Order (Recency)
    customer_df['days_since_last_order'] = (analysis_date - customer_df['last_order_date']).dt.days
    
    # Calculate Average Days Between Orders
    # (Max date - Min date) / (Number of orders - 1). Fill with 0 if only 1 order.
    customer_df['lifetime_days'] = (customer_df['last_order_date'] - customer_df['first_order_date']).dt.days
    customer_df['average_days_between_orders'] = customer_df['lifetime_days'] / (customer_df['total_lifetime_orders'] - 1).clip(lower=1)
    # Fill cases with only 1 order
    customer_df.loc[customer_df['total_lifetime_orders'] == 1, 'average_days_between_orders'] = 0
    
    # Define Churn (e.g., hasn't ordered in the last 90 days)
    customer_df['Churn'] = (customer_df['days_since_last_order'] > 90).astype(int)
    
    # Drop timestamp columns before saving to purely numeric ML features
    customer_df = customer_df.drop(columns=['first_order_date', 'last_order_date', 'lifetime_days'])
    
    print(f"Engineered {len(customer_df)} customers with features.")
    print(f"Churn rate: {customer_df['Churn'].mean():.2%}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    customer_df.to_csv(output_path, index=False)
    print(f"Saved Machine Learning features to {output_path}")

def create_basket_matrix(df, output_path="data/basket_matrix.csv"):
    print("Preparing One-Hot Encoded Market Basket Matrix...")
    
    # To avoid memory crashing on 500k rows, let's filter to the United Kingdom only 
    # (which has the vast majority of transactions) and clean the descriptions
    basket_df = df[df['Country'] == 'United Kingdom'].copy()
    basket_df['Description'] = basket_df['Description'].str.strip()
    basket_df = basket_df.dropna(subset=['Description'])
    
    # Filter out rare items to constrain the matrix size. Keep Top 500 selling items.
    top_items = basket_df['Description'].value_counts().head(500).index
    basket_df = basket_df[basket_df['Description'].isin(top_items)]
    
    print(f"Building matrix for top {len(top_items)} items over {basket_df['InvoiceNo'].nunique()} invoices...")
    
    # Create the matrix (InvoiceNo vs Description, summing Quantities)
    basket = (basket_df.groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))
    
    # The mlxtend FP-Growth algorithm requires binary values (1 if bought, 0 if not)
    # Convert anything > 0 to 1, and everything else to 0
    def encode_units(x):
        return 1 if x >= 1 else 0

    basket_sets = basket.applymap(encode_units)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    basket_sets.to_csv(output_path)
    print(f"Saved Market Basket Matrix to {output_path} with shape {basket_sets.shape}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    raw_data_path = "data/raw/online_retail.xlsx"
    if not os.path.exists(raw_data_path):
        print("Raw data not found! Run 1_download_data.py first.")
        exit(1)
        
    df = load_and_clean_data(raw_data_path)
    create_ml_features(df)
    create_basket_matrix(df)
