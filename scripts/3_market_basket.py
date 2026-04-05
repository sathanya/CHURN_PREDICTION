import os
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

def run_market_basket_analysis(input_path="data/basket_matrix.csv", output_path="data/basket_insights.csv"):
    print(f"Loading basket matrix from {input_path}...")
    
    # Load the matrix, setting InvoiceNo as the index
    basket_sets = pd.read_csv(input_path, index_col=0)
    print(f"Matrix loaded with shape: {basket_sets.shape}")
    
    # Verify the values are binary integer (0 or 1)
    basket_sets = basket_sets.astype(bool)
    
    print("Running FP-Growth algorithm to find frequent item combinations...")
    # min_support = 0.01 means the item combination must appear in at least 1% of transactions
    frequent_itemsets = fpgrowth(basket_sets, min_support=0.015, use_colnames=True)
    
    print(f"Found {len(frequent_itemsets)} frequent itemsets. Generating rules...")
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    # Filter for rules with high confidence and lift
    rules = rules[(rules['lift'] >= 2) & (rules['confidence'] >= 0.3)]
    
    # Sort strongest rules by Lift
    rules = rules.sort_values('lift', ascending=False)
    
    # Format the frozensets to simple strings for CSV export
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    # Export top 100 strongest rules
    top_rules = rules.head(100)
    top_rules.to_csv(output_path, index=False)
    
    print(f"Extracted {len(top_rules)} top rules and saved to {output_path}")

if __name__ == "__main__":
    run_market_basket_analysis()
