import os
import pandas as pd
import numpy as np
import shap
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.metrics import roc_curve, auc, precision_recall_curve

app = FastAPI()

# --- Load Data Safely ---
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "..", "data")
static_dir = os.path.join(current_dir, "static")

# Mount Static Files (HTML, JS, CSS)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

try:
    basket_df = pd.read_csv(os.path.join(data_dir, "basket_insights.csv"))
    churn_df = pd.read_csv(os.path.join(data_dir, "churn_risks.csv"))
    metrics_df = pd.read_csv(os.path.join(data_dir, "model_metrics.csv"))
    full_features_df = pd.read_csv(os.path.join(data_dir, "features.csv"))
    
    model = joblib.load(os.path.join(data_dir, "churn_model.pkl"))
    explainer = shap.TreeExplainer(model)
except Exception as e:
    print(f"Error loading base files: {e}")

try:
    causal_model = joblib.load(os.path.join(data_dir, "causal_model.pkl"))
except Exception as e:
    print(f"Error loading causal model: {e}")
    causal_model = None

try:
    import json
    with open(os.path.join(data_dir, "lstm_eval_samples.json"), "r") as f:
        lstm_data = json.load(f)
except Exception as e:
    lstm_data = {"datasets": []}

active_users = churn_df[churn_df['Churn'] == 0].copy() if 'churn_df' in locals() else pd.DataFrame()
at_risk_pool = active_users[active_users['Churn_Probability'] >= 0.70] if not active_users.empty else pd.DataFrame()

# --- Default Route ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

# --- API Routes ---

@app.get("/api/summary")
def get_summary():
    total_users = len(churn_df)
    churned_users = int(churn_df['Churn'].sum())
    
    valuable_at_risk = float(at_risk_pool['total_spent'].sum())
    
    display_churn = at_risk_pool[['CustomerID', 'Churn_Probability', 'total_spent', 'total_lifetime_orders', 'days_since_last_order']].head(50)
    
    return {
        "total_users": total_users,
        "overall_leaving_rate": churned_users / total_users,
        "valuable_at_risk": valuable_at_risk,
        "at_risk_pool_size": len(at_risk_pool),
        "top_customers": display_churn.to_dict(orient="records")
    }

@app.get("/api/customer_features/{customer_id}")
def get_customer_features(customer_id: int):
    user_features = full_features_df[full_features_df['CustomerID'] == customer_id]
    if user_features.empty:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    data = user_features[['total_lifetime_orders', 'total_spent', 'total_items_bought', 'average_order_value', 'average_days_between_orders']].iloc[0].to_dict()
    return data

@app.get("/api/shap/{customer_id}")
def get_shap(customer_id: int):
    user_features = full_features_df[full_features_df['CustomerID'] == customer_id]
    if user_features.empty:
        raise HTTPException(status_code=404, detail="Customer not found")
        
    features = user_features.drop('CustomerID', axis=1, errors='ignore')
    
    # SHAP gives an object, we need to extract the values for plotting in JS.
    shap_values = explainer(features)
    
    # shap_values[0, :, 1] is for class 1 (churn). 
    # Extract feature names and values.
    sv = shap_values.values[0, :, 1] if len(shap_values.values.shape) == 3 else shap_values.values[0]
    bx = shap_values.base_values[0, 1] if isinstance(shap_values.base_values[0], (list, np.ndarray)) else shap_values.base_values[0]
    
    feature_names = features.columns.tolist()
    feature_vals = features.iloc[0].tolist()
    
    return {
        "base_value": float(bx),
        "feature_names": feature_names,
        "feature_values": feature_vals,
        "shap_values": sv.tolist()
    }

@app.get("/api/lstm_risk")
def get_lstm_risk():
    if lstm_data and lstm_data.get("datasets"):
        return lstm_data
    return {"datasets": []}

@app.get("/api/causal_nudge")
def get_causal_nudge(discount: float = 15.0):
    if causal_model is None or 'full_features_df' not in globals() or full_features_df.empty:
        return {"rescued": 0, "pool_size": 0, "revenue_saved": 0, "cost": 0, "net_profit": 0, "uplift_mean": 0}
        
    df_nudge = full_features_df.copy()
    
    # 1. Base prediction (Treatment = 0)
    df_nudge['Treatment'] = 0
    features = ['total_lifetime_orders', 'total_spent', 'total_items_bought', 'average_order_value', 'average_days_between_orders', 'Treatment']
    base_pred = causal_model.predict(df_nudge[features])
    
    # 2. Intervention prediction (Treatment = 1 with scaling)
    # The higher the discount, the more "treatment" effect is applied (rough proxy scaling for the simulator)
    scaling = discount / 15.0 
    df_nudge['Treatment'] = 1
    intervention_pred = causal_model.predict(df_nudge[features]) * scaling
    
    # 3. Calculate Causal Uplift (CATE proxy)
    uplift = intervention_pred - base_pred
    
    # Filter only positive uplift targets (the Persuadables)
    persuadable_idx = uplift > 0
    pool_size = np.sum(persuadable_idx)
    
    if pool_size == 0:
        return {"rescued": 0, "pool_size": 0, "revenue_saved": 0, "cost": 0, "net_profit": 0, "uplift_mean": 0}
        
    avg_uplift = np.mean(uplift[persuadable_idx])
    rescued_users = int(pool_size)
    revenue_saved = float(np.sum(uplift[persuadable_idx]))
    
    # Calculate Cost
    avg_base_order_val = df_nudge.loc[persuadable_idx, 'average_order_value'].mean()
    campaign_cost = float(rescued_users * (avg_base_order_val * (discount / 100.0)))
    
    net_profit = float(revenue_saved - campaign_cost)
    
    return {
        "rescued": rescued_users,
        "pool_size": int(len(df_nudge)),
        "revenue_saved": revenue_saved,
        "cost": campaign_cost,
        "net_profit": net_profit,
        "uplift_mean": float(avg_uplift)
    }

@app.get("/api/market_basket/items")
def get_basket_items():
    assoc_items = basket_df['antecedents'].unique().tolist()
    return {"items": assoc_items[:100]}

@app.get("/api/market_basket/{item}")
def get_market_basket(item: str):
    matches = basket_df[basket_df['antecedents'] == item]
    if matches.empty:
        return {"matches": []}
        
    sorted_matches = matches.sort_values(by='confidence', ascending=False)
    results = sorted_matches[['consequents', 'confidence', 'lift']].to_dict(orient="records")
    return {"matches": results}

@app.get("/api/metrics")
def get_metrics():
    y_true = metrics_df['y_true']
    y_prob = metrics_df['y_prob']
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    return {
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(roc_auc)
        },
        "pr": {
            "recall": recall.tolist(),
            "precision": precision.tolist(),
            "auc": float(pr_auc)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
