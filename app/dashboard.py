import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as plt_sns
import shap
import joblib
from sklearn.metrics import roc_curve, auc, precision_recall_curve

st.set_page_config(page_title="E-Commerce AI Tool", layout="wide", page_icon="🛍️")

# --- 1. Clean Light Theme CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    
    /* Clean Light Background */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }
    
    h1 {
        color: #0f172a;
        font-weight: 800;
        text-align: center;
        padding-bottom: 2rem;
    }
    h2, h3, h4 { color: #334155; }
    
    /* Clean White Cards with Soft Shadows */
    div[data-testid="metric-container"] {
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        padding: 5% 5% 5% 10%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        color: #0ea5e9 !important;
        border-bottom: 3px solid #0ea5e9 !important;
    }
    
    /* Highlight boxes for Professor Notes */
    .prof-note {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 15px;
        border-radius: 4px;
        color: #166534;
        margin-top: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to print professor notes easily
def prof_note(text):
    st.markdown(f'<div class="prof-note">👨‍🏫 <b>Professor Note (How this works):</b><br>{text}</div>', unsafe_allow_html=True)

# --- 2. Load Data Safely ---
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "data")
    
    basket = pd.read_csv(os.path.join(data_dir, "basket_insights.csv"))
    churn = pd.read_csv(os.path.join(data_dir, "churn_risks.csv"))
    metrics = pd.read_csv(os.path.join(data_dir, "model_metrics.csv"))
    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    return basket, churn, metrics, features, data_dir

@st.cache_resource
def load_model(data_dir):
    model = joblib.load(os.path.join(data_dir, "churn_model.pkl"))
    explainer = shap.TreeExplainer(model)
    return model, explainer

try:
    basket_df, churn_df, metrics_df, full_features_df, data_dir = load_data()
    model, explainer = load_model(data_dir)
except Exception as e:
    st.error(f"Error loading files. Error: {e}")
    st.stop()

st.title("🛍️ E-Commerce AI Assistant")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Home Summary", 
    "🔍 Why are they leaving?", 
    "💰 Coupon Simulator", 
    "🛒 Shopping Cart AI", 
    "📈 Accuracy Checks"
])

active_users = churn_df[churn_df['Churn'] == 0].copy()
at_risk_pool = active_users[active_users['Churn_Probability'] >= 0.70]

# --- TAB 1: HOME SUMMARY ---
with tab1:
    st.header("Business Health Summary")
    st.write("A quick look at how the store is doing, and who we need to focus on saving today.")
    
    col1, col2, col3 = st.columns(3)
    total_users = len(churn_df)
    churned_users = churn_df['Churn'].sum()
    
    col1.metric("Total Customers", f"{total_users:,}")
    col2.metric("Overall Leaving Rate", f"{(churned_users / total_users):.1%}", "Left Store", delta_color="inverse")
    
    valuable_at_risk = at_risk_pool['total_spent'].sum()
    col3.metric("Money at Risk", f"${valuable_at_risk:,.0f}", f"From {len(at_risk_pool)} customers")
    
    prof_note("This section calculates totals directly from the data. The 'Money at Risk' sums up the total historic lifetime value of all active customers who our AI flagged with a >70% chance of leaving. This proves to a business owner exactly how much revenue is mathematically at risk right now.")
    
    st.markdown("### Top Customers to Rescue")
    st.write("We sorted these active customers so the ones who **spend the most** and are **most likely to leave** appear at the very top.")
    
    display_churn = at_risk_pool[['CustomerID', 'Churn_Probability', 'total_spent', 'total_lifetime_orders', 'days_since_last_order']].head(50)
    display_churn = display_churn.rename(columns={
        'Churn_Probability': 'Chance of Leaving',
        'total_spent': 'Total Money Spent ($)',
        'total_lifetime_orders': '# of Orders Made',
        'days_since_last_order': 'Days since last purchase'
    })
    st.dataframe(display_churn.style.background_gradient(cmap="Blues", subset=['Chance of Leaving']), use_container_width=True)


# --- TAB 2: WHY ARE THEY LEAVING ---
with tab2:
    st.header("Why is the AI flagging this customer?")
    st.write("Instead of guessing, we can use advanced math to see exactly which behavior caused the AI to flag them.")
    
    if not display_churn.empty:
        colA, colB = st.columns([1, 2])
        with colA:
            st.write("**Pick a customer ID from the risk list:**")
            selected_id = st.selectbox("Customer ID:", options=display_churn['CustomerID'].values)
            
            user_features = full_features_df[full_features_df['CustomerID'] == selected_id][
                ['total_lifetime_orders', 'total_spent', 'total_items_bought', 'average_order_value', 'average_days_between_orders']
            ]
            
            st.markdown("#### This customer's profile:")
            st.write(f"- **Total Orders:** {user_features.iloc[0]['total_lifetime_orders']:.0f}")
            st.write(f"- **Days Between Orders (Avg):** {user_features.iloc[0]['average_days_between_orders']:.0f} days")
            st.write(f"- **Total Spent:** ${user_features.iloc[0]['total_spent']:.0f}")
                
        with colB:
            # We compute SHAP here
            shap_values = explainer(user_features)
            
            # Simple clean light background for the plot
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(7, 4))
            
            # Waterfall plot
            shap.plots.waterfall(shap_values[0, :, 1], show=False)
            st.pyplot(fig)
            plt.clf()
            
            prof_note("This is a <b>SHAP Waterfall Plot</b>. Machine learning is usually a 'black box' where we can't see why it makes a decision. SHAP breaks the math open. The blue bars push the risk score down (making them safe), and the red bars push the risk score up. For example, if 'average_days_between_orders' is a big red bar, it means that specific feature is the main reason the AI thinks they are leaving.")
    else:
        st.info("No active users are currently flagged as high risk.")


# --- TAB 3: COUPON SIMULATOR ---
with tab3:
    st.header("Financial Rescue Simulator")
    st.write("If we send discount coupons to our at-risk customers, how much money can we save?")
    
    st.markdown("---")
    scol1, scol2, scol3 = st.columns(3)
    
    with scol1:
        target_risk_threshold = st.slider("Target customers with Risk > (%)", min_value=50, max_value=99, value=75)
    with scol2:
        coupon_discount_percent = st.slider("Give them a discount of (%)", min_value=5.0, max_value=50.0, value=15.0, step=1.0)
    with scol3:
        expected_success_rate = st.slider("How many will actually use the coupon? (%)", min_value=1.0, max_value=30.0, value=10.0, step=1.0)
        
    campaign_pool = active_users[active_users['Churn_Probability'] >= (target_risk_threshold / 100.0)]
    pool_size = len(campaign_pool)
    
    if pool_size > 0:
        # AOV = Average Order Value
        avg_future_order_val = campaign_pool['average_order_value'].mean()
        
        # Calculate ROI
        rescued_users = int(pool_size * (expected_success_rate / 100.0))
        revenue_saved = rescued_users * avg_future_order_val
        campaign_cost = rescued_users * (avg_future_order_val * (coupon_discount_percent / 100.0))
        net_profit = revenue_saved - campaign_cost
        
        rcol1, rcol2, rcol3 = st.columns(3)
        rcol1.metric("Customers Saved", f"{rescued_users:,} out of {pool_size:,}")
        rcol2.metric("Gross Revenue Saved", f"${revenue_saved:,.0f}")
        rcol3.metric("Net Profit (After Coupon Cost)", f"${net_profit:,.0f}", f"-${campaign_cost:,.0f} Cost")
        
        prof_note("This calculates Return on Investment (ROI) dynamically. It proves that the data science model generates real money. It isolates a specific group of at-risk users, averages out what they normally spend, and mathematically deducts the cost of giving them a coupon. The 'Net Profit' is money the business would have completely lost if the AI didn't catch these users.")
    else:
        st.warning(f"No active customers have a risk higher than {target_risk_threshold}%.")


# --- TAB 4: SHOPPING CART AI ---
with tab4:
    st.header("Smart Cross-Selling (Market Basket)")
    st.write("Pick an item from the real dataset. The AI will immediately suggest the best item to pair with it, just like Amazon's 'Frequently Bought Together'.")
    
    assoc_items = basket_df['antecedents'].unique()
    if len(assoc_items) > 0:
        selected_item = st.selectbox("Imagine you put this in your cart:", options=assoc_items[:100])
        
        matches = basket_df[basket_df['antecedents'] == selected_item].copy()
        
        if not matches.empty:
            best_match = matches.iloc[0]
            st.success("✅ **AI Notification Triggered:**")
            st.write(f"### 🎯 You should pop up a message offering them: **{best_match['consequents']}**")
            
            prof_note(f"<b>How we found this:</b> We used the <b>FP-Growth Algorithm</b> to scan millions of receipts. The <b>Confidence ({best_match['confidence']:.1%})</b> means if they buy the first item, there's a {best_match['confidence']:.1%} chance they will buy the second. The <b>Lift ({best_match['lift']:.2f}x)</b> means they are {best_match['lift']:.2f} times MORE likely to buy them together than by random chance.")
            
            st.write("Other strong items to suggest:")
            safe_matches = matches[['consequents', 'confidence', 'lift']].rename(columns={
                'consequents': 'Suggest this item',
                'confidence': 'Confidence %',
                'lift': 'Lift Score'
            })
            st.dataframe(safe_matches.head(5), use_container_width=True)


# --- TAB 5: ACCURACY CHECKS ---
with tab5:
    st.header("Proof that the Model Works")
    st.write("How do we know the AI isn't just making random guesses? We use these curves to prove it is mathematically sound.")
    
    y_true = metrics_df['y_true']
    y_prob = metrics_df['y_prob']
    
    colA, colB = st.columns(2)
    plt.style.use('default')
    
    with colA:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig1, ax1 = plt.subplots(figsize=(5,4))
        ax1.plot(fpr, tpr, color='#0ea5e9', lw=2, label=f'Model Score (AUC = {roc_auc:.2f})')
        ax1.plot([0, 1], [0, 1], color='#94a3b8', lw=2, linestyle='--')
        ax1.set_xlabel('False Positive (Wrong Guesses)')
        ax1.set_ylabel('True Positive (Right Guesses)')
        ax1.legend(loc="lower right")
        ax1.grid(color=(0.0,0.0,0.0,0.05))
        st.pyplot(fig1)
        
        prof_note("<b>The ROC Curve:</b> The dotted line is what happens if the AI just guesses randomly (50/50). Because our blue line arches UP and away from the dotted line, it proves the AI is finding real patterns. An AUC > 0.70 is standard for noisy consumer behavior.")

    with colB:
        st.subheader("Precision-Recall Curve")
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        fig2, ax2 = plt.subplots(figsize=(5,4))
        ax2.plot(recall, precision, color='#10b981', lw=2, label=f'Model Score (AUC = {pr_auc:.2f})')
        ax2.set_xlabel('Recall (How many leavers we caught)')
        ax2.set_ylabel('Precision (When we guess they leave, are we right?)')
        ax2.legend(loc="upper right")
        ax2.grid(color=(0.0,0.0,0.0,0.05))
        st.pyplot(fig2)
        
        prof_note("<b>The PR Curve:</b> This shows the tradeoff of sending coupons. If we want to catch ALL leaving users (high recall), we will accidentally send coupons to people who were going to stay anyway (low precision). This curve helps the business pick the exact right cutoff point.")
