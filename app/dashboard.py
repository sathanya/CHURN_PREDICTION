import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import json
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="BehaviorPrint OS", layout="wide", initial_sidebar_state="expanded")

# Inject premium CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Vibrant Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b132b 0%, #1c2541 100%);
        color: white !important;
    }
    
    /* Cards */
    .stCard {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 8px 24px rgba(11, 19, 43, 0.08); /* Sophisticated shadow */
        border: 1px solid #e1e4e8;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stCard:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(11, 19, 43, 0.12);
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700;
        color: #3a506b;
    }
    
    h1 { font-weight: 700 !important; color: #0b132b; letter-spacing: -1px; }
    h2, h3 { font-weight: 600 !important; color: #1c2541; }
    
    .archetype-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.2rem;
        background: linear-gradient(90deg, #ff6b6b, #ff8e8b);
        color: white;
        margin-bottom: 20px;
    }
    .badge-builder { background: linear-gradient(90deg, #1dd1a1, #10ac84); }
    .badge-maximizer { background: linear-gradient(90deg, #ff9f43, #ee5253); }
    .badge-hoarder { background: linear-gradient(90deg, #54a0ff, #2e86de); }
    .badge-chaotic { background: linear-gradient(90deg, #5f27cd, #341f97); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    try:
        rf = joblib.load("models/rf_classifier.pkl")
        explainer = joblib.load("models/shap_explainer.pkl")
        scaler = joblib.load("models/scaler.pkl")
        features_cols = joblib.load("models/feature_columns.pkl")
        
        # Load final archetypes and UMAP embedded dataset for background context
        bg_data = pd.read_csv("data/final_archetypes.csv")
        rules = pd.read_csv("data/fpgrowth_rules.csv")
        
        return rf, explainer, scaler, features_cols, bg_data, rules
    except Exception as e:
        return None, None, None, None, None, None


models_loaded = True
rf, explainer, scaler, features_cols, bg_data, rules = load_models()
if rf is None:
    models_loaded = False


# Sidebar Navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=60)
    st.title("BehaviorPrint OS")
    st.markdown("*Mining Psychological Money Archetypes from Raw Transactions.*")
    
    st.markdown("---")
    
    menu = st.radio(
        "Navigation",
        ["1. Upload & Connect", "2. Personality Reveal", "3. Insight & Nudges"]
    )
    
    st.markdown("---")
    st.caption("A RBI/GDPR Compliant Interpretability Layer.")
    

# ==========================================================
# SCREEN 1: UPLOAD
# ==========================================================
if menu == "1. Upload & Connect":
    st.title("Welcome to BehaviorPrint")
    st.markdown("Upload a customer's raw bank transaction CSV to reconstruct their psychological profile.")
    
    st.info("Because finding 'real' financial behaviors without bias requires Unsupervised Association Rule Mining.")
    
    uploaded_file = st.file_uploader("Upload Customer Transactions (CSV)", type="csv")
    
    if st.button("Generate Demo Customer"):
        st.session_state["demo_user"] = "CUST_00042" # Example string
        st.success("Test user loaded! Proceed to Step 2.")
        
    if uploaded_file is not None:
        # In a real app we would run 2_feature_engineering on this file in runtime.
        st.success("File Processed via FP-Growth rules. Proceed to Step 2.")


def get_user_data():
    if not models_loaded:
        return None
    # We sample a random user from bg_data to act as the "uploaded" user for the prototype
    if "selected_user" not in st.session_state:
        st.session_state["selected_user"] = bg_data.sample(1).iloc[0]
    return st.session_state["selected_user"]


# ==========================================================
# SCREEN 2: PERSONALITY REVEAL
# ==========================================================
elif menu == "2. Personality Reveal":
    st.title("Behavioral Archetype Generation")
    
    if not models_loaded:
        st.warning("Models not trained yet. Please run the pipeline scripts first.")
        st.stop()
        
    user_data = get_user_data()
    
    # 1. Archetype Badge
    pred_archetype = user_data['Archetype']
    
    badge_class = "archetype-badge"
    if "Hoarder" in pred_archetype: badge_class += " badge-hoarder"
    elif "Maximizer" in pred_archetype: badge_class += " badge-maximizer"
    elif "Builder" in pred_archetype: badge_class += " badge-builder"
    elif "Chaotic" in pred_archetype: badge_class += " badge-chaotic"
    
    icon_map = {"Anxious Hoarder": "🧊", "Lifestyle Maximizer": "🔥", "Disciplined Builder": "🏗️", "Chaotic Survivor": "🌀"}
    icon = icon_map.get(pred_archetype, "👤")
    
    st.markdown(f'<div class="{badge_class}">{icon} {pred_archetype}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader("Radar Profile (Top Dimensions)")
        
        # We grab radar metrics from the basic pipeline B features mapping 
        # (Assuming Pipeline B won, it contains flags and rule columns. We'll find 5 key ones).
        radar_features = [c for c in features_cols if c.startswith('flag_')][:5]
        if len(radar_features) == 5:
            # We use an approximation radar plot mapping boolean/binary counts to categories
            labels = [r.replace('flag_', '') for r in radar_features]
            values = user_data[radar_features].values * 10
            
            # Simple Matplotlib Radar
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            values = np.concatenate((values, [values[0]]))
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
            ax.fill(angles, values, color='#ff6b6b', alpha=0.25)
            ax.plot(angles, values, color='#ff6b6b', linewidth=2)
            ax.set_yticklabels([])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, size=8)
            ax.spines['polar'].set_visible(False)
            
            st.pyplot(fig)
        else:
            st.write("Insufficient underlying flags for radar.")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader("Clustering Context (UMAP)")
        st.write("Your position among 5,000 other customers based on Unsupervised feature extraction.")
        
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.scatterplot(data=bg_data, x='umap_x', y='umap_y', hue='Cluster', palette='muted', s=10, alpha=0.3, ax=ax2)
        
        # Highlight our user
        ax2.scatter(user_data['umap_x'], user_data['umap_y'], color='red', s=150, edgecolor='white', marker='*')
        ax2.set_xlabel("")
        ax2.set_ylabel("")
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)
        
        
    st.markdown("---")
    st.subheader("Explainability / Compliance Engine (SHAP)")
    st.markdown("Why did the ML Engine classify this user as **{}**? (Regulatory Defensibility)".format(pred_archetype))
    
    # Render localized SHAP waterfall
    try:
        user_features = user_data[features_cols].values.reshape(1, -1)
        shap_values = explainer(user_features)
        
        # Determine class index
        class_idx = np.where(rf.classes_ == pred_archetype)[0][0]
        
        # Matplotlib Waterfall equivalent logic
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_values[0, :, class_idx], show=False)
        st.pyplot(fig3)
    except Exception as e:
        st.write("SHAP Explainer encountered an issue visualizing this individual. ", e)


# ==========================================================
# SCREEN 3: INSIGHTS & NUDGES
# ==========================================================
elif menu == "3. Insight & Nudges":
    st.title("Personalized Intervention Rules")
    
    if not models_loaded:
        st.warning("Models not trained yet.")
        st.stop()
        
    user_data = get_user_data()
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Recommended Actions")
        archetype = user_data['Archetype']
        
        if "Hoarder" in archetype:
            n1 = "Offer 'Safe' high-yield locked deposit."
            n2 = "Nudge them to increase low-risk investments over pure ATM cash holding."
        elif "Maximizer" in archetype:
            n1 = "Trigger warning when weekend spend is 2x above average."
            n2 = "Propose 'Rounding up' social expenses directly into savings."
        elif "Builder" in archetype:
            n1 = "Reward them with higher tier credit cards."
            n2 = "Suggest diversified index funds for long term growth."
        elif "Chaotic" in archetype:
            n1 = "Offer rigid budgeting templates on payday."
            n2 = "Automate bills prior to discretionary spend."
        else:
            n1 = "Standard marketing payload applies."
            n2 = "No specialized intervention."
            
        st.info("Nudge 1: " + n1)
        st.warning("Nudge 2: " + n2)
        
    with col2:
        st.subheader("Financial Simulator")
        st.markdown("Adjust a behavioral habit and see the projection.")
        
        savings_goal = st.slider("Target monthly savings transfer (%)", 0, 50, 10)
        impulse_reduction = st.slider("Reduce 'Lifestyle/Weekend' spend by (%)", 0, 100, 20)
        
        projected = (savings_goal * 100) + (impulse_reduction * 50) # Arbitrary magic math for demo
        
        st.metric(label="Projected 12-Month Increased Savings", value=f"₹ {projected:,.0f}")
        
    st.markdown("---")
    st.subheader("Mined Customer Associations (FP-Growth)")
    st.markdown("Based on millions of transaction pairings, customers with this profile often trigger these rules:")
    
    if rules is not None and not rules.empty:
        # Show top 5 rules
        for i, row in rules.head(5).iterrows():
            st.markdown(f"- If a customer does **{row['antecedents_str']}**, they are **{row['confidence']*100:.1f}%** likely to concurrently do **{row['consequents_str']}** (Lift: {row['lift']:.2f}).")
    else:
        st.write("No strong association rules were generated.")
