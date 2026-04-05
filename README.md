# 🛒 E-Commerce Intelligence & Retention Platform

An end-to-end Machine Learning ecosystem designed to bridge the gap between predictive data science and actionable business strategy. This full-stack pipeline processes raw transaction logs to generate a Live Cross-Sell Recommendation Engine (Market Basket Analysis) and a predictive Customer Churn classifier, surfaced through an interactive SaaS-style web dashboard.

## 🚀 Key Features

* **Advanced Churn Prediction**: Utilizes a `RandomForestClassifier` to map historical behavioral features (AOV, Purchase Frequency, Recency) into future churn probabilities, prioritizing high-value active users based on their Historic Customer Lifetime Value (CLV).
* **Predictive Explainability (SHAP)**: Strips away the "Black Box" of Machine Learning. Uses `TreeExplainer` to generate individual, localized waterfalls explaining exactly *why* a specific user was flagged for churn, building stakeholder trust.
* **FP-Growth Cross-Sell Engine**: Swapped traditional Apriori for the highly-scalable `mlxtend` FP-Growth algorithm to mine millions of associative pairs across a sparse one-hot matrix, generating high-lift product recommendations for a Live Cart UX.
* **Campaign ROI Simulator**: Transforms analytics into decision-support. A dynamic What-If simulator allows stakeholders to adjust discount parameters against customized risk cohorts to project Net Revenue Recovered vs. Campaign Cost.

## ⚙️ Architecture pipeline

1. **`1_download_data.py`** - Pulls the massive open-source UCI Online Retail dataset via automated HTTPS requests.
2. **`2_preprocess_data.py`** - Drops NAs, defines boolean churn windows, aggregates behavioral metrics, and pivots thousands of invoices into a sparse Boolean DataFrame.
3. **`3_market_basket.py`** - Mines the itemset matrix for frequent antecedents/consequents and filters for combinations yielding high Confidence and Lift.
4. **`4_churn_model.py`** - Trains the RandomForest mapping, calculates SHAP base values, outputs strict Model Verification Metrics (AUC-ROC & PR Curves), and scores the active userbase.
5. **`app/dashboard.py`** - The frontend architecture. A sleek, interactive multi-tab Streamlit dashboard injecting premium glassmorphism CSS.

## 📦 Run Locally

**1. Clone & Install Dependencies:**
```bash
git clone https://github.com/yourusername/ecommerce-intelligence.git
cd ecommerce-intelligence
pip install -r requirements.txt
```

**2. Execute the Pipeline:**
Run the scripts sequentially to generate the models and feature data:
```bash
python scripts/1_download_data.py
python scripts/2_preprocess_data.py
python scripts/3_market_basket.py
python scripts/4_churn_model.py
```

**3. Launch the Application:**
```bash
python -m streamlit run app/dashboard.py
```
Open your browser to `http://localhost:8501`.

## 📈 Evaluation Metrics
* Tested on roughly 5,000 uniquely tracked accounts.
* Baseline Churn Rate: ~33%
* Model accuracy: ~71% out-of-the-box ensuring strong resilience across a highly noisy real-world dataset. Validated strictly through True Positive Rate plotting (ROC) rather than threshold accuracy alone.
