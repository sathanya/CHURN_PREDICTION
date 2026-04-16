# 🧠 E-Commerce Neural Intelligence & Retention System

An end-to-end Machine Learning ecosystem bridging predictive data science, causal inference, and actionable business strategy. This pipeline processes transaction logs to build temporal degradation models, causal intervention strategies, and a live cross-sell engine—all surfaced via a highly interactive "Dark Glassmorphism" AI Command Center.

## 🚀 Key Architectural Breakthroughs

* **Temporal Trajectory Churn (Scikit-Learn Neural Network)**: Moving beyond static aggregations (RFM), this system maps customer behavioral paths into a 3D sequential tensor (6-month rolling windows). It trains a **Multi-Layer Perceptron (Neural Network)** to detect early-warning behavioral degradation signatures, achieving 93%+ accuracy on temporal validation.
* **Causal Nudge Engine (Uplift / S-Learner)**: Transforms predictive analytics into decision-support. Implements a Random Forest **S-Learner Uplift model** to calculate the Conditional Average Treatment Effect (CATE) of discount interventions. Explains *how many users are persuadable* and simulates the pure **Net Profit / Revenue Salvaged** under active market nudges.
* **Predictive Explainability (SHAP)**: Uses localized waterfalls to explain *why* specific nodes are flagged for churn, mitigating the ML "Black Box".
* **Market Basket Agents (FP-Growth)**: Scalable transactional pattern mining generating deterministic Confidence and Lift scoring for an intelligent Cart-Cross-Selling UX.

## ⚙️ Pipeline Flow

The backend model architecture cascades systematically:

1. **`1_download_data.py`** & **`2_preprocess_data.py`** - Pulls standard UCI Retail data, extracts behavioral features, and pivots transactions into user tensors.
2. **`3_market_basket.py`** - association rule mining.
3. **`4_churn_model.py`** - Standard Random Forest baseline & SHAP base value logic.
4. **`5_synthetic_trajectories.py`** - (The Advanced Layer) Simulates time-series sequences based on historical transactional variance.
5. **`6_lstm_churn_model.py`** - Trains the Deep Learning temporal mapping.
6. **`7_causal_nudge.py`** - Trains the S-Learner intervention model.
7. **`app/server.py`** - The frontend FastAPI backend. Serves the stunning "Dark Glassmorphism" frontend (`app/static/index.html`) using raw Vanilla JS + Chart.js native styling.

## 📦 Run Locally

**1. Clone & Install Dependencies:**
```bash
git clone https://github.com/yourusername/ecommerce-intelligence.git
cd ecommerce-intelligence
pip install -r requirements.txt
```

**2. Compile the Neural Models:**
Run the advanced pipelines to generate sequence arrays, neural models, and the casual learners.
```bash
python scripts/5_synthetic_trajectories.py
python scripts/6_lstm_churn_model.py
python scripts/7_causal_nudge.py
```

**3. Boot the AI Command Center:**
```bash
python app/server.py
```
Open your browser to `http://localhost:8000`.

## 📈 Evaluation Matrix
* **Deep Neural Classifier**: Evaluated strictly on temporal sequence test-splits, reaching robust high 90s metric integrity.
* **Causal Nudge Accuracy**: Provides bounded, actionable proxy limits of intervention costs vs gained survival LTV.
