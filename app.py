import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Banking Fraud Detector", page_icon="🏦", layout="wide")

st.markdown("""
<style>
.fraud-box { background:#ff4b4b22; border:2px solid #ff4b4b; border-radius:12px; padding:20px; text-align:center; }
.safe-box  { background:#21c35422; border:2px solid #21c354; border-radius:12px; padding:20px; text-align:center; }
.bar-wrap  { background:#e0e0e0; border-radius:8px; height:20px; margin-top:6px; }
</style>
""", unsafe_allow_html=True)

# Transaction types
TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
TYPE_ENCODING = {t: i for i, t in enumerate(sorted(TRANSACTION_TYPES))}  # alphabetical

# Features expected by the model
MODEL_FEATURES = [
    "step", "amount", "log_amount", "is_high_amount",
    "hour", "is_night",
    "balance_diff_orig", "balance_diff_dest",
    "type_enc"
]

# Load model
@st.cache_resource
def load_model():
    fname = "fraud_model.pkl"
    if not os.path.exists(fname):
        return None
    model = joblib.load(fname)
    return model

model = load_model()
THRESHOLD = 0.5  # default threshold for fraud

# Feature engineering
def build_features(step, tx_type, amount, old_orig, new_orig, old_dest, new_dest):
    hour = step % 24
    is_night = int(hour in [0,1,2,3,4,5,22,23])
    bal_diff_orig = old_orig - new_orig
    bal_diff_dest = new_dest - old_dest
    type_enc = TYPE_ENCODING.get(tx_type, 0)
    log_amount = np.log1p(amount)
    is_high_amount = 1 if amount > 150000 else 0
    return pd.DataFrame([{
        "step": step,
        "amount": amount,
        "log_amount": log_amount,
        "is_high_amount": is_high_amount,
        "hour": hour,
        "is_night": is_night,
        "balance_diff_orig": bal_diff_orig,
        "balance_diff_dest": bal_diff_dest,
        "type_enc": type_enc
    }])

# Prediction
def predict(feature_df: pd.DataFrame):
    # Align columns exactly as MODEL_FEATURES
    X = feature_df.reindex(columns=MODEL_FEATURES, fill_value=0)
    proba = float(model.predict_proba(X)[0,1])
    label = int(proba >= THRESHOLD)
    return proba, label

# Sidebar
with st.sidebar:
    st.header("🏦 Model Info")
    if model is not None:
        st.success("Model loaded successfully!")
        st.write(f"**Algorithm:** {type(model).__name__}")
        st.write(f"**Threshold:** {THRESHOLD:.2f}")
        st.write("**Features:**")
        for f in MODEL_FEATURES:
            st.caption(f"• {f}")
    else:
        st.error("Model file not found. Place `fraud_model.pkl` in the app folder.")

    st.markdown("---")
    st.markdown("**Risk Levels**")
    st.caption("🟢 LOW — prob < 40%")
    st.caption("🟡 MEDIUM — prob 40–70%")
    st.caption("🔴 HIGH — prob > 70%")

# Main UI
st.title("🏦 Banking Fraud Detection System")
st.caption("Fill transaction details and click Analyse.")
st.markdown("---")

st.subheader("📋 Transaction Details")
c1, c2, c3 = st.columns(3)
with c1:
    step    = st.number_input("Step (1–743)", min_value=1, max_value=743, value=10)
    tx_type = st.selectbox("Transaction Type", TRANSACTION_TYPES)
    amount  = st.number_input("Amount ($)", min_value=0.01, value=5000.00, format="%.2f")
with c2:
    old_orig = st.number_input("Origin — Old Balance ($)", min_value=0.0, value=10000.0, format="%.2f")
    new_orig = st.number_input("Origin — New Balance ($)", min_value=0.0, value=5000.0, format="%.2f")
with c3:
    old_dest = st.number_input("Destination — Old Balance ($)", min_value=0.0, value=2000.0, format="%.2f")
    new_dest = st.number_input("Destination — New Balance ($)", min_value=0.0, value=7000.0, format="%.2f")

st.markdown("")
run = st.button("🔎 Analyse Transaction")

# Results
if run:
    if model is None:
        st.error("No model loaded.")
        st.stop()

    feature_df = build_features(step, tx_type, amount, old_orig, new_orig, old_dest, new_dest)
    fraud_prob, is_fraud = predict(feature_df)

    risk = ("🔴 HIGH" if fraud_prob > 0.7 else
            "🟡 MEDIUM" if fraud_prob > 0.4 else
            "🟢 LOW")
    bar_color = "#ff4b4b" if fraud_prob > 0.7 else "#f0a500" if fraud_prob > 0.4 else "#21c354"

    st.markdown("---")
    st.subheader("📊 Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Fraud Probability", f"{fraud_prob*100:.2f}%")
    m2.metric("Risk Level", risk)
    m3.metric("Verdict", "⚠️ Fraudulent" if is_fraud else "✅ Legitimate")

    st.markdown(f"""
    <div style="margin:14px 0 4px 0; font-size:0.9rem; color:#888;">Fraud Score</div>
    <div class="bar-wrap">
        <div style="width:{fraud_prob*100:.2f}%; background:{bar_color}; height:20px; border-radius:8px;"></div>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#888; margin-top:3px;">
        <span>0%</span><span>{fraud_prob*100:.2f}%</span><span>100%</span>
    </div>
    """, unsafe_allow_html=True)

    if is_fraud:
        st.markdown(
            f'<div class="fraud-box"><h3>⚠️ FRAUD DETECTED</h3>'
            f'<p>Transaction flagged as <strong>likely fraudulent</strong> — '
            f'probability <strong>{fraud_prob*100:.2f}%</strong>.</p></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="safe-box"><h3>✅ TRANSACTION LOOKS SAFE</h3>'
            f'<p>Transaction appears <strong>legitimate</strong> — '
            f'probability <strong>{fraud_prob*100:.2f}%</strong>.</p></div>',
            unsafe_allow_html=True
        )

    with st.expander("🔬 Engineered Features (debug)"):
        st.dataframe(feature_df, use_container_width=True)