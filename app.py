import streamlit as st
import numpy as np
import xgboost as xgb
import shap
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="FraudLens", layout="centered")
st.title("🔍 FraudLens: Illicit Transaction Detector")
st.write("An explainable ensemble GNN framework for detecting illicit cryptocurrency transactions.")

@st.cache_resource
def load_everything():
    model = xgb.XGBClassifier()
    model.load_model("models/xgboost_ensemble_final.json")

    X_test = np.load("models/X_test_final.npy")
    y_test = np.load("models/y_test.npy")

    with open("models/feature_names.json") as f:
        feature_names = json.load(f)

    explainer = shap.TreeExplainer(model)

    return model, X_test, y_test, feature_names, explainer

model, X_test, y_test, feature_names, explainer = load_everything()

# Let user pick a transaction
st.subheader("Select a transaction to analyze")
idx = st.number_input(
    f"Transaction index (0 to {len(X_test)-1})",
    min_value=0, max_value=len(X_test)-1, value=0, step=1
)

if st.button("Analyze Transaction"):
    sample = X_test[idx].reshape(1, -1)
    prob = model.predict_proba(sample)[0][1]
    pred = "🚨 ILLICIT" if prob >= 0.5 else "✅ LICIT"
    actual = "Illicit" if y_test[idx] == 1 else "Licit"

    st.subheader(f"Prediction: {pred}")
    st.write(f"Confidence (illicit probability): **{prob:.2%}**")
    st.write(f"Actual label (ground truth): **{actual}**")

    st.subheader("Why this prediction? (SHAP Explanation)")
    shap_values = explainer.shap_values(sample)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=sample[0],
        feature_names=feature_names
    ), show=False)
    st.pyplot(fig)
