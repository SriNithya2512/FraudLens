import streamlit as st
import numpy as np
import xgboost as xgb
import shap
import json
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="FraudLens", layout="centered")
st.title("🔍 FraudLens: Illicit Transaction Detector")
st.write("An explainable ensemble GNN framework for detecting illicit cryptocurrency transactions.")

st.warning(
    "⚠️ **Limitation:** This model only works on transactions already present in our trained "
    "transaction graph — it cannot yet score a brand-new, real-world transaction it has never seen. "
    "This is a known constraint of this type of graph model (called 'transductive' learning). "
    "Extending this to unseen transactions in real time would require an inductive architecture "
    "(e.g. GraphSAGE) — noted as future work in our paper."
)

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

st.subheader("Select a transaction to analyze")

illicit_indices = np.where(y_test == 1)[0][:5]
licit_indices = np.where(y_test == 0)[0][:5]

example_options = {}
for i in illicit_indices:
    example_options[f"Known illicit example #{i}"] = i
for i in licit_indices:
    example_options[f"Known licit example #{i}"] = i

choice = st.selectbox("Pick an example transaction:", list(example_options.keys()))
idx = example_options[choice]

# --- Show the actual raw numbers for this transaction ---
with st.expander("🔢 See the actual data for this transaction"):
    sample_raw = X_test[idx]
    raw_feat_indices = [i for i, name in enumerate(feature_names) if "raw_feat" in name]
    raw_feat_names = [feature_names[i] for i in raw_feat_indices][:15]  # first 15 for readability
    raw_feat_values = [sample_raw[i] for i in raw_feat_indices][:15]

    display_df = pd.DataFrame({
        "Feature": raw_feat_names,
        "Value": [round(v, 4) for v in raw_feat_values]
    })
    st.write("First 15 raw transaction features (of 165 total):")
    st.dataframe(display_df, use_container_width=True)
    st.caption(
        "Note: Elliptic dataset features are anonymized numeric values (not labeled with real-world "
        "names like 'amount' or 'fee') to protect transaction privacy — this is standard for this dataset."
    )

if st.button("Analyze Transaction"):
    sample = X_test[idx].reshape(1, -1)
    prob = model.predict_proba(sample)[0][1]
    pred = "🚨 ILLICIT" if prob >= 0.5 else "✅ LICIT"
    actual = "Illicit" if y_test[idx] == 1 else "Licit"

    st.subheader(f"Prediction: {pred}")
    st.metric("Illicit Probability", f"{prob:.1%}")
    st.write(f"**Ground truth label:** {actual}")

    st.subheader("Why did the model decide this?")
    shap_values = explainer.shap_values(sample)

    contributions = list(zip(feature_names, shap_values[0]))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_5 = contributions[:5]

    def plain_name(fname):
        if "confidence" in fname:
            model_name = "ChebNet" if "chebnet" in fname else "GATv2"
            return f"{model_name}'s own confidence score"
        elif "raw_feat" in fname:
            return f"raw transaction feature #{fname.split('_')[-1]}"
        elif "emb" in fname:
            model_name = "ChebNet" if "chebnet" in fname else "GATv2"
            return f"a learned graph pattern from {model_name}"
        return fname

    st.write("**In simple terms, ranked by influence:**")
    table_rows = []
    for fname, val in top_5:
        direction = "→ ILLICIT" if val > 0 else "→ LICIT"
        table_rows.append({
            "Factor": plain_name(fname),
            "Raw feature name": fname,
            "Impact score": round(val, 3),
            "Direction": direction
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    st.write("")
    st.write("**Full detailed chart (SHAP waterfall):**")
    st.caption(
        "Red bars push toward ILLICIT, blue bars push toward LICIT. Longer bar = stronger influence."
    )

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=sample[0],
        feature_names=feature_names
    ), show=False)
    st.pyplot(fig)
