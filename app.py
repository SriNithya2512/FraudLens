import streamlit as st
import numpy as np
import xgboost as xgb
import shap
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="FraudLens", layout="centered")
st.title("🔍 FraudLens: Illicit Transaction Detector")
st.write("An explainable ensemble GNN framework for detecting illicit cryptocurrency transactions.")

st.info(
    "⚠️ **Important note:** This demo works on transactions from our evaluation dataset "
    "(the Elliptic Bitcoin dataset), since our models need each transaction's position in the "
    "known transaction graph to make predictions. This is a common limitation of this type of "
    "graph model — extending it to brand-new, real-time transactions is noted as future work."
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

# --- Better selection: curated examples, not raw index ---
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

    # --- Plain-English summary, generated from SHAP values ---
    contributions = list(zip(feature_names, shap_values[0]))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_3 = contributions[:3]

    def plain_name(fname):
        if "confidence" in fname:
            model_name = "ChebNet" if "chebnet" in fname else "GATv2"
            return f"the {model_name} model's own confidence score"
        elif "raw_feat" in fname:
            return f"a raw transaction property (feature #{fname.split('_')[-1]})"
        elif "emb" in fname:
            model_name = "ChebNet" if "chebnet" in fname else "GATv2"
            return f"a learned pattern from the {model_name} graph model"
        return fname

    st.write("**In simple terms:**")
    explanation_lines = []
    for fname, val in top_3:
        direction = "pushed toward ILLICIT" if val > 0 else "pushed toward LICIT"
        strength = "strongly" if abs(val) > 1 else "moderately" if abs(val) > 0.3 else "slightly"
        explanation_lines.append(f"- {plain_name(fname)} {strength} {direction} (impact: {val:+.2f})")

    for line in explanation_lines:
        st.write(line)

    st.write("")
    st.write("**Detailed breakdown (SHAP waterfall chart):**")
    st.caption(
        "Each bar shows one factor's push toward ILLICIT (red, pointing right) or LICIT (blue, pointing left). "
        "Longer bars = stronger influence on this specific decision."
    )

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=sample[0],
        feature_names=feature_names
    ), show=False)
    st.pyplot(fig)
