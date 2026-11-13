"""
Pregnancy Health Risk Prediction System
Streamlit app with Random Forest + SHAP + LIME explanations.
Deploy via GitHub → Streamlit Community Cloud.
"""
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "pregnancy_risk_model.joblib"

st.set_page_config(
    page_title="Pregnancy Health Risk Prediction",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load model (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifact():
    artifact = joblib.load(MODEL_PATH)
    return artifact


artifact = load_artifact()
model = artifact["model"]
le = artifact["label_encoder"]
FEATURE_NAMES = artifact["feature_names"]
METRICS = artifact.get("metrics", {})

RISK_COLORS = {
    "low risk": "#2ecc71",
    "mid risk": "#f39c12",
    "high risk": "#e74c3c",
}

FEATURE_HELP = {
    "Age": "Age of the pregnant woman in years (typical range in training data: ~15–50+).",
    "SystolicBP": "Systolic (upper) blood pressure in mmHg.",
    "DiastolicBP": "Diastolic (lower) blood pressure in mmHg.",
    "BS": "Blood glucose / blood sugar level in mmol/L.",
    "BodyTemp": "Body temperature in °F (dataset uses Fahrenheit).",
    "HeartRate": "Resting heart rate in beats per minute (bpm).",
}

# ---------------------------------------------------------------------------
# Sidebar – patient inputs
# ---------------------------------------------------------------------------
st.sidebar.title("Patient measurements")
st.sidebar.caption("Enter clinical values, then click **Predict risk**.")

age = st.sidebar.number_input("Age (years)", min_value=10, max_value=70, value=28, step=1, help=FEATURE_HELP["Age"])
systolic = st.sidebar.number_input("Systolic BP (mmHg)", min_value=70, max_value=180, value=120, step=1, help=FEATURE_HELP["SystolicBP"])
diastolic = st.sidebar.number_input("Diastolic BP (mmHg)", min_value=40, max_value=120, value=80, step=1, help=FEATURE_HELP["DiastolicBP"])
bs = st.sidebar.number_input("Blood sugar BS (mmol/L)", min_value=5.0, max_value=20.0, value=7.0, step=0.1, help=FEATURE_HELP["BS"])
body_temp = st.sidebar.number_input("Body temperature (°F)", min_value=96.0, max_value=104.0, value=98.0, step=0.1, help=FEATURE_HELP["BodyTemp"])
heart_rate = st.sidebar.number_input("Heart rate (bpm)", min_value=40, max_value=120, value=76, step=1, help=FEATURE_HELP["HeartRate"])

predict_btn = st.sidebar.button("Predict risk", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Model:** Random Forest  \n"
    f"**Test accuracy:** {METRICS.get('test_accuracy', 0):.1%}  \n"
    f"**CV accuracy:** {METRICS.get('cv_mean', 0):.1%} ± {METRICS.get('cv_std', 0):.1%}"
)
st.sidebar.caption(
    "⚠️ Educational / research tool only. Not a medical diagnosis. "
    "Always consult a qualified healthcare professional."
)

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------
st.title("🤰 Pregnancy Health Risk Prediction System")
st.markdown(
    "Predict **low / mid / high** maternal health risk from six clinical measurements, "
    "with **SHAP** (global & local) and **LIME** explanations."
)

tab_predict, tab_shap, tab_lime, tab_about = st.tabs(
    ["Prediction", "SHAP explanation", "LIME explanation", "About & model"]
)

# Build feature vector
x_df = pd.DataFrame(
    [[age, systolic, diastolic, bs, body_temp, heart_rate]],
    columns=FEATURE_NAMES,
)

# ---------------------------------------------------------------------------
# Prediction tab
# ---------------------------------------------------------------------------
with tab_predict:
    if predict_btn or "last_pred" not in st.session_state:
        proba = model.predict_proba(x_df)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = le.inverse_transform([pred_idx])[0]
        st.session_state["last_pred"] = {
            "label": pred_label,
            "proba": proba,
            "x_df": x_df.copy(),
        }

    result = st.session_state["last_pred"]
    label = result["label"]
    proba = result["proba"]

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        st.subheader("Predicted risk level")
        color = RISK_COLORS.get(label, "#333")
        st.markdown(
            f"<div style='background:{color};padding:1.2rem 1.5rem;border-radius:12px;"
            f"color:white;font-size:1.6rem;font-weight:700;text-align:center'>{label.upper()}</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.subheader("Class probabilities")
        prob_df = pd.DataFrame(
            {"Risk level": le.classes_, "Probability": proba}
        ).sort_values("Probability", ascending=False)
        st.dataframe(
            prob_df.style.format({"Probability": "{:.1%}"}),
            hide_index=True,
            use_container_width=True,
        )
    with c3:
        st.subheader("Your inputs")
        st.table(x_df.T.rename(columns={0: "Value"}))

    st.markdown("### Probability chart")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    colors = [RISK_COLORS.get(c, "#888") for c in le.classes_]
    bars = ax.barh(le.classes_, proba, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    for bar, p in zip(bars, proba):
        ax.text(min(p + 0.02, 0.95), bar.get_y() + bar.get_height() / 2, f"{p:.1%}", va="center")
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Open the **SHAP explanation** and **LIME explanation** tabs to see why the model "
        "made this prediction for the current inputs."
    )

# ---------------------------------------------------------------------------
# SHAP tab
# ---------------------------------------------------------------------------
with tab_shap:
    st.subheader("SHAP explanations")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) attributes the prediction to each feature "
        "using game-theoretic Shapley values. For this Random Forest we use "
        "**TreeExplainer** (exact and fast on trees)."
    )

    try:
        import shap

        @st.cache_resource
        def get_explainer(_model):
            return shap.TreeExplainer(_model)

        explainer = get_explainer(model)

        # Local explanation for current input
        st.markdown("#### Local explanation (this patient)")
        shap_values = explainer.shap_values(x_df)

        # shap_values may be list (one array per class) for multiclass RF
        if isinstance(shap_values, list):
            # Align with predicted class
            pred_idx = int(np.argmax(result["proba"]))
            local_vals = np.array(shap_values[pred_idx][0])
            base = explainer.expected_value[pred_idx]
        else:
            # newer shap sometimes returns (n_samples, n_features, n_classes)
            if shap_values.ndim == 3:
                pred_idx = int(np.argmax(result["proba"]))
                local_vals = shap_values[0, :, pred_idx]
                base = explainer.expected_value[pred_idx]
            else:
                local_vals = shap_values[0]
                base = float(np.mean(explainer.expected_value))

        # Waterfall-style bar for local SHAP
        order = np.argsort(np.abs(local_vals))
        fig, ax = plt.subplots(figsize=(8, 3.5))
        colors_local = ["#e74c3c" if v < 0 else "#2ecc71" for v in local_vals[order]]
        ax.barh(np.array(FEATURE_NAMES)[order], local_vals[order], color=colors_local)
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_xlabel(f"SHAP value (impact on '{result['label']}' score)")
        ax.set_title(f"Local SHAP — predicted: {result['label']}")
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            f"Base value (average model output for this class): **{base:.3f}**. "
            "Positive SHAP values push toward the predicted class; negative values push away."
        )

        # Global feature importance from model + mean |SHAP| on a sample of training-like grid
        st.markdown("#### Global feature importance (model impurity + mean |SHAP|)")
        # Use a small background sample for mean |SHAP|
        rng = np.random.RandomState(42)
        # synthetic background around dataset ranges for speed (no need to ship full train set)
        bg = pd.DataFrame(
            {
                "Age": rng.randint(15, 50, 80),
                "SystolicBP": rng.randint(90, 140, 80),
                "DiastolicBP": rng.randint(60, 95, 80),
                "BS": rng.uniform(6.0, 12.0, 80),
                "BodyTemp": rng.uniform(98.0, 101.0, 80),
                "HeartRate": rng.randint(65, 90, 80),
            }
        )[FEATURE_NAMES]

        sv_bg = explainer.shap_values(bg)
        if isinstance(sv_bg, list):
            mean_abs = np.mean([np.abs(s).mean(axis=0) for s in sv_bg], axis=0)
        elif sv_bg.ndim == 3:
            mean_abs = np.abs(sv_bg).mean(axis=(0, 2))
        else:
            mean_abs = np.abs(sv_bg).mean(axis=0)

        imp_df = pd.DataFrame(
            {
                "Feature": FEATURE_NAMES,
                "RF impurity importance": model.feature_importances_,
                "Mean |SHAP|": mean_abs,
            }
        ).sort_values("Mean |SHAP|", ascending=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        axes[0].barh(imp_df["Feature"], imp_df["RF impurity importance"], color="#3498db")
        axes[0].set_title("Random Forest impurity importance")
        axes[0].set_xlabel("Importance")
        axes[1].barh(imp_df["Feature"], imp_df["Mean |SHAP|"], color="#9b59b6")
        axes[1].set_title("Mean |SHAP| (approx. global)")
        axes[1].set_xlabel("Mean |SHAP|")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")
        st.exception(e)

# ---------------------------------------------------------------------------
# LIME tab
# ---------------------------------------------------------------------------
with tab_lime:
    st.subheader("LIME explanation (local)")
    st.markdown(
        "LIME (Local Interpretable Model-agnostic Explanations) fits a simple interpretable "
        "model around **this** prediction to show which feature values support or oppose the "
        "predicted class."
    )

    try:
        from lime.lime_tabular import LimeTabularExplainer

        @st.cache_resource
        def get_lime_explainer():
            # Background distribution matching typical ranges (lightweight, no full CSV needed at runtime)
            rng = np.random.RandomState(0)
            train_like = np.column_stack(
                [
                    rng.randint(15, 50, 200),
                    rng.randint(90, 150, 200),
                    rng.randint(60, 100, 200),
                    rng.uniform(6.0, 15.0, 200),
                    rng.uniform(98.0, 102.0, 200),
                    rng.randint(60, 95, 200),
                ]
            ).astype(float)
            return LimeTabularExplainer(
                train_like,
                feature_names=FEATURE_NAMES,
                class_names=list(le.classes_),
                mode="classification",
                discretize_continuous=True,
                random_state=42,
            )

        lime_exp = get_lime_explainer()
        x_row = x_df.values[0]

        with st.spinner("Computing LIME explanation…"):
            explanation = lime_exp.explain_instance(
                x_row,
                model.predict_proba,
                num_features=6,
                top_labels=1,
            )

        pred_idx = int(np.argmax(result["proba"]))
        # Prefer label matching prediction
        label_ids = explanation.available_labels()
        use_label = pred_idx if pred_idx in label_ids else label_ids[0]

        local_list = explanation.as_list(label=use_label)
        feats = [t[0] for t in local_list]
        weights = [t[1] for t in local_list]

        fig, ax = plt.subplots(figsize=(8, 3.5))
        colors_l = ["#e74c3c" if w < 0 else "#2ecc71" for w in weights]
        ax.barh(feats[::-1], weights[::-1], color=colors_l[::-1])
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_xlabel("LIME weight (toward predicted class)")
        ax.set_title(f"LIME — local explanation for '{le.classes_[use_label]}'")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**LIME feature contributions**")
        lime_df = pd.DataFrame(local_list, columns=["Rule / feature", "Weight"])
        st.dataframe(lime_df, hide_index=True, use_container_width=True)

        st.caption(
            "Green bars support the predicted risk class; red bars push against it. "
            "LIME is a local approximation — use it together with SHAP, not alone."
        )

    except Exception as e:
        st.error(f"LIME explanation failed: {e}")
        st.exception(e)

# ---------------------------------------------------------------------------
# About tab
# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("About this system")
    st.markdown(
        """
**Goal.** Estimate maternal health risk level (**low / mid / high**) during pregnancy from six
routinely available measurements, and explain each prediction with SHAP and LIME.

**Data.** Public Maternal Health Risk dataset (IoT-based collection from maternal health
facilities in Bangladesh; ~1,000 records). Features: Age, SystolicBP, DiastolicBP, BS
(blood sugar), BodyTemp (°F), HeartRate.

**Model.** scikit-learn **Random Forest** classifier (`n_estimators=80`, `max_depth=12`,
class-balanced subsample). Chosen for:
- solid accuracy on this tabular task,
- small serialized size (~1.3 MB) suitable for GitHub → Streamlit Cloud,
- native support for SHAP TreeExplainer.

**Explainability.**
- **SHAP TreeExplainer** — exact Shapley values for trees; local waterfall for the current
  patient and approximate global mean |SHAP|.
- **LIME** — local linear surrogate around the current instance for model-agnostic insight.

**Disclaimer.** This application is for **education and research demonstration only**.
It is **not** a medical device and must **not** be used as the sole basis for clinical
decisions. Always seek advice from a qualified healthcare professional.
        """
    )
    st.markdown("#### Feature reference")
    st.table(
        pd.DataFrame(
            {
                "Feature": FEATURE_NAMES,
                "Description": [FEATURE_HELP[f] for f in FEATURE_NAMES],
            }
        )
    )
    st.markdown("#### Deployment")
    st.code(
        "pip install -r requirements.txt\n"
        "streamlit run app.py\n\n"
        "# Or push this repo to GitHub and deploy at https://share.streamlit.io",
        language="bash",
    )
