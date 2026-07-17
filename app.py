import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Healthcare Diagnosis Assistant", page_icon="🩺", layout="wide")

MODEL_PATH = "diagnosis_model.joblib"
SCALER_PATH = "scaler.joblib"


@st.cache_resource
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        return None, None
    model = joblib.load(MODEL_PATH)
    meta = joblib.load(SCALER_PATH)
    return model, meta


model, meta = load_artifacts()

st.title("🩺 AI-Powered Healthcare Diagnosis Assistant")
st.caption("Educational demo: suggests a probable condition from symptoms and vitals using a trained ML model.")

st.warning(
    "⚠️ **This is a student/demo project trained on synthetic data.** "
    "It is not a medical device, has not been clinically validated, and must "
    "never be used for actual diagnosis or treatment. Always consult a "
    "qualified healthcare professional for real medical concerns.",
    icon="⚠️",
)

if model is None:
    st.error(
        "No trained model found. Run `python train_model.py` first "
        "(after generating `patient_data.csv` with generate_synthetic_data.py)."
    )
    st.stop()

scaler = meta["scaler"]
feature_cols = meta["feature_cols"]
label_encoder = meta["label_encoder"]
numeric_cols = meta["numeric_cols"]
binary_cols = meta["binary_cols"]

tab1, tab2 = st.tabs(["🔍 Single Patient Check", "📁 Batch CSV Upload"])

# ---------- Tab 1: Manual single patient ----------
with tab1:
    st.subheader("Enter patient details")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=45)
        sex = st.selectbox("Sex", ["Female", "Male"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.5)
    with col2:
        bp_systolic = st.number_input("Systolic Blood Pressure (mmHg)", min_value=70.0, max_value=250.0, value=120.0)
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100.0, max_value=400.0, value=190.0)
    with col3:
        glucose = st.number_input("Glucose (mg/dL)", min_value=50.0, max_value=400.0, value=95.0)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40.0, max_value=200.0, value=75.0)

    st.write("Symptoms present:")
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    with s_col1:
        fever = st.checkbox("Fever")
        cough = st.checkbox("Cough")
    with s_col2:
        fatigue = st.checkbox("Fatigue")
        chest_pain = st.checkbox("Chest pain")
    with s_col3:
        shortness_of_breath = st.checkbox("Shortness of breath")
        headache = st.checkbox("Headache")
    with s_col4:
        nausea = st.checkbox("Nausea")
        joint_pain = st.checkbox("Joint pain")

    if st.button("🔎 Get Diagnosis Suggestion", type="primary"):
        numeric_vals = [age, bmi, bp_systolic, cholesterol, glucose, heart_rate]
        numeric_scaled = scaler.transform([numeric_vals])[0]

        binary_vals = {
            "sex": 1 if sex == "Male" else 0,
            "fever": int(fever),
            "cough": int(cough),
            "fatigue": int(fatigue),
            "chest_pain": int(chest_pain),
            "shortness_of_breath": int(shortness_of_breath),
            "headache": int(headache),
            "nausea": int(nausea),
            "joint_pain": int(joint_pain),
        }

        row = list(numeric_scaled) + [binary_vals[c] for c in binary_cols]
        X_input = pd.DataFrame([row], columns=feature_cols)

        probs = model.predict_proba(X_input)[0]
        pred_idx = np.argmax(probs)
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        st.divider()
        st.subheader(f"Most likely: **{pred_label}** ({probs[pred_idx]*100:.1f}% confidence)")

        prob_df = pd.DataFrame({
            "Condition": label_encoder.classes_,
            "Probability": probs,
        }).sort_values("Probability", ascending=False)
        st.bar_chart(prob_df.set_index("Condition"))

        st.caption("Reminder: this is a probability estimate from a model trained on synthetic data, not a medical diagnosis.")

# ---------- Tab 2: Batch CSV upload ----------
with tab2:
    st.subheader("Upload a CSV of patients")
    st.write(
        "CSV should have columns: "
        + ", ".join(["age", "sex", "bmi", "bp_systolic", "cholesterol", "glucose", "heart_rate"]
                     + binary_cols[1:])
        + " (sex should be 0=Female, 1=Male; symptom columns should be 0/1)."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(batch_df)} patients.")

        try:
            numeric_scaled = scaler.transform(batch_df[numeric_cols])
            numeric_scaled_df = pd.DataFrame(
                numeric_scaled, columns=[f"{c}_scaled" for c in numeric_cols]
            )
            X_batch = pd.concat(
                [numeric_scaled_df, batch_df[binary_cols].reset_index(drop=True)], axis=1
            )[feature_cols]

            probs = model.predict_proba(X_batch)
            pred_idx = np.argmax(probs, axis=1)
            pred_labels = label_encoder.inverse_transform(pred_idx)
            confidence = probs[np.arange(len(probs)), pred_idx]

            batch_df["Predicted_Diagnosis"] = pred_labels
            batch_df["Confidence"] = confidence

            st.dataframe(
                batch_df[["age", "sex", "Predicted_Diagnosis", "Confidence"]]
                .sort_values("Confidence", ascending=False)
                .head(50),
                use_container_width=True,
            )

            csv_out = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results CSV", csv_out, "diagnosis_predictions.csv", "text/csv")

        except KeyError as e:
            st.error(f"CSV is missing expected column: {e}")

st.divider()
st.caption(
    "Note: This model identifies statistical patterns learned from training data. "
    "Real clinical decision support systems require regulatory approval, "
    "rigorous validation on real patient data, and use under professional "
    "supervision. This demo should never be used for actual medical decisions."
)