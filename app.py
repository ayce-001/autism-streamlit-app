import streamlit as st
import numpy as np
import pickle
import os

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Autism Prediction (Educational)",
    page_icon="🧩",
    layout="centered",
)

st.markdown(
    """
    <style>
        .main {
            background-color: #f5f7fa;
        }
        .stButton>button {
            width: 100%;
            padding: 0.8rem;
            border-radius: 10px;
            background-color: #4a90e2;
            color: white;
            font-size: 18px;
        }
        .stButton>button:hover {
            background-color: #3a78c2;
            color: white;
        }
        .card {
            background-color: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            margin-bottom: 25px;
        }
        .section-title {
            font-size: 20px;
            font-weight: 600;
            color: #4a4a4a;
            padding-bottom: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------- LOAD MODEL ----------------------
MODEL_PATH = "best_model.pkl"
ENCODERS_PATH = "encoders.pkl"

@st.cache_resource
def load_model_and_encoders():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_model_and_encoders()

# Helper for safe label encoding
def safe_encode(val, encoder):
    classes = list(encoder.classes_)
    if val in classes:
        return encoder.transform([val])[0]
    if "Others" in classes:
        return encoder.transform(["Others"])[0]
    return encoder.transform([classes[0]])[0]


# ---------------------- HEADER ----------------------
st.title("🧩 Autism Prediction System")
st.markdown(
    """
    Machine Learning–powered analysis of autism risk indicators.  
    **For educational & research purposes only** — not for diagnosis.
    """
)

# ---------------------- FORM ----------------------
with st.form("prediction_form"):

    # ----- SECTION 1: Basic Info -----
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 1, 120, 5)
    result = c2.number_input("Result Score", value=0.0)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # ----- SECTION 2: A1 - A10 Scores -----
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📝 Questionnaire (A1 - A10)</div>', unsafe_allow_html=True)

    A_scores = {}
    cols = st.columns(5)
    options = ["No (0)", "Yes (1)"]
    for i in range(1, 11):
        A_scores[f"A{i}_Score"] = cols[(i - 1) % 5].selectbox(
            f"A{i}",
            options=options,
            index=0 if i % 2 else 1
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ----- SECTION 3: Categorical Features -----
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🌍 Background Information</div>', unsafe_allow_html=True)

    gender = st.selectbox("Gender", list(encoders["gender"].classes_))
    ethnicity = st.selectbox("Ethnicity", list(encoders["ethnicity"].classes_))
    jaundice = st.selectbox("Jaundice (at birth)", list(encoders["jaundice"].classes_))
    austim = st.selectbox("Self-reported Autism", list(encoders["austim"].classes_))
    country = st.selectbox("Country of Residence", list(encoders["contry_of_res"].classes_))
    used_app = st.selectbox("Used Autism App Before?", list(encoders["used_app_before"].classes_))
    relation = st.selectbox("Relation to Person", list(encoders["relation"].classes_))

    st.markdown("</div>", unsafe_allow_html=True)

    # ----- SUBMIT -----
    submitted = st.form_submit_button("🔍 Predict")

# ---------------------- PREDICTION ----------------------
if submitted:
    try:
        # Build feature sequence
        features = [
            float(age),
            float(result)
        ]

        # A1..A10
        for i in range(1, 11):
            ans = A_scores[f"A{i}_Score"]
            features.append(1 if "1" in ans else 0)

        # Encoded categorical values
        features.extend([
            safe_encode(gender, encoders["gender"]),
            safe_encode(ethnicity, encoders["ethnicity"]),
            safe_encode(jaundice, encoders["jaundice"]),
            safe_encode(austim, encoders["austim"]),
            safe_encode(country, encoders["contry_of_res"]),
            safe_encode(used_app, encoders["used_app_before"]),
            safe_encode(relation, encoders["relation"]),
        ])

        # Convert to numpy
        X = np.array(features).reshape(1, -1)

        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0].max()

        st.markdown('<div class="card">', unsafe_allow_html=True)
        if pred == 1:
            st.error(f"🔴 **Autism Likely**  \nModel Confidence: **{prob:.2f}**")
        else:
            st.success(f"🟢 **Autism Unlikely**  \nModel Confidence: **{prob:.2f}**")
        st.markdown("</div>", unsafe_allow_html=True)

        st.info("This prediction is not a medical diagnosis.")
    
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
