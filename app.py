import streamlit as st
import numpy as np
import joblib

# Load trained models
heart_model = joblib.load("heart_disease_model.joblib")
parkinsons_model = joblib.load("parkinsons_data.joblib")
diabetes_model = joblib.load("diabetes_model.joblib")
st.set_page_config(page_title="Multiple Disease Prediction", layout="wide")

st.title("🩺 Multiple Disease Prediction System")

disease = st.sidebar.selectbox(
    "Select Disease",
    (  "Parkinson's Disease","Diabetes", "Heart Disease")
)
if disease == "Heart Disease":
    st.subheader("❤️ Heart Disease Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 1, 120)
        sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
        cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure")

    with col2:
        chol = st.number_input("Cholesterol")
        fbs = st.selectbox("Fasting Blood Sugar > 120 (1=True)", [0, 1])
        restecg = st.selectbox("Rest ECG (0–2)", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate")

    with col3:
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("Oldpeak")
        slope = st.selectbox("Slope (0–2)", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
        thal = st.selectbox("Thal (1=Normal, 2=Fixed, 3=Reversible)", [1, 2, 3])

    if st.button("Predict Heart Disease"):
        input_data = np.array([[age, sex, cp, trestbps, chol,
                                fbs, restecg, thalach,
                                exang, oldpeak, slope, ca, thal]])

        prediction = heart_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Person is likely to have Heart Disease")
        else:
            st.success("✅ Person is unlikely to have Heart Disease")
if disease == "Parkinson's Disease":
    st.subheader("🧠 Parkinson's Disease Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        fo = st.number_input("MDVP:Fo(Hz)")
        fhi = st.number_input("MDVP:Fhi(Hz)")
        flo = st.number_input("MDVP:Flo(Hz)")
        jitter = st.number_input("MDVP:Jitter(%)")
        jitter_abs = st.number_input("MDVP:Jitter(Abs)")
        rap = st.number_input("MDVP:RAP")
        ppq = st.number_input("MDVP:PPQ")
        ddp = st.number_input("Jitter:DDP")

    with col2:
        shimmer = st.number_input("MDVP:Shimmer")
        shimmer_db = st.number_input("MDVP:Shimmer(dB)")
        apq3 = st.number_input("Shimmer:APQ3")
        apq5 = st.number_input("Shimmer:APQ5")
        apq = st.number_input("MDVP:APQ")
        dda = st.number_input("Shimmer:DDA")
        nhr = st.number_input("NHR")
        hnr = st.number_input("HNR")

    with col3:
        rpde = st.number_input("RPDE")
        dfa = st.number_input("DFA")
        spread1 = st.number_input("Spread1")
        spread2 = st.number_input("Spread2")
        d2 = st.number_input("D2")
        ppe = st.number_input("PPE")

    if st.button("Predict Parkinson's"):
        input_data = np.array([[fo, fhi, flo, jitter, jitter_abs, rap, ppq, ddp,
                                shimmer, shimmer_db, apq3, apq5, apq, dda,
                                nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])

        prediction = parkinsons_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Parkinson's Disease Detected")
        else:
            st.success("✅ No Parkinson's Disease Detected")


if disease == "Diabetes":
    st.subheader("🩸 Diabetes Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0)
        glucose = st.number_input("Glucose Level")
        blood_pressure = st.number_input("Blood Pressure")

    with col2:
        skin_thickness = st.number_input("Skin Thickness")
        insulin = st.number_input("Insulin Level")
        bmi = st.number_input("BMI")

    with col3:
        dpf = st.number_input("Diabetes Pedigree Function")
        age = st.number_input("Age", min_value=1)

    if st.button("Predict Diabetes"):
        input_data = np.array([[pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi, dpf, age]])

        prediction = diabetes_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Person is likely to have Diabetes")
        else:
            st.success("✅ Person is not Diabetic")
