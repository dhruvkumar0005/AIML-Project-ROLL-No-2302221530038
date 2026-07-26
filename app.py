import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("Heart Disease Prediction")
st.write("Use the form below to enter patient details and predict heart disease risk.")

model = joblib.load("outputs/logistic_model.joblib")
scaler = joblib.load("outputs/standard_scaler.joblib")

feature_columns = [
    'age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca',
    'cp_0', 'cp_1', 'cp_2', 'cp_3',
    'restecg_0', 'restecg_1', 'restecg_2',
    'slope_0', 'slope_1', 'slope_2',
    'thal_1', 'thal_2', 'thal_3',
    'age_decade_20', 'age_decade_30', 'age_decade_40',
    'age_decade_50', 'age_decade_60', 'age_decade_70'
]

with st.form(key='input_form'):
    age = st.number_input("Age", min_value=18, max_value=120, value=55)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest pain type (cp)", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting blood pressure (trestbps)", min_value=80, max_value=250, value=130)
    chol = st.number_input("Serum cholesterol (chol)", min_value=100, max_value=600, value=220)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", options=[0, 1])
    restecg = st.selectbox("Resting ECG (restecg)", options=[0, 1, 2])
    thalach = st.number_input("Max heart rate achieved (thalach)", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise induced angina (exang)", options=[0, 1])
    oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, format="%.1f")
    slope = st.selectbox("Slope of ST segment (slope)", options=[0, 1, 2])
    ca = st.selectbox("Number of major vessels colored by fluoroscopy (ca)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thal (thal)", options=[1, 2, 3])

    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    input_data = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'ca': ca,
        'cp': cp,
        'restecg': restecg,
        'slope': slope,
        'thal': thal
    }])

    input_data['age_decade'] = (input_data['age'] // 10) * 10
    input_data['age_decade'] = input_data['age_decade'].astype(int)

    input_encoded = pd.get_dummies(input_data, columns=['cp', 'restecg', 'slope', 'thal', 'age_decade'], prefix=['cp', 'restecg', 'slope', 'thal', 'age_decade'], drop_first=False)

    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[feature_columns]

    continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    input_encoded[continuous_cols] = scaler.transform(input_encoded[continuous_cols])

    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0][1]

    st.subheader("Prediction")
    if prediction == 1:
        st.error("The model predicts the presence of heart disease.")
    else:
        st.success("The model predicts no heart disease.")

    st.write(f"Probability of heart disease: {proba:.2f}")
    st.write("---")
    st.write("### Input summary")
    st.write(input_data)
