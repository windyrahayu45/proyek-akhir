import streamlit as st
import pandas as pd
import pickle
import json
import numpy as np

# Load model
with open('model_rf.pkl', 'rb') as file:
    model = pickle.load(file)

# Load thresholds
with open('thresholds.json', 'r') as file:
    thresholds = json.load(file)
dropout_threshold = thresholds['dropout_threshold']
enrolled_threshold = thresholds['enrolled_threshold']

# Mapping label angka ke nama status
label_mapping = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}

st.title("🎓 Prediksi Status Mahasiswa - Jaya Jaya Institut")
st.write("Masukkan data mahasiswa untuk memprediksi apakah akan Dropout, Enrolled, atau Graduate.")

# Buat form input
with st.form(key='input_form'):
    st.subheader("Informasi Mahasiswa")
    
    # Ambil semua fitur input sesuai dengan fitur saat training
    marital_status = st.selectbox("Marital Status", [1, 2, 3, 4])
    application_mode = st.selectbox("Application Mode", [1, 2, 5, 10, 15, 39, 44, 51, 53, 57, 58, 62, 65, 72, 73, 75, 77, 81, 82, 99])
    application_order = st.number_input("Application Order", min_value=1, max_value=20, value=1)
    course = st.number_input("Course", min_value=1, value=1)
    daytime_evening_attendance = st.selectbox("Daytime/Evening Attendance", [1, 0])
    previous_qualification = st.selectbox("Previous Qualification", [1, 2, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43, 44, 46, 47, 51, 53, 57, 60, 62, 64, 65, 70, 71, 72, 73, 75, 77, 78, 79, 80])
    previous_qualification_grade = st.number_input("Previous Qualification Grade", min_value=0.0, max_value=200.0, value=100.0)
    nationality = st.selectbox("Nationality", [1, 2, 6, 11, 13, 14, 17, 21, 22, 24, 25, 26, 32, 41, 62, 100, 101])
    mothers_qualification = st.selectbox("Mother's Qualification", [1, 2, 3, 4, 5, 6, 9])
    fathers_qualification = st.selectbox("Father's Qualification", [1, 2, 3, 4, 5, 6, 9])
    mothers_occupation = st.selectbox("Mother's Occupation", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14])
    fathers_occupation = st.selectbox("Father's Occupation", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14])
    admission_grade = st.number_input("Admission Grade", min_value=0.0, max_value=200.0, value=100.0)
    displaced = st.selectbox("Displaced", [0, 1])
    educational_special_needs = st.selectbox("Educational Special Needs", [0, 1])
    debtor = st.selectbox("Debtor", [0, 1])
    tuition_fees_up_to_date = st.selectbox("Tuition Fees Up to Date", [0, 1])
    gender = st.selectbox("Gender", [0, 1])
    scholarship_holder = st.selectbox("Scholarship Holder", [0, 1])
    age_at_enrollment = st.number_input("Age at Enrollment", min_value=15, max_value=80, value=18)
    international = st.selectbox("International", [0, 1])
    curricular_units_1st_sem_grade = st.number_input("Curricular Units 1st Sem Grade", min_value=0.0, max_value=20.0, value=10.0)
    curricular_units_2nd_sem_grade = st.number_input("Curricular Units 2nd Sem Grade", min_value=0.0, max_value=20.0, value=10.0)
    unemployment_rate = st.number_input("Unemployment Rate", min_value=0.0, max_value=50.0, value=10.0)
    gdp = st.number_input("GDP", min_value=0.0, max_value=50000.0, value=20000.0)

    submit_button = st.form_submit_button(label='Prediksi')

if submit_button:
    input_data = pd.DataFrame({
        'marital_status': [marital_status],
        'application_mode': [application_mode],
        'application_order': [application_order],
        'course': [course],
        'daytime_evening_attendance': [daytime_evening_attendance],
        'previous_qualification': [previous_qualification],
        'previous_qualification_grade': [previous_qualification_grade],
        'nacionality': [nationality],
        'mothers_qualification': [mothers_qualification],
        'fathers_qualification': [fathers_qualification],
        'mothers_occupation': [mothers_occupation],
        'fathers_occupation': [fathers_occupation],
        'admission_grade': [admission_grade],
        'displaced': [displaced],
        'educational_special_needs': [educational_special_needs],
        'debtor': [debtor],
        'tuition_fees_up_to_date': [tuition_fees_up_to_date],
        'gender': [gender],
        'scholarship_holder': [scholarship_holder],
        'age_at_enrollment': [age_at_enrollment],
        'international': [international],
        'curricular_units_1st_sem_grade': [curricular_units_1st_sem_grade],
        'curricular_units_2nd_sem_grade': [curricular_units_2nd_sem_grade],
        'unemployment_rate': [unemployment_rate],
        'gdp': [gdp]
    })

    # Predict probability
    probs = model.predict_proba(input_data)[0]

    # Apply threshold logic
    if probs[0] >= dropout_threshold:
        pred = 0  # Dropout
    elif probs[1] >= enrolled_threshold:
        pred = 1  # Enrolled
    else:
        pred = 2  # Graduate

    st.success(f"Prediksi Status Mahasiswa: **{label_mapping[pred]}**")
    st.write(f"Probabilitas (Dropout, Enrolled, Graduate): {np.round(probs, 3)}")
