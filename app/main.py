import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# Judul
st.set_page_config(page_title="Prediksi Attrition Mahasiswa")
st.title("\U0001F393 Prediksi Attrition Mahasiswa - Jaya Jaya Institut")
st.write("Upload data mahasiswa untuk memprediksi kemungkinan dropout berdasarkan model machine learning yang telah dilatih.")

# Load model dan threshold
try:
    model = joblib.load('model/model_rf.pkl')
    with open('model/thresholds.json', 'r') as f:
        thresholds = json.load(f)
    dropout_threshold = thresholds['dropout_threshold']
    enrolled_threshold = thresholds['enrolled_threshold']
except Exception as e:
    st.error(f"❌ Gagal memuat model atau threshold: {e}")
    st.stop()

# Kolom-kolom yang digunakan saat training model
feature_columns = [
    'marital_status', 'application_mode', 'application_order', 'course',
    'daytime_evening_attendance', 'previous_qualification',
    'previous_qualification_grade', 'nacionality', 'mothers_qualification',
    'fathers_qualification', 'mothers_occupation', 'fathers_occupation',
    'admission_grade', 'displaced', 'educational_special_needs', 'debtor',
    'tuition_fees_up_to_date', 'gender', 'scholarship_holder',
    'age_at_enrollment', 'international', 'curricular_units_1st_sem_credited',
    'curricular_units_1st_sem_enrolled', 'curricular_units_1st_sem_evaluations',
    'curricular_units_1st_sem_approved', 'curricular_units_1st_sem_grade',
    'curricular_units_1st_sem_without_evaluations', 'curricular_units_2nd_sem_credited',
    'curricular_units_2nd_sem_enrolled', 'curricular_units_2nd_sem_evaluations',
    'curricular_units_2nd_sem_approved', 'curricular_units_2nd_sem_grade',
    'curricular_units_2nd_sem_without_evaluations', 'unemployment_rate', 'inflation_rate',
    'gdp'
]

# Upload file
uploaded_file = st.file_uploader("\U0001F4E4 Upload file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Baca data
        data = pd.read_csv(uploaded_file)
        st.subheader("\U0001F4C4 Data yang Diupload")
        st.dataframe(data)

        # Filter data hanya kolom yang dipakai model
        try:
            data_for_prediction = data[feature_columns]
        except KeyError as e:
            st.error(f"Data yang diupload tidak memiliki kolom yang sesuai: {e}")
            st.stop()

        # Prediksi
        st.subheader("\U0001F50D Hasil Prediksi")
        probs = model.predict_proba(data_for_prediction)[:, 1]  # Ambil probabilitas dropout

        def classify(prob):
            if prob >= dropout_threshold:
                return "Dropout"
            elif prob <= enrolled_threshold:
                return "Enrolled"
            else:
                return "Uncertain"

        predictions = [classify(p) for p in probs]
        result_df = data.copy()
        result_df['Attrition_Prediction'] = predictions

        st.dataframe(result_df)

        # Visualisasi
        st.subheader("\U0001F4CA Distribusi Hasil Prediksi")
        fig, ax = plt.subplots()
        result_df['Attrition_Prediction'].value_counts().plot.pie(
            autopct='%1.1f%%', ax=ax, startangle=90, colors=['#FF9999','#99FF99','#CCCCFF'])
        ax.set_ylabel('')
        st.pyplot(fig)

        # Download hasil
        st.subheader("⬇️ Download Hasil Prediksi")
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='hasil_prediksi_attrition.csv',
            mime='text/csv'
        )
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
else:
    st.info("Silakan upload file CSV untuk memulai prediksi.")
