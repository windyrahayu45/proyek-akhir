import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# Judul
st.set_page_config(page_title="Prediksi Attrition Mahasiswa")
st.title("🎓 Prediksi Attrition Mahasiswa - Jaya Jaya Institut")
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

# Upload file
uploaded_file = st.file_uploader("📤 Upload file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Baca data
        data = pd.read_csv(uploaded_file)
        st.subheader("📄 Data yang Diupload")
        st.dataframe(data)

        # Prediksi
        st.subheader("🔍 Hasil Prediksi")
        probs = model.predict_proba(data)  # probs shape (n_samples, 3)

        def classify(prob):
            if prob[0] >= dropout_threshold:
                return "Dropout"
            elif prob[1] >= enrolled_threshold:
                return "Enrolled"
            else:
                return "Graduate"

        predictions = [classify(p) for p in probs]
        result_df = data.copy()
        result_df['Attrition_Prediction'] = predictions

        st.dataframe(result_df)

        # Visualisasi
        st.subheader("📊 Distribusi Hasil Prediksi")
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
