import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# --- Helper function ---
def highlight_prediction(row):
    color = ''
    if row['Attrition_Prediction'] == 'Dropout':
        color = 'background-color: #ffcccc'
    elif row['Attrition_Prediction'] == 'Enrolled':
        color = 'background-color: #ccffcc'
    elif row['Attrition_Prediction'] == 'Graduate':
        color = 'background-color: #ccccff'
    return [color] * len(row)

# --- Config ---
st.set_page_config(page_title="Prediksi Attrition Mahasiswa", layout="wide")

# --- Title ---
st.title("🎓 Prediksi Attrition Mahasiswa - Jaya Jaya Institute")
st.write("Upload data mahasiswa atau isi manual untuk memprediksi kemungkinan dropout.")

# --- Load Model and Thresholds ---
with st.spinner('Loading model...'):
    try:
        model = joblib.load('model/model_rf.pkl')
        with open('model/thresholds.json', 'r') as f:
            thresholds = json.load(f)
        dropout_threshold = thresholds['dropout_threshold']
        enrolled_threshold = thresholds['enrolled_threshold']
    except Exception as e:
        st.error(f"❌ Gagal memuat model atau threshold: {e}")
        st.stop()

# --- Sidebar ---
st.sidebar.header("📂 Upload Data atau Isi Manual")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
manual_input = st.sidebar.checkbox("Atau Isi Data Manual")

# --- Main ---
data = None

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("✅ Data CSV berhasil diupload!")
    except Exception as e:
        st.error(f"❌ Gagal membaca file CSV: {e}")
        st.stop()

elif manual_input:
    st.sidebar.subheader("📝 Input Data Manual")

    with open('data_cleaned_columns.json', 'r') as f:
        expected_columns = json.load(f)

    manual_data = {}
    for col in expected_columns:
        manual_data[col] = st.sidebar.number_input(f"{col}", value=0.0)
    data = pd.DataFrame([manual_data])
    st.success("✅ Data manual siap diprediksi!")

else:
    st.info("Silakan upload file CSV atau isi data manual di sidebar.")

if data is not None:

    st.subheader("📄 Data Input")
    st.dataframe(data)

    # --- Check Column Names ---
    with open('data_cleaned_columns.json', 'r') as f:
        expected_columns = json.load(f)

    if list(data.columns) != expected_columns:
        st.error("❌ Kolom dalam data tidak sesuai dengan model yang dilatih.")
        st.write("Diharapkan kolom:", expected_columns)
        st.stop()

    # --- Prediksi ---
    with st.spinner('🔍 Sedang melakukan prediksi...'):
        probs = model.predict_proba(data)

        def classify(prob):
            if prob[0] >= dropout_threshold:
                return "Dropout", prob[0]*100
            elif prob[1] >= enrolled_threshold:
                return "Enrolled", prob[1]*100
            else:
                return "Graduate", prob[2]*100

        preds_conf = [classify(p) for p in probs]
        preds = [p[0] for p in preds_conf]
        confs = [p[1] for p in preds_conf]

        result_df = data.copy()
        result_df['Attrition_Prediction'] = preds
        result_df['Confidence (%)'] = np.round(confs, 2)

        st.subheader("🔍 Hasil Prediksi")
        st.dataframe(result_df.style.apply(highlight_prediction, axis=1))

        # --- Pie Chart ---
        st.subheader("📊 Distribusi Prediksi")
        fig, ax = plt.subplots()
        result_df['Attrition_Prediction'].value_counts().plot.pie(
            autopct='%1.1f%%', ax=ax, startangle=90,
            colors=['#FF9999', '#99FF99', '#CCCCFF']
        )
        ax.set_ylabel('')
        st.pyplot(fig)

        # --- Download Hasil ---
        st.subheader("⬇️ Download Hasil Prediksi")
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='hasil_prediksi_attrition.csv',
            mime='text/csv'
        )