import pandas as pd
import joblib
from sqlalchemy import create_engine

# Load model
model = joblib.load("model/model_rf.pkl")

# Load data mahasiswa
df = pd.read_csv("data/data_cleaned.csv")  

# Simpan student_id sebelum preprocessing
student_ids =  df.index + 1   # sesuaikan jika nama kolom berbeda

# Pisahkan fitur (pastikan urutan kolom sama seperti saat training)
X = df[[
    'marital_status', 'application_mode', 'application_order', 'course',
    'daytime_evening_attendance', 'previous_qualification',
    'previous_qualification_grade', 'nacionality', 'mothers_qualification',
    'fathers_qualification', 'mothers_occupation', 'fathers_occupation',
    'admission_grade', 'displaced', 'educational_special_needs', 'debtor',
    'tuition_fees_up_to_date', 'gender', 'scholarship_holder',
    'age_at_enrollment', 'international', 'curricular_units_1st_sem_credited',
    'curricular_units_1st_sem_enrolled', 'curricular_units_1st_sem_evaluations',
    'curricular_units_1st_sem_approved', 'curricular_units_1st_sem_grade',
    'curricular_units_1st_sem_without_evaluations',
    'curricular_units_2nd_sem_credited', 'curricular_units_2nd_sem_enrolled',
    'curricular_units_2nd_sem_evaluations', 'curricular_units_2nd_sem_approved',
    'curricular_units_2nd_sem_grade', 'curricular_units_2nd_sem_without_evaluations',
    'unemployment_rate', 'inflation_rate', 'gdp'
]]

# Prediksi probabilitas
probabilities = model.predict_proba(X)
predictions = model.predict(X)

# Buat dataframe hasil prediksi
df_pred = pd.DataFrame({
    'student_id': student_ids,
    'prob_dropout': probabilities[:, 0],
    'prob_enrolled': probabilities[:, 1],
    'prob_graduate': probabilities[:, 2],
    'predicted_status': predictions
})

# Simpan ke PostgreSQL
engine = create_engine("postgresql://metabase:admin123@localhost:5432/students")
df_pred.to_sql("predictions", engine, if_exists='replace', index=False)

print("Hasil prediksi berhasil disimpan ke database PostgreSQL.")
