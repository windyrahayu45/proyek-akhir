import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine

# Load model
model = joblib.load("model/model_rf.pkl")

# Kolom fitur (isi lengkap)
columns = [
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
]

# Ambil feature importance
importance = model.feature_importances_

# Buat DataFrame
df_importance = pd.DataFrame({
    'feature': columns,
    'importance': importance
}).sort_values(by='importance', ascending=False).head(10)

# Simpan ke database PostgreSQL
engine = create_engine("postgresql://metabase:admin123@localhost:5432/students")
df_importance.to_sql("feature_importance", engine, if_exists='replace', index=False)

print("Feature importance berhasil diekspor ke database PostgreSQL.")
