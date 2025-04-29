import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from imblearn.over_sampling import SMOTE
from collections import Counter
from itertools import product

st.set_page_config(page_title="Jaya Jaya Institute - Student Status Prediction", layout="wide")
st.title("\ud83c\udf93 Jaya Jaya Institute - Predict Student Status")

# Sidebar
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ("Upload & Clean Data", "Train Model", "Evaluate Model", "Predict New Data"))

# Load Model if available
def load_model():
    with open('model/model_rf.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/thresholds.json', 'r') as f:
        thresholds = json.load(f)
    return model, thresholds

if "model" not in st.session_state:
    try:
        st.session_state.model, st.session_state.thresholds = load_model()
    except:
        st.session_state.model, st.session_state.thresholds = None, None

# Step 1 - Upload & Clean Data
if menu == "Upload & Clean Data":
    uploaded_file = st.file_uploader("Upload your data.csv file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, sep=';')
        data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
        st.write("Original Data:")
        st.dataframe(data.head())

        # Drop duplicates
        data = data.drop_duplicates()

        # Convert numeric columns
        numeric_cols = [
            'previous_qualification_grade', 'admission_grade',
            'curricular_units_1st_sem_grade', 'curricular_units_2nd_sem_grade',
            'unemployment_rate', 'gdp'
        ]
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # Remove outliers
        for col in numeric_cols:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                data = data[(data[col] >= lower) & (data[col] <= upper)]

        data.reset_index(drop=True, inplace=True)
        st.success("Data cleaned successfully!")
        st.dataframe(data.head())

        st.session_state.cleaned_data = data

# Step 2 - Train Model
elif menu == "Train Model":
    if "cleaned_data" not in st.session_state:
        st.warning("Please upload and clean data first.")
    else:
        data = st.session_state.cleaned_data
        X = data.drop('status', axis=1)
        y = data['status'].map({'Dropout':0, 'Enrolled':1, 'Graduate':2})

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # SMOTE
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        st.write("Training Random Forest with GridSearchCV...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced']
        }

        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, scoring='accuracy', cv=3, n_jobs=-1)
        grid_search.fit(X_train_resampled, y_train_resampled)

        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)

        st.write("Test Set Evaluation:")
        st.text(classification_report(y_test, y_pred, target_names=['Dropout', 'Enrolled', 'Graduate']))

        # Threshold Tuning
        y_probs = best_rf.predict_proba(X_test)
        dropout_thresholds = np.arange(0.3, 0.6, 0.05)
        enrolled_thresholds = np.arange(0.3, 0.6, 0.05)

        best_score = 0
        best_thresholds = (0.5, 0.5)
        best_preds = None

        for dt, et in product(dropout_thresholds, enrolled_thresholds):
            y_pred_custom = []
            for probs in y_probs:
                if probs[0] >= dt:
                    y_pred_custom.append(0)
                elif probs[1] >= et:
                    y_pred_custom.append(1)
                else:
                    y_pred_custom.append(2)
            macro_recall = recall_score(y_test, y_pred_custom, average='macro')
            if macro_recall > best_score:
                best_score = macro_recall
                best_thresholds = (dt, et)
                best_preds = y_pred_custom

        st.success(f"Best Macro Recall: {best_score:.4f} with thresholds {best_thresholds}")

        # Save model
        with open('model_rf.pkl', 'wb') as f:
            pickle.dump(best_rf, f)
        with open('thresholds.json', 'w') as f:
            json.dump({'dropout_threshold': best_thresholds[0], 'enrolled_threshold': best_thresholds[1]}, f)

        st.session_state.model = best_rf
        st.session_state.thresholds = {'dropout_threshold': best_thresholds[0], 'enrolled_threshold': best_thresholds[1]}

# Step 3 - Evaluate Model
elif menu == "Evaluate Model":
    if st.session_state.model is None:
        st.warning("No model found. Train model first!")
    else:
        model = st.session_state.model
        data = st.session_state.cleaned_data
        X = data.drop('status', axis=1)
        y = data['status'].map({'Dropout':0, 'Enrolled':1, 'Graduate':2})

        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        y_probs = model.predict_proba(X_test)

        thresholds = st.session_state.thresholds
        y_pred = []
        for probs in y_probs:
            if probs[0] >= thresholds['dropout_threshold']:
                y_pred.append(0)
            elif probs[1] >= thresholds['enrolled_threshold']:
                y_pred.append(1)
            else:
                y_pred.append(2)

        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred, target_names=['Dropout', 'Enrolled', 'Graduate']))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Dropout', 'Enrolled', 'Graduate'], yticklabels=['Dropout', 'Enrolled', 'Graduate'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)

# Step 4 - Predict New Data
elif menu == "Predict New Data":
    if st.session_state.model is None:
        st.warning("No model found. Train model first!")
    else:
        uploaded_new = st.file_uploader("Upload new data for prediction", type=["csv"], key="newdata")
        if uploaded_new is not None:
            new_data = pd.read_csv(uploaded_new)
            st.dataframe(new_data.head())

            model = st.session_state.model
            thresholds = st.session_state.thresholds

            preds = []
            probs = model.predict_proba(new_data)
            for prob in probs:
                if prob[0] >= thresholds['dropout_threshold']:
                    preds.append('Dropout')
                elif prob[1] >= thresholds['enrolled_threshold']:
                    preds.append('Enrolled')
                else:
                    preds.append('Graduate')

            new_data['Predicted_Status'] = preds
            st.write("Predictions:")
            st.dataframe(new_data)

            csv = new_data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
