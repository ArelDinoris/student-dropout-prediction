import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.set_page_config(page_title="Prediksi Student Dropout", page_icon="🎓", layout="wide")

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'student_dropout_model.pkl')
    return joblib.load(model_path)

model = load_model()

st.title("🎓 Sistem Prediksi Student Dropout")
st.markdown("### Jaya Jaya Institut")
st.markdown("---")

st.sidebar.header("Input Data Mahasiswa")

with st.sidebar.form("input_form"):
    marital_status = st.selectbox("Status Pernikahan", [1, 2, 3, 4, 5, 6])
    gender = st.selectbox("Jenis Kelamin", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    age = st.number_input("Umur", min_value=17, max_value=70, value=20)
    course = st.selectbox("Program Studi", [33, 171, 9254, 9070, 9773, 8014, 9003, 9853, 12, 9085])
    admission_grade = st.number_input("Nilai Masuk", min_value=0.0, max_value=200.0, value=130.0)
    scholarship = st.selectbox("Beasiswa", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    tuition = st.selectbox("Biaya Terbayar", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    debtor = st.selectbox("Debtor", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    displaced = st.selectbox("Displaced", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    
    submitted = st.form_submit_button("Prediksi")

if submitted:
    input_data = pd.DataFrame({
        'Marital_status': [marital_status], 'Application_mode': [1], 'Application_order': [1],
        'Course': [course], 'Daytime_evening_attendance': [1], 'Previous_qualification': [1],
        'Previous_qualification_grade': [130.0], 'Nacionality': [1], 'Mothers_qualification': [1],
        'Fathers_qualification': [1], 'Mothers_occupation': [1], 'Fathers_occupation': [1],
        'Admission_grade': [admission_grade], 'Displaced': [displaced],
        'Educational_special_needs': [0], 'Debtor': [debtor], 'Tuition_fees_up_to_date': [tuition],
        'Gender': [gender], 'Scholarship_holder': [scholarship], 'Age_at_enrollment': [age],
        'International': [0], 'Curricular_units_1st_sem_credited': [0],
        'Curricular_units_1st_sem_enrolled': [0], 'Curricular_units_1st_sem_evaluations': [0],
        'Curricular_units_1st_sem_approved': [0], 'Curricular_units_1st_sem_grade': [0.0],
        'Curricular_units_1st_sem_without_evaluations': [0], 'Curricular_units_2nd_sem_credited': [0],
        'Curricular_units_2nd_sem_enrolled': [0], 'Curricular_units_2nd_sem_evaluations': [0],
        'Curricular_units_2nd_sem_approved': [0], 'Curricular_units_2nd_sem_grade': [0.0],
        'Curricular_units_2nd_sem_without_evaluations': [0], 'Unemployment_rate': [10.0],
        'Inflation_rate': [1.5], 'GDP': [1.5]
    })
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0]
    
    st.subheader("Hasil Prediksi")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction[0] == 1:
            st.metric("Prediksi", "DROPOUT", delta="Berisiko", delta_color="inverse")
        else:
            st.metric("Prediksi", "GRADUATE", delta="Aman", delta_color="normal")
    with col2:
        st.metric("Probabilitas Dropout", f"{probability[1]:.1%}")
    with col3:
        st.metric("Probabilitas Graduate", f"{probability[0]:.1%}")
    
    st.progress(float(probability[1]))
    
    if prediction[0] == 1:
        st.warning("Mahasiswa berisiko DROPOUT. Berikan bimbingan khusus!")
    else:
        st.success("Mahasiswa diprediksi GRADUATE.")

st.markdown("---")
st.markdown("Jaya Jaya Institut - Sistem Prediksi Dropout")
