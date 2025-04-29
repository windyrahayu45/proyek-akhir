# Proyek Akhir: Menyelesaikan Permasalahan Jaya Jaya Institute

## Business Understanding
Jaya Jaya Institut adalah sebuah lembaga pendidikan (edutech) yang berfokus pada penyediaan program pendidikan tinggi dan kursus pelatihan profesional. Tantangan utama yang dihadapi oleh institut ini adalah meningkatkan tingkat kelulusan dan mempertahankan mahasiswa agar tidak drop-out di tengah studi mereka.

## Permasalahan Bisnis
- Tingginya angka mahasiswa yang keluar (drop-out) sebelum menyelesaikan program.
- Kurangnya sistem prediksi dini untuk mengidentifikasi mahasiswa berisiko tinggi.
- Tidak adanya dashboard monitoring yang mudah digunakan untuk mengawasi attrition rate dan performa akademik mahasiswa.

## Cakupan Proyek
- Melakukan analisis data historis mahasiswa.
- Membangun model machine learning untuk memprediksi kemungkinan attrition mahasiswa.
- Membuat dashboard bisnis untuk membantu tim manajemen memantau tingkat attrition dan faktor-faktor yang berkontribusi.
- Deploy prototype sistem prediksi berbasis web menggunakan Streamlit Community Cloud.

## Persiapan
**Sumber data:**
- Dataset mahasiswa dari Jaya Jaya Institut, berisi informasi demografis, akademik, dan status kelulusan.

**Setup environment:**
- Python 3.10+
- Library utama: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, streamlit, joblib
- Deployment menggunakan Streamlit Community Cloud

## Business Dashboard
Dashboard bisnis dibuat menggunakan Metabase, dengan visualisasi utama seperti:
- Tingkat kelulusan vs drop-out
- Rata-rata nilai akademik
- Analisis faktor-faktor utama penyebab attrition

**Link dashboard:** (contoh)  
[Metabase Dashboard Jaya Jaya Institut](https://metabase.jayajayainstitute.com/dashboard-attrition) *(simulasi link)*

## Menjalankan Sistem Machine Learning
Sistem machine learning dapat dijalankan dengan mengakses aplikasi Streamlit yang telah di-deploy.

**Langkah penggunaan:**
1. Buka link aplikasi.
2. Upload data mahasiswa baru atau gunakan data sampel.
3. Sistem akan memberikan prediksi apakah mahasiswa tersebut berisiko tinggi untuk attrition.

**Link prototype:** (contoh)  
[Streamlit App Predict Attrition](https://jaya-jaya-institute.streamlit.app)

## Conclusion
Berdasarkan analisis dan modeling yang dilakukan, model Random Forest yang dibangun berhasil mencapai akurasi prediksi yang baik dalam mengidentifikasi mahasiswa yang berisiko drop-out. Dashboard bisnis yang dibuat juga memungkinkan pihak manajemen untuk melakukan monitoring attrition secara real-time.

Proyek ini memberikan solusi end-to-end mulai dari prediksi dini hingga visualisasi analitis, sehingga membantu Jaya Jaya Institut meningkatkan tingkat kelulusan dan mengurangi tingkat drop-out.

## Rekomendasi Action Items
**Action item 1:**
- Implementasi sistem monitoring rutin berbasis dashboard untuk analisa faktor risiko attrition setiap semester.

**Action item 2:**
- Meluncurkan program intervensi akademik khusus untuk mahasiswa yang masuk dalam kategori "risiko tinggi" berdasarkan hasil prediksi machine learning.

