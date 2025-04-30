# Proyek Akhir: Menyelesaikan Permasalahan Jaya Jaya Institute

##  1. Business Understanding
Jaya Jaya Institut adalah sebuah lembaga pendidikan  yang berfokus pada penyediaan program pendidikan tinggi dan kursus pelatihan profesional. Tantangan utama yang dihadapi oleh institut ini adalah meningkatkan tingkat kelulusan dan mempertahankan mahasiswa agar tidak drop-out di tengah studi mereka.

### Permasalahan Bisnis
- Tingginya angka mahasiswa yang keluar (drop-out) sebelum menyelesaikan program.
- Kurangnya sistem prediksi dini untuk mengidentifikasi mahasiswa berisiko tinggi.
- Tidak adanya dashboard monitoring yang mudah digunakan untuk mengawasi dropout rate dan performa akademik mahasiswa.

### Cakupan Proyek
- Melakukan analisis data historis mahasiswa.
- Membangun model machine learning untuk memprediksi kemungkinan dropout mahasiswa.
- Membuat dashboard bisnis untuk membantu tim manajemen memantau tingkat dropout dan faktor-faktor yang berkontribusi.
- Deploy prototype sistem prediksi berbasis web menggunakan Streamlit Community Cloud.

### Persiapan
**Sumber data:**
- Dataset mahasiswa dari Jaya Jaya Institut, berisi informasi demografis, akademik, dan status kelulusan.

**Setup environment:**
- Python 3.10+
- Library utama: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, streamlit, joblib
- Deployment menggunakan Streamlit Community Cloud

## 2. Business Dashboard
Dashboard bisnis dibuat menggunakan Metabase, dengan visualisasi utama seperti:
- Tingkat kehadiran pada Semetster 1 berdasarkan status
- Distribusi Mahasiswa Dropout berdasarkan Course
- Dropout Rate per Admission Grade Bucket
- Top 10 Feature Paling Memengaruhi Dropout (didapatkan dari export feature importance dari model)
- Early Warning: Siswa Risiko Dropout (didapatkan dari export predictions dari model)
- Analisis Dropout Berdasarkan Status Debtor & Scholarship
- Data Droupout berdasarkan Age At Enrollment
- Korelasi Terhadap Dropout (Matrik Heatmap dibuat mengunakan heatmap.py berdarkan data hasil kolerasi)


### tujuan dashboard
- Mendeteksi resiko dini dropout
- Analisis faktor kritis dropout
- Memantau tren dropout


### Akses Dashboard

- **Email:** root@mail.com  
- **Password:** root123

### Ekspor Database Metabase

Untuk menjaga konfigurasi dashboard dan query yang telah dibuat:
```bash
docker cp metabase:/metabase.db/metabase.db.mv.db ./
```

---


**Link prototype:** (contoh)  
[Streamlit App Predict Dropout](https://proyek-akhir-windi.streamlit.app/)

**Langkah penggunaan:**
1. Buka link aplikasi.
2. Upload data mahasiswa baru atau gunakan data sampel.
3. atau masukan data mahasiswa sesuai dengan form yang telah disediakan


## Conclusion
Berdasarkan analisis dan modeling yang dilakukan, model Random Forest yang dibangun berhasil mencapai akurasi prediksi yang baik dalam mengidentifikasi mahasiswa yang berisiko drop-out. Dashboard bisnis yang dibuat juga memungkinkan pihak manajemen untuk melakukan monitoring dropout secara real-time.

Proyek ini memberikan solusi end-to-end mulai dari prediksi dini hingga visualisasi analitis, sehingga membantu Jaya Jaya Institut meningkatkan tingkat kelulusan dan mengurangi tingkat drop-out.

## Rekomendasi Action Items
**Action item 1:**
- Implementasi sistem monitoring rutin berbasis dashboard untuk analisa faktor risiko dropout setiap semester.

**Action item 2:**
- Meluncurkan program intervensi akademik khusus untuk mahasiswa yang masuk dalam kategori "risiko tinggi" berdasarkan hasil prediksi machine learning.