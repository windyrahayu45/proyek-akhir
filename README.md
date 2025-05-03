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
- Dataset yang digunakan berasal dari repositori Dicoding Academy dan dapat diakses melalui tautan berikut:  
[Dicoding dataset](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv)

Dataset ini berisi informasi tentang performa siswa yang mencakup berbagai fitur seperti skor matematika, membaca, menulis, gender, status makan siang, dan latar belakang pendidikan orang tua.

**Setup environment:**
- Versi Python
  Python **3.10+**

- Instalasi Library
Jalankan perintah berikut untuk menginstal semua dependensi:
```bash
pip install -r requirements.txt

- 3. jalankan aplikasi streamlit
streamlit run app/main.py

- 4. membuka notebook
jupyter notebook Notebook.ipynb



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


**Link prototype:**
[Streamlit App Predict Dropout](https://proyek-akhir-windi.streamlit.app/)

**Langkah penggunaan:**
1. Buka link aplikasi.
2. Upload data mahasiswa baru atau gunakan data sampel.
3. atau masukan data mahasiswa sesuai dengan form yang telah disediakan


##  Conclusion
Berdasarkan hasil analisis data dan pemodelan machine learning terhadap data mahasiswa Jaya Jaya Institut, ditemukan bahwa kemungkinan mahasiswa mengalami dropout sangat dipengaruhi oleh performa akademik, kepatuhan terhadap administrasi keuangan, serta latar belakang pendidikan sebelumnya.

Berikut adalah **10 fitur terpenting** yang paling memengaruhi risiko dropout:
1. **curricular_units_2nd_sem_approved** – Jumlah mata kuliah semester 2 yang disetujui
2. **curricular_units_1st_sem_approved** – Jumlah mata kuliah semester 1 yang disetujui
3. **tuition_fees_up_to_date** – Status pembayaran biaya kuliah
4. **curricular_units_2nd_sem_grade** – Nilai rata-rata semester 2
5. **curricular_units_1st_sem_grade** – Nilai rata-rata semester 1
6. **admission_grade** – Nilai saat masuk perguruan tinggi
7. **previous_qualification_grade** – Nilai pendidikan sebelumnya
8. **course** – Program studi yang diambil
9. **curricular_units_2nd_sem_evaluations** – Jumlah evaluasi pada semester 2
10. **gdp** – Indikator kondisi ekonomi (kemungkinan berasal dari data eksternal)

Secara umum, mahasiswa yang:
- Memiliki performa rendah dalam perkuliahan (nilai dan mata kuliah tidak disetujui),
- Menunggak biaya kuliah,
- Berasal dari latar belakang akademik lemah,
- Dan berasal dari program studi tertentu,
memiliki risiko lebih tinggi untuk mengalami dropout.

##  Rekomendasi Action Items

**1. Bangun sistem pemantauan risiko dropout berbasis machine learning**  
Gunakan model prediktif yang memanfaatkan fitur penting seperti nilai akademik, status pembayaran, dan latar belakang pendidikan untuk memantau risiko dropout sejak semester awal.

**2. Sediakan program remedial untuk mahasiswa dengan performa rendah**  
Tawarkan dukungan akademik berupa kelas tambahan atau tutor untuk mahasiswa yang gagal dalam banyak mata kuliah atau memiliki nilai rendah di semester awal.

**3. Evaluasi sistem keuangan dan fasilitasi bantuan biaya kuliah**  
Tinjau kembali kebijakan pembayaran kuliah, termasuk opsi cicilan dan beasiswa bagi mahasiswa yang kesulitan finansial, mengingat keterlambatan pembayaran menjadi indikator dropout.

**4. Tinjau kurikulum dan tingkat kesulitan program studi tertentu**  
Analisis lebih lanjut terhadap program studi yang menyumbang angka dropout tertinggi untuk mengidentifikasi potensi beban akademik yang tidak seimbang.

**5. Gunakan evaluasi semester sebagai indikator intervensi cepat**  
Perhatikan jumlah evaluasi dan hasil akademik mahasiswa secara berkelanjutan, terutama pada semester kedua, untuk mengambil tindakan preventif lebih awal.
