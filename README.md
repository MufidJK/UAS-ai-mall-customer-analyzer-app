---
title: Mall Customer Analyzer
emoji: 🛍️
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
---

# 🛍️ AI Mall Customer Cluster Analyzer App

Aplikasi berbasis **Machine Learning (Unsupervised Learning)** yang dirancang untuk membantu manajemen mall dalam memetakan karakteristik perilaku pelanggan secara otomatis. Dengan segmentasi yang akurat, mall dapat mengambil keputusan strategis yang tepat sasaran, seperti personalisasi promosi, efisiensi anggaran marketing, hingga perencanaan event bertarget.

Aplikasi web interaktif ini dibangun menggunakan **Streamlit** untuk _frontend_ dan didukung oleh grafik interaktif **Plotly** agar visualisasi klastering dapat dieksplorasi secara dinamis dalam ruang 3D (bisa diputar, digeser, dan di-zoom).

---

## 🚀 Fitur Utama (Dual-Mode Analysis)

Aplikasi ini menyediakan dua mode analisis klastering terpisah yang disesuaikan dengan kebutuhan strategi bisnis mall:

1. **Mode 2 Fitur (Macro Targeting):**
   - Menggunakan kombinasi fitur **Annual Income** dan **Spending Score** (K = 5).
   - Cocok digunakan untuk strategi marketing massal/universal tanpa memandang batasan usia pelanggan.
2. **Mode 3 Fitur (Hyper-Targeting):**
   - Menggunakan kombinasi fitur **Age**, **Annual Income**, dan **Spending Score** (K = 6).
   - Cocok digunakan untuk strategi promosi produk spesifik yang sangat sensitif terhadap tren pengelompokan generasi usia.

---

## 🛠️ Tech Stack yang Digunakan

- **Language:** Python
- **Web Framework:** Streamlit
- **Machine Learning Engine:** Scikit-Learn
- **Data Manipulation:** Pandas & NumPy
- **Interactive Visualization:** Plotly Express & Plotly Graph Objects

---

## 📁 Struktur Projek

Susunan direktori dan file di dalam repositori projek ini diatur dengan struktur baku sebagai berikut:

```text
MALL-CUSTOMER-CLUSTER-ANALYZER-APP/
├── dataset/
│   └── Mall_Customers.csv                 # Dataset mentah Mall Customers dari Kaggle
├── laporan/
│   └── Laporan_UAS_Mufid_Refaya_38250013.pdf # Dokumen Laporan Resmi format PDF
├── models/
│   ├── kmeans_2d.pkl                      # Model K-Means hasil training 2 Fitur (K=5)
│   ├── kmeans_3d.pkl                      # Model K-Means hasil training 3 Fitur (K=6)
│   ├── scaler_2d.pkl                      # Scaler MinMaxScaler untuk data 2 Fitur
│   └── scaler_3d.pkl                      # Scaler MinMaxScaler untuk data 3 Fitur
├── notebook/
│   └── Project_UAS_Mufid_38250013.ipynb   # Jupyter Notebook proses training & analisis data
├── app.py                                 # Script utama aplikasi UI Streamlit & Backend
├── README.md                              # Dokumentasi panduan projek (File ini)
└── requirements.txt                       # Daftar dependensi library Python yang dibutuhkan
```

---

## 💻 Cara Menjalankan Aplikasi Secara Lokal

1. Persiapan Direktori
   Pastikan semua struktur folder dan file di atas sudah berada di dalam satu direktori kerja yang sama di komputer Anda.

2. Instalasi Dependensi Library
   Buka Terminal atau Command Prompt pada direktori projek, lalu jalankan perintah berikut untuk menginstal semua library pendukung yang terdaftar di requirements.txt:

   pip install -r requirements.txt

3. Jalankan Aplikasi Streamlit
   Eksekusi perintah di bawah ini untuk menyalakan server lokal Streamlit:

   streamlit run app.py

   Setelah perintah dijalankan, aplikasi web otomatis akan terbuka pada peramban (browser) Anda di alamat lokal: http://localhost:8501.

---

## 📊 Pemaknaan Segmen Pasar (Business Interpretation)

### Mode Analisis 2D Fitur (K = 5)

- Cluster 1: Moderat (Standard) — Karakteristik pendapatan dan pengeluaran serba di tengah-tengah.
- Cluster 2: Sultan Boros (Target Utama) — Pendapatan tinggi dan sangat konsumtif. Target utama profit mall.
- Cluster 3: Impulsif — Pendapatan cenderung rendah namun gaya belanja sangat tinggi.
- Cluster 4: Bijak (Rich & Frugal) — Pendapatan tinggi namun sangat menahan diri dalam berbelanja.
- Cluster 5: Ekonomis (Hemat) — Pendapatan rendah dengan pengeluaran yang sangat hemat.

### Mode Analisis 3D Fitur (K = 6)

- Cluster 1: Senior Moderat — Pengunjung usia matang dengan pendapatan menengah dan belanja stabil seperlunya.
- Cluster 2: Sultan Dewasa Standar — Kelompok mapan usia matang, kaya raya, namun belanja masih terukur.
- Cluster 3: Sultan Dewasa Bijak (Hemat) — Kelompok kaya raya usia matang yang menerapkan gaya hidup hemat (frugal living).
- Cluster 4: Sultan Boros Premium (Semua Usia) — Kelompok terkaya di mall yang sangat loyal dan boros tanpa memandang batasan usia.
- Cluster 5: Anak Muda Impulsif — Kelompok usia muda dengan pendapatan pas-pasan namun gaya belanja sangat konsumtif.
- Cluster 6: Customer Ekonomis — Kelompok pendapatan minim dengan pola pengeluaran paling irit.

---

## 🌐 Deployment

Aplikasi ini sepenuhnya kompatibel dan siap di-deploy secara online menggunakan platform Hugging Face Spaces berbasis SDK Streamlit. Setiap kali ada perubahan kode yang di-push ke repositori Space, platform akan otomatis melakukan rebuild dan memperbarui sistem secara berkala.
