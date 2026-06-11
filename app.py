import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# 1. Pengaturan Halaman dan Konfigurasi UI
st.set_page_config(
    page_title="Mall Customer Cluster Analyzer",
    page_icon="🛍️",
    layout="wide"
)

theme_base = st.get_option("theme.base")

if theme_base == "dark":
    border_color = "white"
else:
    border_color = "black"

# Judul Utama Aplikasi
st.title("🛍️ AI Mall Customer Cluster Analyzer App")
st.markdown(
    """
    <p style='font-size:22px;'>
    Aplikasi berbasis Machine Learning untuk mengelompokkan karakteristik pelanggan mall secara otomatis.
    </p>
    """,
    unsafe_allow_html=True
)


# 2. Fungsi Load Model & Dataset (Bagian Backend)
@st.cache_resource
def load_cached_models():
    # Load Component Mode 2D (2 Fitur) | (Path langsung dari root folder)
    with open('models/scaler_2d.pkl', 'rb') as f:
        scaler_2d = pickle.load(f)
    
    with open('models/kmeans_2d.pkl', 'rb') as f:
        kmeans_2d = pickle.load(f)

    # Load Component Mode 3D (3 Fitur)
    with open('models/scaler_3d.pkl', 'rb') as f:
        scaler_3d = pickle.load(f)

    with open('models/kmeans_3d.pkl', 'rb') as f:
        kmeans_3d = pickle.load(f)

    return scaler_2d, kmeans_2d, scaler_3d, kmeans_3d

@st.cache_data
def load_dataset():
    # Load data aseli untuk keperluan visualisasi background scatter plot
    return pd.read_csv('dataset/Mall_Customers.csv')

# Eksekusi Load data dan model
scaler_2d, kmeans_2d, scaler_3d, kmeans_3d = load_cached_models()
df = load_dataset()

# 3. Konfigurasi Label Pemaknaan Bisnis Kelompok (Dictionary)
labels_2d_desc = {
    1: "Moderat (Standard)",
    2: "Sultan Boros (Target Utama)",
    3: "Impulsif",
    4: "Bijak (Rich & Frugal)",
    5: "Ekonomis (Hemat)"
}

labels_3d_desc = {
    1: "Senior Moderat",
    2: "Sultan Dewasa Standar",
    3: "Sultan Dewasa Bijak (Hemat)",
    4: "Sultan Boros Premium (Semua Usia)",
    5: "Anak Muda Impulsif",
    6: "Customer Ekonomis"
}

# 4. Advanced Adaptive CSS Injector (Theme + Laptop + Smartphone)
st.markdown(
    """
    <style>
    /* Centering the tabs container */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 24px;
        border-bottom: 2px solid rgba(128, 128, 128, 0.2);
    }
    
    /* 1. Dynamic Tab Font Contrast & Wrap Fix */
    .stTabs [data-baseweb="tab"] p {
        color: var(--text-color) !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        transition: color 0.3s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover p {
        color: var(--text-color) !important;
        opacity: 0.8;
    }
    .stTabs [aria-selected="true"] {
        border-bottom-color: #1f77b4 !important;
    }
    /* Slider visual alignment */
    .stSlider {
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    /* Tablet & Small Laptop Breakpoint (992px and below) */
    @media (max-width: 992px) {
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 1.5rem !important;
        }
        div[data-testid="column"] {
            width: 100% !important;
        }
    }
    
    /* 2. Global Smartphone Layout Overrides (width <= 767px) */
    @media (max-width: 767px) {
        /* Keep Tabs Side-by-Side (Forced Row Layout with Inner Text Wrap) */
        div[data-baseweb="tab-list"] {
            flex-wrap: nowrap !important;
            display: flex !important;
            width: 100% !important;
            gap: 6px !important;
        }
        div[data-baseweb="tab"] {
            width: 50% !important;
        }
        div[data-baseweb="tab"] p {
            white-space: normal !important;
            text-align: center !important;
            line-height: 1.2 !important;
        }
        
        /* Collapse side-by-side columns to 100% stacked rows */
        div[data-testid="stHorizontalBlock"] {
            display: block !important;
        }
        div[data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
            flex-basis: 100% !important;
            margin-bottom: 2rem !important;
        }
        
        /* Prevent Plotly charts from squeezing into unreadable thin lines */
        div[data-testid="stPlotlyChart"] {
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch;
        }
        div[data-testid="stPlotlyChart"] > div {
            min-width: 600px !important; /* Forces a readable canvas width on phone screens */
        }
        
        /* Clean scaling for mobile header typography */
        h1 { font-size: 1.5rem !important; }
        .stSubheader h2 { font-size: 1.2rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 4. Inisialisasi Centered Tab Switcher
tab1, tab2 = st.tabs([
    "Mode 2 Fitur (Income vs Spending)",
    "Mode 3 Fitur (Age vs Income vs Spending)"
])

# ==================== TAB 1: MODE 2 FITUR ====================
with tab1:
    st.subheader("📊 Analisis Cluster Ruang 2D (K = 5)")
    
    # Membagi layout: 70% Kolom Kiri (Visualisasi), 30% Kolom Kanan (Kontrol / Input)
    col1, col2 = st.columns([7, 3])
    
    # 1. Tampilkan Input Panel di Kolom Kanan (Dievaluasi terlebih dahulu agar variable terdefinisi)
    with col2:
        st.markdown("### 📥 Input Data Customer Baru")
        # Menggunakan container dengan border aesthetic
        with st.container(border=True):
            income_input = st.slider("Annual Income (k$)", 15, 140, 50, key="income_2d")
            spending_input = st.slider("Spending Score (1-100)", 1, 100, 50, key="spending_2d")
            
    # 2. Proses Visualisasi dan Prediksi di Kolom Kiri
    with col1:
        # --- Proses Prediksi AI (100% Sesuai Logika Asli) ---
        new_data = np.array([[income_input, spending_input]])
        data_scaled = scaler_2d.transform(new_data)
        cluster_prediction = kmeans_2d.predict(data_scaled)[0] + 1
        
        # Tampilkan Banner Hasil Prediksi di bagian atas kolom visualisasi
        st.markdown(
            f"""
            <div style="
                background-color:#1f77b4;
                padding:18px;
                border-radius:12px;
                font-size:20px;
                font-weight:bold;
                color:white;
                margin-bottom:20px;
                box-shadow: 0 4px 15px rgba(31, 119, 180, 0.3);
            ">
            🔮 Hasil Prediksi AI: Customer Baru Masuk ke 
            Cluster {cluster_prediction} — {labels_2d_desc[cluster_prediction]}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # --- Pembuatan Visualisasi Grafik 2D (Plotly) ---
        df['Cluster_2D'] = (
            kmeans_2d.predict(
                scaler_2d.transform(df[['Annual Income (k$)', 'Spending Score (1-100)']])
            ) + 1
        )
        
        df['Cluster_2D_Label'] = df['Cluster_2D'].apply(
            lambda x: f"Cluster {x} — {labels_2d_desc[x]}"
        )
        
        fig = px.scatter(
            df, x='Annual Income (k$)', y='Spending Score (1-100)', color='Cluster_2D_Label',
            size_max=14,
            labels={'Cluster_2D_Label': 'Segmentasi Customer Mall'},
            category_orders={
                'Cluster_2D_Label': [
                    f"Cluster {i} — {labels_2d_desc[i]}" for i in range(1, 6)
                ]
            },
            title="Sebaran Segmen Pasar Ruang 2D",
            color_discrete_sequence=px.colors.qualitative.Set1,
            hover_data=['CustomerID']
        )
        
        # Menambahkan tanda bintang kuning besar untuk posisi customer baru yang di-input
        fig.update_traces(marker=dict(size=8))
        fig.add_trace(go.Scatter(
            x=[income_input], y=[spending_input],
            mode='markers', name='Data Baru',
            marker=dict(color='yellow', size=18, symbol='star', line=dict(color='#444444', width=2))
        ))
        
        fig.update_layout(
            legend=dict(
                font=dict(size=17),
                itemsizing='constant'
            ),
            legend_title=dict(
                font=dict(size=20)
            ),
            title={
                'x': 0.29,
                'xanchor': 'left'
            },
            title_font_size=26,
            height=600,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 2: MODE 3 FITUR ====================
with tab2:
    st.subheader("📊 Analisis Cluster Ruang 3D (K = 6)")
    
    # Membagi layout: 70% Kolom Kiri (Visualisasi), 30% Kolom Kanan (Kontrol / Input)
    col1_3d, col2_3d = st.columns([7, 3])
    
    # 1. Tampilkan Input Panel di Kolom Kanan (Dievaluasi terlebih dahulu)
    with col2_3d:
        st.markdown("### 📥 Input Data Customer Baru")
        # Menggunakan container dengan border aesthetic
        with st.container(border=True):
            age_input = st.slider("Age", 18, 70, 30, key="age_3d")
            income_input_3d = st.slider("Annual Income (k$)", 15, 140, 50, key="income_3d")
            spending_input_3d = st.slider("Spending Score (1-100)", 1, 100, 50, key="spending_3d")
            
    # 2. Proses Visualisasi dan Prediksi di Kolom Kiri
    with col1_3d:
        # --- Proses Prediksi AI (100% Sesuai Logika Asli) ---
        new_data_3d = np.array([[age_input, income_input_3d, spending_input_3d]])
        data_scaled_3d = scaler_3d.transform(new_data_3d)
        cluster_prediction_3d = kmeans_3d.predict(data_scaled_3d)[0] + 1
        
        # Tampilkan Banner Hasil Prediksi di bagian atas kolom visualisasi
        st.markdown(
            f"""
            <div style="
                background-color:#2ca02c;
                padding:18px;
                border-radius:12px;
                font-size:20px;
                font-weight:bold;
                color:white;
                margin-bottom:20px;
                box-shadow: 0 4px 15px rgba(44, 160, 44, 0.3);
            ">
            🔮 Hasil Prediksi AI: Customer Baru Masuk ke 
            Cluster {cluster_prediction_3d} — {labels_3d_desc[cluster_prediction_3d]}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # --- Pembuatan Visualisasi Grafik 3D (Plotly) ---
        df['Cluster_3D'] = (
            kmeans_3d.predict(
                scaler_3d.transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
            ) + 1
        )
        
        df['Cluster_3D_Label'] = df['Cluster_3D'].apply(
            lambda x: f"Cluster {x} — {labels_3d_desc[x]}"
        )
        
        fig_3d = px.scatter_3d(
            df, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)', color='Cluster_3D_Label',
            labels={'Cluster_3D_Label': 'Segmentasi Customer Mall'},
            category_orders={
                'Cluster_3D_Label': [
                    f"Cluster {i} — {labels_3d_desc[i]}" for i in range(1, 7)
                ]
            },
            title="Peta Komparasi Segmen Pasar Ruang 3D (Seret untuk Memutar)",
            color_discrete_sequence=px.colors.qualitative.Set1,
            hover_data=['CustomerID']
        )
        
        # Tambahkan penanda posisi data baru berbentuk diamond melayang di koordinat 3D
        fig_3d.update_traces(marker=dict(size=7))
        fig_3d.add_trace(go.Scatter3d(
            x=[age_input], y=[income_input_3d], z=[spending_input_3d],
            mode='markers', name='Data Baru',
            marker=dict(color='yellow', size=12, symbol='diamond', line=dict(color='#444444', width=3))
        ))
        
        fig_3d.update_layout(
            scene=dict(
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.5)
                )
            ),
            legend=dict(
                font=dict(size=17),
                itemsizing='constant'
            ),
            legend_title=dict(
                font=dict(size=20)
            ),
            title={
                'x': 0.13,
                'xanchor': 'left'
            },
            title_font_size=26,
            height=800,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)