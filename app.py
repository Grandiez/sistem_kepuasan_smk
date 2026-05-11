import streamlit as st
import pandas as pd
import plotly.express as px
import os
import datetime
from fpdf import FPDF
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from supabase import create_client, Client

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sistem Penilaian Kepuasan SMK", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 🔌 KONEKSI DATABASE SUPABASE
# ==========================================
@st.cache_resource
def init_connection():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception as e:
        return None

supabase = init_connection()

# ==========================================
# 💎 CUSTOM CSS: LIQUID GLASS & HIGH CONTRAST TEXT
# ==========================================
glass_css = """
<style>
/* 1. Background Animasi / Gradient Premium */
.stApp {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    background-attachment: fixed;
    color: white !important;
}

/* 🌟 PENINGKATAN KETERBACAAN TEKS (HIGH CONTRAST SHADOW) 🌟 */
p, span, label, li, div[data-testid="stMarkdownContainer"] {
    text-shadow: 0px 1px 4px rgba(0, 0, 0, 0.9) !important;
    letter-spacing: 0.2px;
}
h1, h2, h3, h4, h5, h6 {
    text-shadow: 0px 3px 8px rgba(0, 0, 0, 0.9), 0px 0px 15px rgba(255, 255, 255, 0.2) !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px;
}
[data-testid="stMetricValue"] {
    text-shadow: 0px 2px 10px rgba(0, 0, 0, 0.8), 0px 0px 20px rgba(255, 255, 255, 0.4) !important;
}

/* 2. Sidebar Efek Kaca Volumetrik */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(20px) saturate(120%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(120%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: inset -1px 0 0 rgba(255, 255, 255, 0.1);
}

/* 3. Komponen Wadah (Metrics, Expander, Alerts, Form) */
[data-testid="metric-container"], 
[data-testid="stExpander"], 
div[data-testid="stAlert"],
[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(20px) saturate(120%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(120%) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 20px !important;
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.2), 
        inset 0 1px 0 rgba(255, 255, 255, 0.5), 
        inset 0 -1px 0 rgba(255, 255, 255, 0.1), 
        inset 0 0 20px 5px rgba(255, 255, 255, 0.05) !important;
    overflow: hidden;
}

/* 4. PENYATUAN SEMUA TOMBOL (REGULER, DOWNLOAD, & BROWSE FILE) */
div.stButton > button,
div.stDownloadButton > button,
[data-testid="stFileUploadDropzone"] button,
[data-testid="baseButton-formSubmit"] {
    background: rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(15px) saturate(150%) !important;
    -webkit-backdrop-filter: blur(15px) saturate(150%) !important;
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
    color: white !important;
    border-radius: 50px !important; 
    transition: all 0.4s ease !important;
    box-shadow: 
        0 4px 16px rgba(0, 0, 0, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.6),
        inset 0 -1px 0 rgba(255, 255, 255, 0.2) !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
}

/* Tombol Submit Form Khusus Biru Glow */
[data-testid="baseButton-formSubmit"] {
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
    border: none !important;
}

/* Efek Menyala (Glow/Bloom) Seragam saat Tombol di-Hover */
div.stButton > button:hover,
div.stDownloadButton > button:hover,
[data-testid="stFileUploadDropzone"] button:hover,
[data-testid="baseButton-formSubmit"]:hover {
    background: rgba(255, 255, 255, 0.25) !important;
    transform: translateY(-3px) scale(1.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.8) !important;
    box-shadow: 
        0 0 25px rgba(255, 255, 255, 0.4), 
        0 8px 24px rgba(0, 0, 0, 0.3), 
        inset 0 0 20px rgba(255, 255, 255, 0.6), 
        inset 0 2px 0 rgba(255, 255, 255, 0.9) !important; 
    text-shadow: 0px 0px 10px rgba(255, 255, 255, 0.9) !important;
}

/* Efek saat Tombol diklik (Active - Menekan ke dalam) */
div.stButton > button:active,
div.stDownloadButton > button:active,
[data-testid="stFileUploadDropzone"] button:active,
[data-testid="baseButton-formSubmit"]:active {
    transform: translateY(1px) scale(0.98) !important;
    box-shadow: 
        0 2px 8px rgba(0, 0, 0, 0.2),
        inset 0 4px 10px rgba(0, 0, 0, 0.3) !important;
}

/* 5. Area Dropzone File Uploader (Efek Sumur & Hover) */
[data-testid="stFileUploadDropzone"] {
    background: rgba(0, 0, 0, 0.3) !important; 
    backdrop-filter: blur(15px) saturate(120%) !important;
    -webkit-backdrop-filter: blur(15px) saturate(120%) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 20px !important;
    box-shadow: 
        inset 0 10px 20px rgba(0, 0, 0, 0.6), 
        inset 0 -1px 5px rgba(255, 255, 255, 0.15) !important; 
    transition: all 0.3s ease;
}
[data-testid="stFileUploadDropzone"]:hover {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.6) !important;
    box-shadow: 
        0 0 30px rgba(255, 255, 255, 0.15), 
        inset 0 0 30px rgba(255, 255, 255, 0.2), 
        inset 0 10px 20px rgba(0, 0, 0, 0.6) !important; 
}

/* 6. Kotak File Uploaded */
[data-testid="stUploadedFile"] {
    background: rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(15px) saturate(130%) !important;
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.6) !important;
}

/* 7. MultiSelect Dropdown & Text Input Hover */
[data-baseweb="select"] > div, [data-baseweb="input"] > div {
    background: rgba(0, 0, 0, 0.2) !important; 
    backdrop-filter: blur(15px) saturate(120%) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 12px !important;
    box-shadow: inset 0 4px 10px rgba(0, 0, 0, 0.4), inset 0 -1px 0 rgba(255, 255, 255, 0.2) !important;
    transition: all 0.3s ease;
    color: white !important;
}
[data-baseweb="select"] > div:hover, [data-baseweb="input"] > div:hover {
    border: 1px solid rgba(255, 255, 255, 0.6) !important;
    box-shadow: 
        0 0 15px rgba(255, 255, 255, 0.2), 
        inset 0 0 15px rgba(255, 255, 255, 0.2), 
        inset 0 4px 10px rgba(0, 0, 0, 0.4) !important;
}
span[data-baseweb="tag"] {
    background: rgba(255, 255, 255, 0.2) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
    border-radius: 8px !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6), 0 2px 4px rgba(0,0,0,0.2) !important;
    color: white !important;
}

/* 8. Slider Jumlah Klaster & Hover */
[data-testid="stSliderTickBar"] {
    background: rgba(255, 255, 255, 0.3) !important;
    border-radius: 10px !important;
}
div[data-baseweb="slider"] div[role="slider"] {
    background: rgba(255, 255, 255, 0.9) !important;
    border: 2px solid rgba(255, 255, 255, 1) !important;
    box-shadow: 0 0 12px rgba(255, 255, 255, 0.6), inset 0 0 5px rgba(0,0,0,0.3) !important;
    transition: all 0.2s ease;
}
div[data-baseweb="slider"] div[role="slider"]:hover {
    background: #ffffff !important;
    transform: scale(1.2);
    box-shadow: 
        0 0 20px rgba(255, 255, 255, 1), 
        inset 0 0 5px rgba(0,0,0,0.2) !important;
}

/* 9. Penyesuaian Tabel Dataframe agar agak gelap dan kontras */
[data-testid="stDataFrame"] {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 12px;
    padding: 5px;
}
</style>
"""
st.markdown(glass_css, unsafe_allow_html=True)
# ==========================================

# --- MENU NAVIGASI SIDEBAR ---
st.sidebar.title("Navigasi Sistem")
menu = st.sidebar.selectbox("Pilih Halaman", ["Isi Kuesioner (Siswa)", "Dashboard Analisis (Admin)"])
st.sidebar.markdown("---")

# ==========================================
# HALAMAN 1: KUESIONER SISWA (DENGAN LOGIN)
# ==========================================
if menu == "Isi Kuesioner (Siswa)":
    
    # Inisialisasi Session State
    if 'sudah_login' not in st.session_state:
        st.session_state['sudah_login'] = False
    if 'data_diri' not in st.session_state:
        st.session_state['data_diri'] = {}

    # --- TAMPILAN A: BELUM LOGIN ---
    if not st.session_state['sudah_login']:
        st.title("🔐 Registrasi Responden")
        st.markdown("Silakan lengkapi data diri kamu sebelum mengisi kuesioner evaluasi.")

        with st.form(key='login_form'):
            st.subheader("Data Diri Siswa")
            nama_siswa = st.text_input("Nama Lengkap")
            
            col1, col2 = st.columns(2)
            with col1:
                kelas = st.selectbox("Kelas", ["10", "11", "12"])
            with col2:
                jurusan = st.selectbox("Jurusan", ["TKJT", "Busana", "DKV", "TO", "ATU"])
            
            jenis_kelamin = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"], horizontal=True)
            submit_login = st.form_submit_button(label='Mulai Kuesioner 🚀')

        if submit_login:
            if nama_siswa.strip() == "":
                st.error("⚠️ Nama lengkap wajib diisi!")
            else:
                st.session_state['data_diri'] = {
                    "Nama": nama_siswa, "Kelas": kelas,
                    "Jurusan": jurusan, "Jenis_Kelamin": jenis_kelamin
                }
                st.session_state['sudah_login'] = True
                st.rerun()

    # --- TAMPILAN B: SUDAH LOGIN (ISI KUESIONER) ---
    else:
        st.title(f"👋 Halo, {st.session_state['data_diri']['Nama']}!")
        st.markdown("Silakan isi kuesioner layanan SMKN 1 Dawuan secara objektif. Skala **1 (Sangat Tidak Puas)** hingga **5 (Sangat Puas)**.")

        with st.form(key='kuesioner_form'):
            st.subheader("1. Fasilitas Sekolah")
            p1 = st.slider("Kelengkapan fasilitas lab/bengkel untuk praktik (P1)", 1, 5, 3)
            p2 = st.slider("Kondisi dan kelayakan peralatan belajar di kelas (P2)", 1, 5, 3)
            p3 = st.slider("Kenyamanan dan keamanan area sekolah secara umum (P3)", 1, 5, 3)
            p4 = st.slider("Akses dan kebersihan fasilitas umum (Toilet, Kantin) (P4)", 1, 5, 3)
            p5 = st.slider("Dukungan fasilitas terhadap kelancaran belajar (P5)", 1, 5, 3)
            st.divider()

            st.subheader("2. Kurikulum dan Materi Pembelajaran")
            p6 = st.slider("Kesesuaian materi dengan kebutuhan industri/dunia kerja (P6)", 1, 5, 3)
            p7 = st.slider("Tingkat kemudahan dalam memahami materi yang diajarkan (P7)", 1, 5, 3)
            p8 = st.slider("Kesesuaian kurikulum dengan minat dan bakat siswa (P8)", 1, 5, 3)
            p9 = st.slider("Variasi metode pembelajaran yang digunakan (tidak membosankan) (P9)", 1, 5, 3)
            p10 = st.slider("Manfaat materi pelajaran untuk masa depan karir (P10)", 1, 5, 3)
            st.divider()

            st.subheader("3. Kinerja dan Kompetensi Guru")
            p11 = st.slider("Penguasaan materi oleh guru saat mengajar di kelas (P11)", 1, 5, 3)
            p12 = st.slider("Kejelasan guru dalam menyampaikan penjelasan (P12)", 1, 5, 3)
            p13 = st.slider("Sikap keteladanan, kedisiplinan, dan etika guru (P13)", 1, 5, 3)
            p14 = st.slider("Kemudahan menghubungi guru saat mengalami kesulitan belajar (P14)", 1, 5, 3)
            p15 = st.slider("Ketepatan waktu guru dalam mengisi jam pelajaran (P15)", 1, 5, 3)
            st.divider()

            st.subheader("4. Lingkungan Sekolah")
            p16 = st.slider("Tingkat kebersihan lingkungan sekolah secara keseluruhan (P16)", 1, 5, 3)
            p17 = st.slider("Keamanan sekolah dari gangguan luar/ketertiban (P17)", 1, 5, 3)
            p18 = st.slider("Kondusivitas suasana di dalam kelas saat belajar (P18)", 1, 5, 3)
            p19 = st.slider("Keharmonisan hubungan antar sesama siswa dan warga sekolah (P19)", 1, 5, 3)
            p20 = st.slider("Penerapan budaya sopan santun (5S) di lingkungan sekolah (P20)", 1, 5, 3)
            
            submit_button = st.form_submit_button(label='Kirim Evaluasi')

        if submit_button:
            data_responden = {
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Nama": st.session_state['data_diri']['Nama'],
                "Kelas": st.session_state['data_diri']['Kelas'],
                "Jurusan": st.session_state['data_diri']['Jurusan'],
                "Jenis_Kelamin": st.session_state['data_diri']['Jenis_Kelamin'],
                "P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5,
                "P6": p6, "P7": p7, "P8": p8, "P9": p9, "P10": p10,
                "P11": p11, "P12": p12, "P13": p13, "P14": p14, "P15": p15,
                "P16": p16, "P17": p17, "P18": p18, "P19": p19, "P20": p20
            }
            
            if supabase:
                try:
                    supabase.table("kuesioner").insert(data_responden).execute()
                    st.success("Terima kasih! Data evaluasi berhasil disimpan ke Database Utama.")
                    st.balloons()
                except Exception as e:
                    st.error(f"Gagal menyimpan ke database: {e}")
            else:
                st.warning("Koneksi Supabase belum diatur di Streamlit Secrets! Data tidak tersimpan permanen.")

        if st.button("🔄 Isi Kuesioner Baru (Reset)"):
            st.session_state['sudah_login'] = False
            st.session_state['data_diri'] = {}
            st.rerun()

# ==========================================
# HALAMAN 2: DASHBOARD ADMIN (DENGAN LOGIN & SUPABASE FETCH)
# ==========================================
elif menu == "Dashboard Analisis (Admin)":
    
    # Inisialisasi Session State Admin
    if 'admin_logged_in' not in st.session_state:
        st.session_state['admin_logged_in'] = False

    # --- TAMPILAN A: ADMIN BELUM LOGIN ---
    if not st.session_state['admin_logged_in']:
        st.title("🔒 Akses Terbatas Manajemen")
        st.markdown("Silakan masukkan **Username** dan **Password** untuk mengakses Dashboard.")

        col_space1, col_login, col_space2 = st.columns([1, 2, 1])
        with col_login:
            with st.form(key='admin_login_form'):
                st.subheader("Login Sistem")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit_admin = st.form_submit_button(label='Masuk Dashboard')

            if submit_admin:
                if username == "admin" and password == "admin123":
                    st.session_state['admin_logged_in'] = True
                    st.rerun()
                else:
                    st.error("⚠️ Username atau Password salah!")

    # --- TAMPILAN B: ADMIN SUDAH LOGIN (DASHBOARD) ---
    else:
        st.sidebar.markdown("---")
        if st.sidebar.button("🚪 Logout Admin", use_container_width=True):
            st.session_state['admin_logged_in'] = False
            st.rerun()
        st.sidebar.markdown("---")

        st.title("📊 Dashboard Analisis Kepuasan - SMK N 1 Dawuan")
        st.markdown("Sistem Analisis Kepuasan Siswa Berbasis **Machine Learning (PCA & K-Means Clustering)**.")

        # --- MANAJEMEN SUMBER DATA ---
        st.sidebar.header("1. Sumber Data")
        sumber_data = st.sidebar.radio("Pilih metode pengambilan data:", ["Tarik dari Database (Otomatis)", "Upload Manual (Excel/CSV)"])

        df = None # Inisialisasi DataFrame kosong

        if sumber_data == "Tarik dari Database (Otomatis)":
            if st.sidebar.button("🔄 Tarik Data Kuesioner Sekarang", use_container_width=True):
                with st.spinner("Mengambil data dari server..."):
                    if supabase:
                        response = supabase.table("kuesioner").select("*").execute()
                        df_raw = pd.DataFrame(response.data)
                        if not df_raw.empty:
                            st.session_state['df_raw'] = df_raw
                            st.sidebar.success(f"Berhasil menarik {len(df_raw)} data!")
                        else:
                            st.sidebar.warning("Database masih kosong (Belum ada siswa yang mengisi).")
                    else:
                        st.sidebar.error("Koneksi Database Supabase Belum Diatur!")
            
            # Gunakan data yang sudah ditarik di session
            if 'df_raw' in st.session_state and not st.session_state['df_raw'].empty:
                df = st.session_state['df_raw'].copy()
                st.info(f"✅ Menggunakan {len(df)} data kuesioner dari database realtime.")
            else:
                st.info("Silakan klik 'Tarik Data Kuesioner Sekarang' di sidebar.")

        else: # Metode Upload Manual
            uploaded_file = st.sidebar.file_uploader("Unggah file cadangan", type=["csv", "xlsx"])
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
                    else: df = pd.read_excel(uploaded_file)
                    
                    nama_kolom_baru = ['Timestamp', 'Nama', 'Kelas', 'Jurusan', 'Jenis_Kelamin'] + [f'P{i}' for i in range(1, 21)]
                    if len(df.columns) >= 25: 
                        df = df.iloc[:, :25]
                        df.columns = nama_kolom_baru
                    else:
                        st.error("Format file tidak sesuai!")
                        df = None
                except Exception as e:
                    st.error(f"Error membaca file: {e}")
                    df = None

        # --- JIKA DATA (df) SUDAH ADA, JALANKAN PCA & K-MEANS ---
        if df is not None:
            kolom_nilai = [f'P{i}' for i in range(1, 21)]
            jml_awal = len(df)
            df = df.dropna(subset=kolom_nilai).reset_index(drop=True)
            jml_akhir = len(df)
            
            if jml_awal != jml_akhir:
                st.sidebar.warning(f"⚠️ Ditemukan {jml_awal - jml_akhir} data kosong diabaikan.")

            data_numeric = df[kolom_nilai]
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data_numeric)

            # --- MODE PCA ---
            st.sidebar.header("2. Mode PCA")
            mode_pca = st.sidebar.radio("Target Reduksi:", ["Mode Visualisasi 3D (3 Komponen)", "Mode Validasi Akademis (Target >70%)"])

            if mode_pca == "Mode Visualisasi 3D (3 Komponen)":
                pca = PCA(n_components=3)
                pca_data = pca.fit_transform(scaled_data)
                variansi_terjelaskan = sum(pca.explained_variance_ratio_) * 100
                df['PC1'] = pca_data[:, 0]
                df['PC2'] = pca_data[:, 1]
                df['PC3'] = pca_data[:, 2]
                st.sidebar.info(f"👁️ Mode Visual. Variansi: **{variansi_terjelaskan:.2f}%**")
                tampilkan_3d = True 
            else:
                pca = PCA(n_components=0.72) 
                pca_data = pca.fit_transform(scaled_data)
                variansi_terjelaskan = sum(pca.explained_variance_ratio_) * 100
                df['PC1'] = pca_data[:, 0]
                df['PC2'] = pca_data[:, 1]
                st.sidebar.success(f"🎓 Mode Akademis ({pca.n_components_} Dimensi). Variansi: **{variansi_terjelaskan:.2f}%**")
                tampilkan_3d = False 

            # --- K-MEANS ---
            st.sidebar.header("3. Konfigurasi Klaster")
            gunakan_pca = st.sidebar.toggle("🟢 Aktifkan Reduksi PCA", value=True)
            data_untuk_kmeans = pca_data if gunakan_pca else scaled_data 

            n_clusters = st.sidebar.slider("Jumlah Klaster (K)", 2, 5, 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df['Cluster'] = kmeans.fit_predict(data_untuk_kmeans)

            if n_clusters >= 2:
                skor_siluet = silhouette_score(data_untuk_kmeans, df['Cluster'])
                with st.sidebar.expander("🔬 Validasi K-Means"):
                    st.write(f"Silhouette Score (k={n_clusters}): **{skor_siluet:.3f}**")
                
            # --- FILTER ---
            st.sidebar.header("4. Filter Data")
            pilih_jurusan = st.sidebar.multiselect("Jurusan:", df['Jurusan'].unique(), default=df['Jurusan'].unique())
            pilih_kelas = st.sidebar.multiselect("Kelas:", sorted(df['Kelas'].unique()), default=sorted(df['Kelas'].unique()))
            
            df_filtered = df[df['Jurusan'].isin(pilih_jurusan) & df['Kelas'].isin(pilih_kelas)].copy()

            if df_filtered.empty:
                st.warning("⚠️ Data filter kosong.")
            else:
                df_filtered['Fasilitas'] = df_filtered[[f'P{i}' for i in range(1, 6)]].mean(axis=1)
                df_filtered['Kurikulum'] = df_filtered[[f'P{i}' for i in range(6, 11)]].mean(axis=1)
                df_filtered['Guru'] = df_filtered[[f'P{i}' for i in range(11, 16)]].mean(axis=1)
                df_filtered['Lingkungan'] = df_filtered[[f'P{i}' for i in range(16, 21)]].mean(axis=1)

                profile = df_filtered.groupby('Cluster')[['Fasilitas', 'Kurikulum', 'Guru', 'Lingkungan']].mean()
                counts = df_filtered['Cluster'].value_counts().sort_index()

                # Kamus mapping sama seperti sebelumnya
                kamus_masalah = {
                    'P1': 'Fasilitas pendukung dirasa kurang lengkap', 'P2': 'Peralatan belajar banyak rusak/kuno',
                    'P3': 'Area sekolah dirasa kurang aman/nyaman', 'P4': 'Lokasi fasilitas (kantin/toilet) sulit dijangkau',
                    'P5': 'Fasilitas belum maksimal membantu belajar', 'P6': 'Materi kurang relevan dengan industri',
                    'P7': 'Siswa kesulitan memahami materi', 'P8': 'Kurikulum belum sesuai minat dan bakat',
                    'P9': 'Metode pembelajaran membosankan', 'P10': 'Materi dirasa kurang bermanfaat bagi masa depan',
                    'P11': 'Guru kurang menguasai materi mendalam', 'P12': 'Penjelasan guru berbelit-belit',
                    'P13': 'Guru kurang memberi teladan etika/disiplin', 'P14': 'Guru sulit dihubungi saat siswa kesulitan',
                    'P15': 'Guru sering terlambat/jam kosong', 'P16': 'Kebersihan lingkungan sekolah kurang terjaga',
                    'P17': 'Lingkungan sekolah rawan gangguan', 'P18': 'Suasana kelas kurang kondusif',
                    'P19': 'Hubungan sosial warga sekolah kurang harmonis', 'P20': 'Budaya sopan santun belum maksimal'
                }

                kamus_solusi = {
                    'P1': 'Inventarisasi lab dan ajukan pengadaan.', 'P2': 'Maintenance rutin dan perbarui teknologi.',
                    'P3': 'Tingkatkan keamanan patroli sekolah.', 'P4': 'Perbaiki layout fasilitas toilet/kantin.',
                    'P5': 'Evaluasi penggunaan lab agar efektif.', 'P6': 'Undang praktisi industri sinkronisasi kurikulum.',
                    'P7': 'Adakan tutor sebaya, sederhanakan modul.', 'P8': 'Perkuat program konseling karir guru BK.',
                    'P9': 'Training metode belajar interaktif.', 'P10': 'Seminar prospek karir/kunjungan industri.',
                    'P11': 'Ikutkan guru dalam diklat/magang.', 'P12': 'Evaluasi cara mengajar & kumpulkan feedback.',
                    'P13': 'Tegakkan kode etik guru teladan.', 'P14': 'Wajibkan jam konsultasi di luar kelas.',
                    'P15': 'Terapkan presensi ketat.', 'P16': 'Galakkan program kebersihan evaluasi OB.',
                    'P17': 'Tindak tegas pelanggaran.', 'P18': 'Perbaiki fasilitas kelas & tegakkan tatib.',
                    'P19': 'Kegiatan kebersamaan lintas kelas.', 'P20': 'Kampanyekan 5S secara masif.'
                }

                dimensi_map = {
                    'Fasilitas': [f'P{i}' for i in range(1, 6)], 'Kurikulum': [f'P{i}' for i in range(6, 11)],
                    'Guru': [f'P{i}' for i in range(11, 16)], 'Lingkungan': [f'P{i}' for i in range(16, 21)]
                }

                tab1, tab2 = st.tabs(["📊 Dashboard Analisis", "📑 Laporan Eksekutif"])

                # --- TAB 1: DASHBOARD ---
                with tab1:
                    st.subheader("👥 Statistik Responden")
                    cols = st.columns(len(df_filtered['Cluster'].unique()))
                    for i, (cls, count) in enumerate(counts.items()):
                        cols[i].metric(f"Klaster {cls}", f"{count} Siswa")

                    if tampilkan_3d:
                        fig_3d = px.scatter_3d(df_filtered, x='PC1', y='PC2', z='PC3', color=df_filtered['Cluster'].astype(str), hover_data=['Nama', 'Kelas', 'Jurusan'], title=f"Visualisasi 3D")
                        fig_3d.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=600, font_color='white')
                        st.plotly_chart(fig_3d, use_container_width=True)
                    else:
                        st.markdown("**📉 Proyeksi 2D (Komponen Utama 1 vs Komponen Utama 2)**")
                        fig_2d = px.scatter(df_filtered, x='PC1', y='PC2', color=df_filtered['Cluster'].astype(str), hover_data=['Nama', 'Kelas', 'Jurusan'], title="Bayangan Klaster 2D")
                        fig_2d.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                        st.plotly_chart(fig_2d, use_container_width=True)

                    st.subheader("📈 Perbandingan Skor Rata-rata per Klaster")
                    profile_reset = profile.reset_index()
                    profile_reset['Label_Klaster'] = profile_reset['Cluster'].apply(lambda x: f"Klaster {x}<br>({counts[x]})")
                    fig_bar = px.bar(profile_reset, x='Label_Klaster', y=['Fasilitas', 'Kurikulum', 'Guru', 'Lingkungan'], barmode='group', color_discrete_sequence=['#3b82f6', '#f59e0b', '#10b981', '#ef4444'])
                    fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                    st.plotly_chart(fig_bar, use_container_width=True)

                    st.subheader("🔍 Bedah Investigasi per Klaster")
                    for cluster in profile.index:
                        dim_otomatis = profile.loc[cluster].idxmin()
                        dim_pilihan = st.selectbox(f"Pilih Dimensi Bedah (Klaster {cluster}):", options=['Fasilitas', 'Kurikulum', 'Guru', 'Lingkungan'], index=['Fasilitas', 'Kurikulum', 'Guru', 'Lingkungan'].index(dim_otomatis), key=f"select_dim_{cluster}")
                        
                        skor_dimensi = profile.loc[cluster][dim_pilihan]
                        cols_dimensi = dimensi_map[dim_pilihan]
                        df_cluster = df_filtered[df_filtered['Cluster'] == cluster]
                        
                        rata_item = df_cluster[cols_dimensi].mean()
                        item_code = rata_item.idxmin()
                        skor_item = rata_item.min()
                        
                        if skor_item >= 4.0: status, warna_alert = "Sangat Memuaskan 🌟", st.success
                        elif skor_item >= 3.0: status, warna_alert = "Cukup Memuaskan 🔵", st.info
                        elif skor_item >= 2.0: status, warna_alert = "Perlu Perbaikan 🟠", st.warning
                        else: status, warna_alert = "Kritis (Segera!) 🔴", st.error

                        warna_alert(f"**Klaster {cluster} ({dim_pilihan}) - {status}**\n\nMasalah Utama: *{kamus_masalah[item_code]}* (Skor: {skor_item:.2f}/5.00)\n\nRekomendasi: {kamus_solusi[item_code]}")

                # --- TAB 2: LAPORAN EKSEKUTIF ---
                with tab2:
                    st.header("📑 Executive Summary")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("🕸️ Radar Chart Kinerja Global")
                        rata_global = {'Fasilitas': df_filtered['Fasilitas'].mean(), 'Kurikulum': df_filtered['Kurikulum'].mean(), 'Guru': df_filtered['Guru'].mean(), 'Lingkungan': df_filtered['Lingkungan'].mean()}
                        df_radar = pd.DataFrame(dict(Skor=list(rata_global.values()), Dimensi=list(rata_global.keys())))
                        fig_radar = px.line_polar(df_radar, r='Skor', theta='Dimensi', line_close=True, range_r=[0,5], markers=True)
                        fig_radar.update_traces(fill='toself', line_color='cyan')
                        fig_radar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                        st.plotly_chart(fig_radar, use_container_width=True)

                    with col2:
                        klaster_terburuk = profile.mean(axis=1).idxmin()
                        df_kritis = df_filtered[df_filtered['Cluster'] == klaster_terburuk]
                        st.subheader(f"🍩 Jurusan Kritis (Klaster {klaster_terburuk})")
                        fig_donut = px.pie(df_kritis, names='Jurusan', hole=0.4)
                        fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                        st.plotly_chart(fig_donut, use_container_width=True)

                    st.divider()
                    st.subheader("📋 Rapor Evaluasi Layanan Global")
                    for dimensi, skor_dimensi in pd.Series(rata_global).sort_values().items():
                        rata_item_dimensi = df_filtered[dimensi_map[dimensi]].mean()
                        item_terendah = rata_item_dimensi.idxmin()
                        
                        if skor_dimensi >= 3.5: st.success(f"**{dimensi} (Skor: {skor_dimensi:.2f}) - AMAN** | Saran Perbaikan: {kamus_solusi[item_terendah]}")
                        else: st.error(f"**{dimensi} (Skor: {skor_dimensi:.2f}) - KRITIS** | Masalah Utama: {kamus_masalah[item_terendah]} | Solusi: {kamus_solusi[item_terendah]}")
