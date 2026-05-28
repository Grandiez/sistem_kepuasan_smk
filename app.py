import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import streamlit.components.v1 as components
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from supabase import create_client, Client

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sistem Penilaian Kepuasan SMK", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# KONEKSI DATABASE SUPABASE
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
# CUSTOM CSS: LIQUID GLASS & ELEGANT UI
# ==========================================
glass_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* 1. Background Dinamis & Reset Font */
.stApp {
    background-image: url('https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=2564&auto=format&fit=crop'); 
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-color: rgba(10, 10, 15, 0.7); 
    background-blend-mode: overlay;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* 2. WADAH UTAMA: KACA TEBAL HALUS */
[data-testid="stForm"], 
[data-testid="metric-container"], 
[data-testid="stExpander"],
[data-testid="stSidebar"],
[data-testid="stFileUploadDropzone"] {
    background: rgba(18, 18, 24, 0.45) !important; 
    backdrop-filter: blur(28px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(180%) !important;
    border-radius: 20px !important; 
    border-top: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-left: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.02) !important;
    border-bottom: 1px solid rgba(0, 0, 0, 0.6) !important;
    box-shadow: 
        0 30px 60px rgba(0, 0, 0, 0.5), 
        inset 0 2px 4px rgba(255, 255, 255, 0.05), 
        inset 0 -4px 8px rgba(0, 0, 0, 0.5) !important; 
    transition: transform 0.4s cubic-bezier(0.25, 1, 0.5, 1), box-shadow 0.4s ease !important;
}

[data-testid="stForm"]:hover, 
[data-testid="stExpander"]:hover {
    transform: translateY(-2px);
    box-shadow: 
        0 40px 80px rgba(0, 0, 0, 0.6), 
        inset 0 2px 4px rgba(255, 255, 255, 0.1), 
        inset 0 -4px 8px rgba(0, 0, 0, 0.6) !important;
}

h1, h2, h3, h4, p, label, span, li, div[data-testid="stMarkdownContainer"] {
    color: rgba(255, 255, 255, 0.9) !important;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.9) !important;
    letter-spacing: 0.2px;
}

/* 3. SLIDER & TOGGLE ALA KUBE.IO (LIQUID GLASS) */
div[data-baseweb="slider"] > div > div {
    background: rgba(5, 5, 10, 0.6) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(0, 0, 0, 0.8) !important;
    box-shadow: inset 0 3px 8px rgba(0, 0, 0, 0.9), inset 0 -1px 2px rgba(255, 255, 255, 0.1) !important; 
    height: 14px !important;
}

div[data-baseweb="slider"] > div > div > div {
    background: linear-gradient(90deg, rgba(59, 130, 246, 0.6), rgba(96, 165, 250, 0.9)) !important;
    box-shadow: inset 0 2px 4px rgba(255, 255, 255, 0.4) !important;
    border-radius: 12px !important;
}

div[data-baseweb="slider"] div[role="slider"] {
    height: 32px !important; width: 56px !important; 
    background: rgba(255, 255, 255, 0.08) !important; 
    backdrop-filter: blur(16px) !important; -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.4) !important; border-radius: 16px !important;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.6), inset 0 4px 8px rgba(255, 255, 255, 0.7), inset 0 -4px 8px rgba(0, 0, 0, 0.5) !important; 
    transition: transform 0.1s cubic-bezier(0.25, 1, 0.5, 1), box-shadow 0.1s ease !important; cursor: grab !important;
}

div[data-baseweb="slider"] div[role="slider"]:active {
    transform: scale(0.92) !important; cursor: grabbing !important;
    background: rgba(255, 255, 255, 0.15) !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.6), inset 0 2px 4px rgba(255, 255, 255, 0.5), inset 0 -2px 4px rgba(0, 0, 0, 0.4) !important;
}

[data-testid="stCheckbox"] > label > div[data-baseweb="checkbox"] > div { background: rgba(5, 5, 10, 0.6) !important; box-shadow: inset 0 3px 8px rgba(0, 0, 0, 0.9), inset 0 -1px 2px rgba(255, 255, 255, 0.1) !important; }
[data-testid="stCheckbox"] > label > div[data-baseweb="checkbox"] > div > div { background: rgba(255, 255, 255, 0.8) !important; box-shadow: 0 4px 8px rgba(0,0,0,0.5), inset 0 2px 4px rgba(255,255,255,0.9), inset 0 -2px 4px rgba(0,0,0,0.3) !important; }

/* 4. TINTED DARK GLASS */
.glass-alert {
    padding: 24px; border-radius: 16px; margin-bottom: 1.5rem;
    backdrop-filter: blur(25px) saturate(180%); -webkit-backdrop-filter: blur(25px) saturate(180%);
    color: rgba(255, 255, 255, 0.95); transition: transform 0.3s ease, box-shadow 0.3s ease;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4), inset 0 -3px 6px rgba(0, 0, 0, 0.6); border-bottom: 1px solid rgba(0, 0, 0, 0.5);
}
.glass-green { background: rgba(16, 185, 129, 0.12); border-top: 1px solid rgba(16, 185, 129, 0.3); border-left: 1px solid rgba(16, 185, 129, 0.1); }
.glass-blue { background: rgba(59, 130, 246, 0.12); border-top: 1px solid rgba(59, 130, 246, 0.3); border-left: 1px solid rgba(59, 130, 246, 0.1); }
.glass-orange { background: rgba(245, 158, 11, 0.12); border-top: 1px solid rgba(245, 158, 11, 0.3); border-left: 1px solid rgba(245, 158, 11, 0.1); }
.glass-red { background: rgba(239, 68, 68, 0.12); border-top: 1px solid rgba(239, 68, 68, 0.3); border-left: 1px solid rgba(239, 68, 68, 0.1); }

/* Tombol Form */
div.stButton > button, div.stDownloadButton > button {
    background: rgba(255, 255, 255, 0.05) !important; backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important; border-radius: 12px !important;
    color: white !important; font-weight: 500 !important; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important; transition: all 0.3s ease !important;
}
div.stButton > button:hover, div.stDownloadButton > button:hover { background: rgba(255, 255, 255, 0.1) !important; border-color: rgba(255, 255, 255, 0.3) !important; }
[data-testid="baseButton-formSubmit"] { background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(96, 165, 250, 0.2) 100%) !important; border: 1px solid rgba(96, 165, 250, 0.4) !important; border-radius: 12px !important; }
[data-testid="baseButton-formSubmit"]:hover { border-color: rgba(255, 255, 255, 0.4) !important; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4) !important; }

/* FIX FULLSCREEN & MOBILE */
footer { display: none !important; visibility: hidden !important; }
.block-container { padding-top: 2rem !important; padding-bottom: 1rem !important; padding-left: 1rem !important; padding-right: 1rem !important; }
.stApp { min-height: -webkit-fill-available !important; }

/* ==========================================
   5. PRINT MODE (HTML TO PDF OPTIMIZATION)
   ========================================== */
@media print {
    html, body, .stApp, div { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
    
    /* SEMBUNYIKAN ELEMEN YANG TIDAK PERLU DI-PRINT */
    [data-testid="stSidebar"], 
    header[data-testid="stHeader"], 
    footer,
    .no-print, /* Kelas custom untuk area tombol dan header download */
    [data-testid="stDownloadButton"], /* Tombol Download CSV bawaan Streamlit */
    iframe[title="streamlit_components.v1.components.html"] { /* Iframe tombol JS */
        display: none !important;
        height: 0 !important;
    }

    .stApp { background-image: none !important; background-color: #0b0b10 !important; }
    
    /* Cegah grafik terpotong separuh saat ganti halaman PDF */
    .js-plotly-plot, .plotly, [data-testid="stPlotlyChart"] {
        page-break-inside: avoid !important;
        break-inside: avoid !important;
    }
}
</style>
"""
st.markdown(glass_css, unsafe_allow_html=True)

# --- MENU NAVIGASI SIDEBAR ---
st.sidebar.title("Navigasi Sistem")
menu = st.sidebar.selectbox("Pilih Halaman", ["Isi Kuesioner (Siswa)", "Dashboard Analisis (Admin)"])
st.sidebar.markdown("---")

# ==========================================
# HALAMAN 1: KUESIONER SISWA
# ==========================================
if menu == "Isi Kuesioner (Siswa)":
    
    if 'sudah_login' not in st.session_state:
        st.session_state['sudah_login'] = False
    if 'data_diri' not in st.session_state:
        st.session_state['data_diri'] = {}

    if not st.session_state['sudah_login']:
        st.title("Registrasi Responden")
        st.markdown("Silakan lengkapi data diri Anda sebelum mengisi kuesioner evaluasi.")

        with st.form(key='login_form'):
            st.subheader("Data Diri Siswa")
            nama_siswa = st.text_input("Nama Lengkap")
            
            col1, col2 = st.columns(2)
            with col1:
                kelas = st.selectbox("Kelas", ["10", "11", "12"])
            with col2:
                jurusan = st.selectbox("Jurusan", ["TKJT", "Busana", "DKV", "TO", "ATU"])
            
            jenis_kelamin = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"], horizontal=True)
            submit_login = st.form_submit_button(label='Mulai Kuesioner')

        if submit_login:
            if nama_siswa.strip() == "":
                st.error("Nama lengkap wajib diisi.")
            else:
                st.session_state['data_diri'] = {
                    "Nama": nama_siswa, "Kelas": kelas,
                    "Jurusan": jurusan, "Jenis_Kelamin": jenis_kelamin
                }
                st.session_state['sudah_login'] = True
                st.rerun()

    else:
        st.title(f"Halo, {st.session_state['data_diri']['Nama']}")
        st.markdown("Silakan isi kuesioner layanan secara objektif. Skala 1 (Sangat Tidak Puas) hingga 5 (Sangat Puas).")

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
            p9 = st.slider("Variasi metode pembelajaran yang digunakan (P9)", 1, 5, 3)
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
                    st.success("Terima kasih. Data evaluasi berhasil disimpan.")
                except Exception as e:
                    st.error(f"Gagal menyimpan ke database: {e}")
            else:
                st.warning("Koneksi Supabase belum diatur. Data tidak tersimpan permanen.")

        if st.button("Isi Kuesioner Baru"):
            st.session_state['sudah_login'] = False
            st.session_state['data_diri'] = {}
            st.rerun()

# ==========================================
# HALAMAN 2: DASHBOARD ADMIN 
# ==========================================
elif menu == "Dashboard Analisis (Admin)":
    
    if 'admin_logged_in' not in st.session_state:
        st.session_state['admin_logged_in'] = False

    if not st.session_state['admin_logged_in']:
        st.title("Akses Terbatas Manajemen")
        st.markdown("Silakan masukkan Username dan Password untuk mengakses Dashboard.")

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
                    st.error("Username atau Password salah.")
        
        st.stop() 

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout Admin", use_container_width=True):
        st.session_state['admin_logged_in'] = False
        st.rerun()
    st.sidebar.markdown("---")

    st.title("Analisis Kepuasan Siswa")
    st.markdown("Sistem Analisis Kepuasan Siswa Berbasis Machine Learning (PCA & K-Means Clustering).")

    st.sidebar.header("1. Sumber Data")
    sumber_data = st.sidebar.radio("Pilih metode pengambilan data:", ["Tarik dari Database (Otomatis)", "Upload Manual (Excel/CSV)"])

    df = None 

    if sumber_data == "Tarik dari Database (Otomatis)":
        if st.sidebar.button("Tarik Data Kuesioner Sekarang", use_container_width=True):
            with st.spinner("Mengambil data dari server..."):
                if supabase:
                    response = supabase.table("kuesioner").select("*").execute()
                    df_raw = pd.DataFrame(response.data)
                    if not df_raw.empty:
                        st.session_state['df_raw'] = df_raw
                        st.sidebar.success(f"Berhasil menarik {len(df_raw)} data.")
                    else:
                        st.sidebar.warning("Database masih kosong.")
                else:
                    st.sidebar.error("Koneksi Database Supabase Belum Diatur.")
        
        if 'df_raw' in st.session_state and not st.session_state['df_raw'].empty:
            df = st.session_state['df_raw'].copy()
            st.info(f"Menggunakan {len(df)} data kuesioner dari database realtime.")
        else:
            st.info("Silakan klik 'Tarik Data Kuesioner Sekarang' di sidebar.")

    else:
        uploaded_file = st.sidebar.file_uploader("Unggah file Excel/CSV", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                    
                nama_kolom_baru = ['Timestamp', 'Nama', 'Kelas', 'Jurusan', 'Jenis_Kelamin'] + [f'P{i}' for i in range(1, 21)]
                
                if len(df.columns) >= 25: 
                    df = df.iloc[:, :25]
                    df.columns = nama_kolom_baru
                else:
                    st.error("Format file tidak sesuai.")
                    df = None
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan saat membaca file: {e}")
                df = None

    if df is not None:
        kolom_nilai = [f'P{i}' for i in range(1, 21)]
        jml_awal = len(df)
        df = df.dropna(subset=kolom_nilai).reset_index(drop=True)
        jml_akhir = len(df)
        
        if jml_awal != jml_akhir:
            st.sidebar.warning(f"Ditemukan {jml_awal - jml_akhir} data tidak lengkap. Diabaikan.")

        data_numeric = df[kolom_nilai]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_numeric)

        st.sidebar.header("2. Mode Konfigurasi PCA")
        mode_pca = st.sidebar.radio(
            "Pilih Target Reduksi Dimensi:",
            ["Mode Visualisasi 3D (3 Komponen)", "Mode Validasi Akademis (Target >70%)"]
        )

        if mode_pca == "Mode Visualisasi 3D (3 Komponen)":
            pca = PCA(n_components=3)
            pca_data = pca.fit_transform(scaled_data)
            variansi_terjelaskan = sum(pca.explained_variance_ratio_) * 100
            
            df['PC1'] = pca_data[:, 0]
            df['PC2'] = pca_data[:, 1]
            df['PC3'] = pca_data[:, 2]
            
            st.sidebar.info(f"Mode Visual Aktif.\nTotal Variansi: {variansi_terjelaskan:.2f}%")
            tampilkan_3d = True 
            
        else:
            pca = PCA(n_components=0.72) 
            pca_data = pca.fit_transform(scaled_data)
            variansi_terjelaskan = sum(pca.explained_variance_ratio_) * 100
            jumlah_dimensi_baru = pca.n_components_
            
            df['PC1'] = pca_data[:, 0]
            
            if jumlah_dimensi_baru > 1:
                df['PC2'] = pca_data[:, 1]
            else:
                df['PC2'] = 0 
            
            st.sidebar.success(f"Mode Akademis Aktif.\nDimensi: {jumlah_dimensi_baru}\nVariansi: {variansi_terjelaskan:.2f}%")
            st.sidebar.caption("Grafik 3D dinonaktifkan.")
            tampilkan_3d = False 

        st.sidebar.header("3. Konfigurasi Klastering")
        gunakan_pca = st.sidebar.toggle("Aktifkan Reduksi PCA", value=True)
        data_untuk_kmeans = pca_data if gunakan_pca else scaled_data 

        n_clusters = st.sidebar.slider("Jumlah Klaster (K)", 2, 5, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(data_untuk_kmeans)

        if n_clusters >= 2:
            skor_siluet = silhouette_score(data_untuk_kmeans, df['Cluster'])
            st.sidebar.markdown("---")
            st.sidebar.subheader("Validasi K-Means")
            st.sidebar.metric(label=f"Silhouette Score (K={n_clusters})", value=f"{skor_siluet:.3f}")
            
            if skor_siluet >= 0.5:
                st.sidebar.success("Struktur klaster sangat baik.")
            elif skor_siluet >= 0.25:
                st.sidebar.info("Struktur klaster cukup baik.")
            else:
                st.sidebar.warning("Batas antar klaster kurang tegas.")

        st.sidebar.header("4. Filter Data")
        list_jurusan = df['Jurusan'].unique().tolist()
        pilih_jurusan = st.sidebar.multiselect("Jurusan:", list_jurusan, default=list_jurusan)
        list_kelas = sorted(df['Kelas'].unique().tolist()) 
        pilih_kelas = st.sidebar.multiselect("Kelas:", list_kelas, default=list_kelas)
        list_jk = df['Jenis_Kelamin'].unique().tolist()
        pilih_jk = st.sidebar.multiselect("Jenis Kelamin:", list_jk, default=list_jk)

        df_filtered = df[
            df['Jurusan'].isin(pilih_jurusan) & 
            df['Kelas'].isin(pilih_kelas) &
            df['Jenis_Kelamin'].isin(pilih_jk)
        ].copy()

        if df_filtered.empty:
            st.warning("Data tidak ditemukan dengan kombinasi filter tersebut.")
        else:
            df_filtered['Fasilitas'] = df_filtered[[f'P{i}' for i in range(1, 6)]].mean(axis=1)
            df_filtered['Kurikulum'] = df_filtered[[f'P{i}' for i in range(6, 11)]].mean(axis=1)
            df_filtered['Guru'] = df_filtered[[f'P{i}' for i in range(11, 16)]].mean(axis=1)
            df_filtered['Lingkungan'] = df_filtered[[f'P{i}' for i in range(16, 21)]].mean(axis=1)

            profile = df_filtered.groupby('Cluster')[['Fasilitas', 'Kurikulum', 'Guru', 'Lingkungan']].mean()
            counts = df_filtered['Cluster'].value_counts().sort_index()

            kamus_masalah = {
                'P1': 'Fasilitas pendukung dirasa kurang lengkap', 'P2': 'Peralatan belajar banyak rusak',
                'P3': 'Area sekolah dirasa kurang aman', 'P4': 'Lokasi fasilitas sulit dijangkau',
                'P5': 'Fasilitas belum maksimal membantu belajar', 'P6': 'Materi kurang relevan dengan industri',
                'P7': 'Siswa kesulitan memahami materi', 'P8': 'Kurikulum belum sesuai minat',
                'P9': 'Metode pembelajaran membosankan', 'P10': 'Materi dirasa kurang bermanfaat',
                'P11': 'Guru kurang menguasai materi', 'P12': 'Penjelasan guru kurang jelas',
                'P13': 'Guru kurang memberi teladan', 'P14': 'Guru sulit dihubungi saat kesulitan',
                'P15': 'Guru sering terlambat mengajar', 'P16': 'Kebersihan lingkungan kurang terjaga',
                'P17': 'Lingkungan sekolah rawan gangguan', 'P18': 'Suasana kelas kurang kondusif',
                'P19': 'Hubungan sosial kurang harmonis', 'P20': 'Budaya sopan santun belum maksimal'
            }

            kamus_solusi = {
                'P1': 'Inventarisasi lab dan ajukan pengadaan alat.', 'P2': 'Maintenance rutin peralatan.',
                'P3': 'Tingkatkan keamanan sekolah.', 'P4': 'Perbaiki layout fasilitas publik.',
                'P5': 'Evaluasi penggunaan lab.', 'P6': 'Undang praktisi dan sinkronisasi kurikulum.',
                'P7': 'Sederhanakan modul pembelajaran.', 'P8': 'Perkuat program konseling.',
                'P9': 'Pelatihan guru untuk metode interaktif.', 'P10': 'Seminar prospek karir.',
                'P11': 'Sertakan guru dalam magang industri.', 'P12': 'Kumpulkan feedback metode mengajar.',
                'P13': 'Tegakkan kode etik pengajar.', 'P14': 'Sediakan jam konsultasi siswa.',
                'P15': 'Terapkan presensi ketat bagi pengajar.', 'P16': 'Galakkan program kebersihan.',
                'P17': 'Tindak tegas pelanggaran ketertiban.', 'P18': 'Perbaiki fasilitas kelas.',
                'P19': 'Adakan kegiatan kebersamaan.', 'P20': 'Kampanyekan budaya 5S.'
            }

            dimensi_map = {
                'Fasilitas': [f'P{i}' for i in range(1, 6)], 'Kurikulum': [f'P{i}' for i in range(6, 11)],
                'Guru': [f'P{i}' for i in range(11, 16)], 'Lingkungan': [f'P{i}' for i in range(16, 21)]
            }
            
            tab1, tab2 = st.tabs(["Dashboard Analisis", "Laporan Eksekutif"])

            with tab1:
                st.subheader("Statistik Responden")
                cols = st.columns(len(df_filtered['Cluster'].unique()))
                for i, (cls, count) in enumerate(counts.items()):
                    cols[i].metric(f"Klaster {cls}", f"{count} Siswa")

                if tampilkan_3d:
                    fig_3d = px.scatter_3d(
                        df_filtered, x='PC1', y='PC2', z='PC3', color=df_filtered['Cluster'].astype(str),
                        hover_data=['Nama', 'Kelas', 'Jurusan'], title=f"Visualisasi 3D ({len(df_filtered)} Siswa)",
                        labels={'color': 'Klaster'}
                    )
                    # Modifikasi Kamera agar otomatis ter-zoom out dan utuh di PDF
                    fig_3d.update_layout(
                        scene_camera=dict(
                            up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=1.5, y=1.5, z=1.2)
                        ),
                        margin=dict(l=0, r=0, b=0, t=40),
                        paper_bgcolor='rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)', 
                        height=600, 
                        font_color='white'
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.info(f"Algoritma beroperasi di ruang {pca.n_components_} dimensi. Ditampilkan sebagai Proyeksi 2D Utama.")
                    
                    st.markdown("**Proyeksi 2D (Komponen Utama 1 vs Komponen Utama 2)**")
                    fig_2d = px.scatter(
                        df_filtered, x='PC1', y='PC2', color=df_filtered['Cluster'].astype(str),
                        hover_data=['Nama', 'Kelas', 'Jurusan'], 
                        labels={'color': 'Klaster'},
                        title="Sebaran Klaster (PC1 vs PC2)"
                    )
                    fig_2d.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
                    fig_2d.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                    st.plotly_chart(fig_2d, use_container_width=True)

                st.subheader("Perbandingan Skor Rata-rata per Klaster")
                profile_reset = profile.reset_index()
                profile_reset['Label_Klaster'] = profile_reset['Cluster'].apply(lambda x: f"Klaster {x}<br>({counts[x]} Siswa)")
                
                fig_bar = px.bar(
                    profile_reset, x='Label_Klaster', y=['Fasilitas', 'Kurikulum', 'Guru', 'Lingkungan'],
                    barmode='group', labels={'value': 'Skor (1-5)', 'variable': 'Dimensi', 'Label_Klaster': 'Kelompok'},
                    color_discrete_sequence=['#3b82f6', '#f59e0b', '#10b981', '#ef4444']
                )
                
                fig_bar.update_traces(marker_cornerradius=12, marker_line_width=0, base=0.05)
                fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_bar, use_container_width=True)

                st.subheader("Bedah Investigasi per Klaster")
                
                # Fitur Toggle Baru untuk membuka Data Target saat PDF
                tampilkan_nama = st.toggle("📋 Tampilkan Daftar Nama Target Investigasi secara Terbuka (Untuk Laporan PDF)", value=False)
                
                for cluster in profile.index:
                    dim_otomatis = profile.loc[cluster].idxmin()
                    
                    dim_pilihan = st.selectbox(
                        f"Pilih Dimensi untuk dibedah pada Klaster {cluster}:",
                        options=['Fasilitas', 'Kurikulum', 'Guru', 'Lingkungan'],
                        index=['Fasilitas', 'Kurikulum', 'Guru', 'Lingkungan'].index(dim_otomatis),
                        key=f"select_dim_{cluster}" 
                    )
                    
                    skor_dimensi = profile.loc[cluster][dim_pilihan]
                    cols_dimensi = dimensi_map[dim_pilihan]
                    df_cluster = df_filtered[df_filtered['Cluster'] == cluster]
                    
                    rata_item = df_cluster[cols_dimensi].mean()
                    item_code = rata_item.idxmin()
                    skor_item = rata_item.min()
                    
                    isi_masalah = kamus_masalah[item_code] 
                    solusi_masalah = kamus_solusi[item_code] 

                    kelas_terbanyak = int(df_cluster['Kelas'].mode()[0]) if not df_cluster.empty else "N/A"
                    jml_kelas = len(df_cluster[df_cluster['Kelas'] == kelas_terbanyak])
                    jurusan_terbanyak = df_cluster['Jurusan'].mode()[0] if not df_cluster.empty else "N/A"
                    jml_jurusan = len(df_cluster[df_cluster['Jurusan'] == jurusan_terbanyak])

                    if skor_item >= 4.0:
                        status, glass_color = "Sangat Memuaskan", "glass-green"
                    elif skor_item >= 3.0:
                        status, glass_color = "Cukup Memuaskan", "glass-blue"
                    elif skor_item >= 2.0:
                        status, glass_color = "Perlu Perbaikan", "glass-orange"
                    else:
                        status, glass_color = "Kritis", "glass-red"

                    if skor_item >= 4.0:
                        pesan_html = f"""
                        <b>Klaster {cluster} (Dimensi: {dim_pilihan} - Status: {status})</b><br><br>
                        <b>Kondisi Aman:</b> Mayoritas siswa di klaster ini merasa puas dengan <b>{dim_pilihan}</b>. Aspek terendah ada pada poin <i>'{isi_masalah}'</i> namun skor masih aman di angka <b>{skor_item:.2f}/5.00</b>.<br><br>
                        <b>Saran:</b> {solusi_masalah}
                        """
                        st.markdown(f'<div class="glass-alert {glass_color}">{pesan_html}</div>', unsafe_allow_html=True)
                        with st.expander(f"Daftar Mayoritas Siswa (Klaster {cluster})"):
                            st.write(f"Tidak memerlukan investigasi mendalam untuk masalah {dim_pilihan}.")
                    else:
                        pesan_html = f"""
                        <b>Klaster {cluster} (Dimensi: {dim_pilihan} - Status: {status})</b><br><br>
                        <b>Akar Masalah:</b> <i>'{isi_masalah}'</i> (Skor: <b>{skor_item:.2f}/5.00</b>).<br><br>
                        <b>Rekomendasi Solusi:</b> {solusi_masalah}<br><br>
                        <b>Target Investigasi:</b> Fokus pada <b>Kelas {kelas_terbanyak}</b> ({jml_kelas} anak) dan <b>Jurusan {jurusan_terbanyak}</b> ({jml_jurusan} anak).
                        """
                        st.markdown(f'<div class="glass-alert {glass_color}">{pesan_html}</div>', unsafe_allow_html=True)
                        
                        df_kelas = df_cluster[df_cluster['Kelas'] == kelas_terbanyak][['Nama', 'Kelas', 'Jurusan', 'Jenis_Kelamin']].reset_index(drop=True)
                        df_kelas.index += 1 
                        df_jurusan = df_cluster[df_cluster['Jurusan'] == jurusan_terbanyak][['Nama', 'Kelas', 'Jurusan', 'Jenis_Kelamin']].reset_index(drop=True)
                        df_jurusan.index += 1 
                        
                        # Logika Toggle untuk Daftar Target
                        if tampilkan_nama:
                            st.markdown(f"**Target Kelas {kelas_terbanyak}:**")
                            st.dataframe(df_kelas, use_container_width=True)
                            st.markdown("---")
                            st.markdown(f"**Target Jurusan {jurusan_terbanyak}:**")
                            st.dataframe(df_jurusan, use_container_width=True)
                        else:
                            with st.expander(f"Daftar Nama Target Investigasi (Klaster {cluster} - Kasus: {dim_pilihan})"):
                                st.markdown(f"**Target Kelas {kelas_terbanyak}:**")
                                st.dataframe(df_kelas, use_container_width=True)
                                st.markdown("---")
                                st.markdown(f"**Target Jurusan {jurusan_terbanyak}:**")
                                st.dataframe(df_jurusan, use_container_width=True)

                # ==========================================
                # AREA MENU CETAK (Disembunyikan saat di-Print)
                # ==========================================
                st.markdown("""
                <div class="no-print">
                    <hr>
                    <h3 style="color:white; margin-bottom:10px;">📥 Menu Cetak & Download</h3>
                    <p style="background: rgba(59, 130, 246, 0.15); padding: 12px; border-radius: 8px; border-left: 4px solid #3b82f6;">
                        💡 <b>Panduan Cetak PDF:</b> Klik tombol Cetak di bawah. Pada jendela *Print* browser, pastikan kamu mencentang opsi <b>'Background graphics' (Grafik Latar Belakang)</b> agar efek kaca tetap terbawa.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # HTML Button Print
                html_btn_1 = """
                <style>
                .print-btn {
                    background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(96, 165, 250, 0.2) 100%);
                    border: 1px solid rgba(96, 165, 250, 0.4); border-radius: 12px; color: white;
                    padding: 12px 20px; font-size: 15px; font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
                    cursor: pointer; transition: all 0.3s ease; width: 100%; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2); font-weight: 500;
                }
                .print-btn:hover { border-color: rgba(255, 255, 255, 0.6); box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4); transform: translateY(-2px); }
                </style>
                <button class="print-btn" onclick="window.parent.print()">🖨️ Cetak Dashboard / Save as PDF</button>
                """
                components.html(html_btn_1, height=60)

                st.markdown('<div class="no-print"><br><h4 style="color:white; margin-bottom:10px;">💾 Download Database Mentah</h4></div>', unsafe_allow_html=True)
                csv = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button("Download Database Lengkap (.csv)", csv, "Database_Kepuasan.csv", "text/csv")

            with tab2:
                st.header("Executive Summary")
                st.markdown(f"Laporan merangkum hasil analisis kepuasan dari **{len(df_filtered)} responden**.")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Radar Chart Kinerja Global")
                    rata_global = {
                        'Fasilitas': df_filtered['Fasilitas'].mean(),
                        'Kurikulum': df_filtered['Kurikulum'].mean(),
                        'Guru': df_filtered['Guru'].mean(),
                        'Lingkungan': df_filtered['Lingkungan'].mean()
                    }
                    df_radar = pd.DataFrame(dict(Skor=list(rata_global.values()), Dimensi=list(rata_global.keys())))
                    fig_radar = px.line_polar(df_radar, r='Skor', theta='Dimensi', line_close=True, range_r=[0,5], markers=True)
                    
                    fig_radar.update_traces(fill='toself', line_color='cyan', fillcolor='rgba(0, 242, 254, 0.2)')
                    fig_radar.update_layout(
                        height=400, margin=dict(l=40, r=40, t=40, b=40),
                        polar=dict(
                            bgcolor='rgba(15, 15, 20, 0.6)',
                            radialaxis=dict(visible=True, gridcolor='rgba(255,255,255,0.1)', color='rgba(255,255,255,0.4)'),
                            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='rgba(255,255,255,0.8)')
                        ),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white'
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                with col2:
                    klaster_terburuk = profile.mean(axis=1).idxmin()
                    df_kritis = df_filtered[df_filtered['Cluster'] == klaster_terburuk]
                    st.subheader(f"Proporsi Jurusan Kritis (Klaster {klaster_terburuk})")
                    
                    dark_colors = ['#1e3a8a', '#5b21b6', '#0f766e', '#991b1b', '#9a3412']
                    
                    fig_donut = px.pie(df_kritis, names='Jurusan', hole=0.5, color_discrete_sequence=dark_colors)
                    fig_donut.update_traces(marker=dict(line=dict(color='rgba(18, 18, 24, 1)', width=3)))
                    fig_donut.update_layout(
                        height=400, margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white'
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)

                st.divider()

                st.subheader("Rapor Evaluasi 4 Dimensi Layanan")
                rata_global_series = pd.Series(rata_global).sort_values() 

                for dimensi, skor_dimensi in rata_global_series.items():
                    cols_dimensi = dimensi_map[dimensi]
                    rata_item_dimensi = df_filtered[cols_dimensi].mean()
                    item_terendah = rata_item_dimensi.idxmin()
                    skor_item_terendah = rata_item_dimensi.min()

                    isi_masalah, solusi_masalah = kamus_masalah[item_terendah], kamus_solusi[item_terendah]

                    if skor_dimensi >= 4.0:
                        status, glass_color = "Sangat Memuaskan", "glass-green"
                        pesan_html = f"<b>Dimensi {dimensi} (Skor: {skor_dimensi:.2f}/5.00) - {status}</b><br><br><b>Saran Preventif:</b> Perhatikan poin <i>'{isi_masalah}'</i> (Skor: {skor_item_terendah:.2f}). <b>Tindakan:</b> {solusi_masalah}"
                    else:
                        if skor_dimensi >= 3.0: 
                            status, glass_color = "Cukup Memuaskan", "glass-blue"
                        elif skor_dimensi >= 2.0: 
                            status, glass_color = "Perlu Perbaikan", "glass-orange"
                        else: 
                            status, glass_color = "Kritis", "glass-red"
                            
                        pesan_html = f"<b>Dimensi {dimensi} (Skor: {skor_dimensi:.2f}/5.00) - {status}</b><br><br><b>Titik Terlemah:</b> <i>'{isi_masalah}'</i> (Skor: <b>{skor_item_terendah:.2f}</b>).<br><br><b>Tindakan:</b> {solusi_masalah}"

                    st.markdown(f'<div class="glass-alert {glass_color}">{pesan_html}</div>', unsafe_allow_html=True)

                st.markdown("""
                <div class="no-print">
                    <hr>
                    <h3 style="color:white; margin-bottom:10px;">📥 Cetak Laporan Eksekutif (PDF)</h3>
                    <p style="background: rgba(16, 185, 129, 0.15); padding: 12px; border-radius: 8px; border-left: 4px solid #10b981;">
                        💡 Klik tombol di bawah ini untuk mencetak dan menyimpan halaman Laporan Eksekutif ini menjadi PDF.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                html_btn_2 = """
                <style>
                .print-btn {
                    background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(52, 211, 153, 0.2) 100%);
                    border: 1px solid rgba(52, 211, 153, 0.4); border-radius: 12px; color: white;
                    padding: 12px 20px; font-size: 15px; font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
                    cursor: pointer; transition: all 0.3s ease; width: 100%; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2); font-weight: 500;
                }
                .print-btn:hover { border-color: rgba(255, 255, 255, 0.6); box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4); transform: translateY(-2px); }
                </style>
                <button class="print-btn" onclick="window.parent.print()">📄 Simpan Halaman Eksekutif ke PDF</button>
                """
                components.html(html_btn_2, height=60)
