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
# KONSTANTA MODULE-LEVEL (DIBUAT SEKALI)
# ==========================================
KAMUS = {
    'P1': {'masalah': 'Fasilitas pendukung dirasa kurang lengkap', 'solusi': 'Inventarisasi lab dan ajukan pengadaan alat.'},
    'P2': {'masalah': 'Peralatan belajar banyak rusak', 'solusi': 'Maintenance rutin peralatan.'},
    'P3': {'masalah': 'Area sekolah dirasa kurang aman', 'solusi': 'Tingkatkan keamanan sekolah.'},
    'P4': {'masalah': 'Lokasi fasilitas sulit dijangkau', 'solusi': 'Perbaiki layout fasilitas publik.'},
    'P5': {'masalah': 'Fasilitas belum maksimal membantu belajar', 'solusi': 'Evaluasi penggunaan lab.'},
    'P6': {'masalah': 'Materi kurang relevan dengan industri', 'solusi': 'Undang praktisi dan sinkronisasi kurikulum.'},
    'P7': {'masalah': 'Siswa kesulitan memahami materi', 'solusi': 'Sederhanakan modul pembelajaran.'},
    'P8': {'masalah': 'Kurikulum belum sesuai minat', 'solusi': 'Perkuat program konseling.'},
    'P9': {'masalah': 'Metode pembelajaran membosankan', 'solusi': 'Pelatihan guru untuk metode interaktif.'},
    'P10': {'masalah': 'Materi dirasa kurang bermanfaat', 'solusi': 'Seminar prospek karir.'},
    'P11': {'masalah': 'Guru kurang menguasai materi', 'solusi': 'Sertakan guru dalam magang industri.'},
    'P12': {'masalah': 'Penjelasan guru kurang jelas', 'solusi': 'Kumpulkan feedback metode mengajar.'},
    'P13': {'masalah': 'Guru kurang memberi teladan', 'solusi': 'Tegakkan kode etik pengajar.'},
    'P14': {'masalah': 'Guru sulit dihubungi saat kesulitan', 'solusi': 'Sediakan jam konsultasi siswa.'},
    'P15': {'masalah': 'Guru sering terlambat mengajar', 'solusi': 'Terapkan presensi ketat bagi pengajar.'},
    'P16': {'masalah': 'Kebersihan lingkungan kurang terjaga', 'solusi': 'Galakkan program kebersihan.'},
    'P17': {'masalah': 'Lingkungan sekolah rawan gangguan', 'solusi': 'Tindak tegas pelanggaran ketertiban.'},
    'P18': {'masalah': 'Suasana kelas kurang kondusif', 'solusi': 'Perbaiki fasilitas kelas.'},
    'P19': {'masalah': 'Hubungan sosial kurang harmonis', 'solusi': 'Adakan kegiatan kebersamaan.'},
    'P20': {'masalah': 'Budaya sopan santun belum maksimal', 'solusi': 'Kampanyekan budaya 5S.'}
}

DIMENSI_MAP = {
    'Fasilitas': [f'P{i}' for i in range(1, 6)], 
    'Kurikulum': [f'P{i}' for i in range(6, 11)],
    'Guru': [f'P{i}' for i in range(11, 16)], 
    'Lingkungan': [f'P{i}' for i in range(16, 21)]
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_status_color(skor):
    """Mengembalikan status dan kelas CSS warna berdasarkan skor rata-rata."""
    if skor >= 4.0: return "Sangat Memuaskan", "glass-green"
    elif skor >= 3.0: return "Cukup Memuaskan", "glass-blue"
    elif skor >= 2.0: return "Perlu Perbaikan", "glass-orange"
    else: return "Kritis", "glass-red"

def cetak_pdf_button(label, bg_gradient, border_color, shadow_color):
    """Komponen HTML terpadu untuk tombol cetak bergaya liquid glass."""
    html = f"""
    <style>
    .print-btn {{
        background: {bg_gradient};
        border: 1px solid {border_color};
        border-top: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 12px;
        color: white;
        padding: 12px 20px;
        font-size: 15px;
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.25, 1, 0.5, 1);
        width: 100%;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3), inset 0 2px 4px rgba(255,255,255,0.15);
        font-weight: 500;
    }}
    .print-btn:hover {{
        border-color: rgba(255, 255, 255, 0.6);
        box-shadow: 0 10px 25px {shadow_color}, inset 0 2px 4px rgba(255,255,255,0.3);
        transform: translateY(-2px);
    }}
    </style>
    <button class="print-btn" onclick="window.parent.print()">{label}</button>
    """
    components.html(html, height=60)

# ==========================================
# CACHED ML PIPELINE
# ==========================================
@st.cache_data(show_spinner=False)
def run_ml_pipeline(df_numeric, mode_pca, n_clusters, gunakan_pca):
    """Pipeline ML yang dicache untuk menghindari re-kalkulasi saat slider UI berubah."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)

    if mode_pca == "Mode Visualisasi 3D (3 Komponen)":
        pca = PCA(n_components=3)
    else:
        pca = PCA(n_components=0.72)

    pca_data = pca.fit_transform(scaled_data)
    variansi_terjelaskan = sum(pca.explained_variance_ratio_) * 100
    jumlah_dimensi_baru = pca.n_components_

    data_untuk_kmeans = pca_data if gunakan_pca else scaled_data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_untuk_kmeans)
    
    skor_siluet = 0
    if n_clusters >= 2:
        skor_siluet = silhouette_score(data_untuk_kmeans, cluster_labels)

    return pca_data, variansi_terjelaskan, jumlah_dimensi_baru, cluster_labels, kmeans.cluster_centers_, skor_siluet


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
# CUSTOM CSS: LIQUID GLASS UPGRADE (KUBE.IO STYLE)
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
    min-height: -webkit-fill-available !important; 
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
    border-top: 1px solid rgba(255, 255, 255, 0.15) !important; /* Specular top */
    border-left: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.02) !important;
    border-bottom: 1px solid rgba(0, 0, 0, 0.6) !important;
    box-shadow: 
        0 30px 60px rgba(0, 0, 0, 0.5), 
        inset 0 2px 4px rgba(255, 255, 255, 0.08), 
        inset 0 -4px 8px rgba(0, 0, 0, 0.5) !important; 
    transition: transform 0.4s cubic-bezier(0.25, 1, 0.5, 1), box-shadow 0.4s ease !important;
}

[data-testid="stForm"]:hover, 
[data-testid="stExpander"]:hover {
    transform: translateY(-2px);
    box-shadow: 
        0 40px 80px rgba(0, 0, 0, 0.6), 
        inset 0 2px 6px rgba(255, 255, 255, 0.15), 
        inset 0 -4px 8px rgba(0, 0, 0, 0.6) !important;
}

h1, h2, h3, h4, p, label, span, li, div[data-testid="stMarkdownContainer"] {
    color: rgba(255, 255, 255, 0.9) !important;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.9) !important;
    letter-spacing: 0.2px;
}

/* 3. INPUT FIELDS & SELECTBOX (BARU) */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background: rgba(18, 18, 24, 0.45) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.25) !important;
    border-radius: 12px !important;
    box-shadow: inset 0 3px 6px rgba(0,0,0,0.4), 0 2px 8px rgba(0,0,0,0.2) !important;
    color: white !important;
}

/* 4. SLIDER & TOGGLE ALA KUBE.IO (FIXED BUG DRAGGING - BISA JALAN SEKARANG) */

/* Track Base (Jalur Slider Gelap) */
div[data-baseweb="slider"] > div > div {
    background: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(0, 0, 0, 0.8) !important;
    box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.8) !important;
}

/* Track Aktif (Biru Kaca) - CARA AMAN ANTI-BUG */
div[data-baseweb="slider"] > div > div > div[style*="background"] {
    background: linear-gradient(90deg, rgba(59, 130, 246, 0.7), rgba(96, 165, 250, 1)) !important;
    box-shadow: 0 0 10px rgba(59, 130, 246, 0.4) !important;
}

/* Slider Knob (Bulat Kaca) */
div[data-baseweb="slider"] div[role="slider"] {
    height: 24px !important;
    width: 24px !important; 
    background: rgba(255, 255, 255, 0.15) !important; 
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
    border-top: 2px solid rgba(255, 255, 255, 0.9) !important; /* Specular rim */
    border-radius: 50% !important;
    box-shadow: 
        0 4px 10px rgba(0, 0, 0, 0.5), 
        inset 0 4px 6px rgba(255, 255, 255, 0.4), 
        inset 0 -4px 6px rgba(0, 0, 0, 0.4) !important; 
    /* Dihapus: transform: none !important; (Biar Streamlit bisa ngegeser tombolnya) */
    transition: background 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
}

/* Efek Glow saat Slider ditahan / di-drag */
div[data-baseweb="slider"] div[role="slider"]:active {
    background: rgba(255, 255, 255, 0.3) !important;
    border-color: #60a5fa !important;
    box-shadow: 
        0 0 15px rgba(96, 165, 250, 0.8), 
        inset 0 2px 4px rgba(255, 255, 255, 0.8), 
        inset 0 -2px 4px rgba(0, 0, 0, 0.2) !important;
    /* Dihapus: transform: scale (Biar gak lompat pas pertama diklik) */
}

/* Lip Bezel Toggle/Switch (Tetap Aman) */
[data-testid="stCheckbox"] > label > div[data-baseweb="checkbox"] > div {
    background: rgba(5, 5, 10, 0.6) !important;
    box-shadow: inset 0 4px 8px rgba(0, 0, 0, 0.9), inset 0 -1px 2px rgba(255, 255, 255, 0.2) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(0, 0, 0, 0.8) !important;
}
[data-testid="stCheckbox"] > label > div[data-baseweb="checkbox"] > div > div {
    background: linear-gradient(180deg, #ffffff 0%, #e0e0e0 100%) !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.6), inset 0 2px 4px rgba(255,255,255,1), inset 0 -2px 4px rgba(0,0,0,0.3) !important;
    border-radius: 50% !important;
}/* 5. TINTED DARK GLASS (KOTAK PERINGATAN / KLASTER) */
.glass-alert {
    padding: 24px;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(25px) saturate(180%);
    -webkit-backdrop-filter: blur(25px) saturate(180%);
    color: rgba(255, 255, 255, 0.95);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4), inset 0 -3px 6px rgba(0, 0, 0, 0.6);
    border-bottom: 1px solid rgba(0, 0, 0, 0.5);
}

.glass-green { background: rgba(16, 185, 129, 0.12); border-top: 1px solid rgba(16, 185, 129, 0.4); border-left: 1px solid rgba(16, 185, 129, 0.1); }
.glass-blue { background: rgba(59, 130, 246, 0.12); border-top: 1px solid rgba(59, 130, 246, 0.4); border-left: 1px solid rgba(59, 130, 246, 0.1); }
.glass-orange { background: rgba(245, 158, 11, 0.12); border-top: 1px solid rgba(245, 158, 11, 0.4); border-left: 1px solid rgba(245, 158, 11, 0.1); }
.glass-red { background: rgba(239, 68, 68, 0.12); border-top: 1px solid rgba(239, 68, 68, 0.4); border-left: 1px solid rgba(239, 68, 68, 0.1); }

/* Tombol Form */
div.stButton > button,
div.stDownloadButton > button {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.4) !important; /* Specular highlight */
    border-radius: 12px !important;
    color: white !important;
    font-weight: 500 !important;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3), inset 0 2px 4px rgba(255, 255, 255, 0.1) !important;
    transition: all 0.3s cubic-bezier(0.25, 1, 0.5, 1) !important;
}
div.stButton > button:hover,
div.stDownloadButton > button:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    border-color: rgba(255, 255, 255, 0.4) !important;
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4), inset 0 2px 4px rgba(255, 255, 255, 0.2) !important;
    transform: translateY(-2px);
}
[data-testid="baseButton-formSubmit"] {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(96, 165, 250, 0.2) 100%) !important;
    border: 1px solid rgba(96, 165, 250, 0.3) !important;
    border-top: 1px solid rgba(147, 197, 253, 0.6) !important;
    border-radius: 12px !important;
}
[data-testid="baseButton-formSubmit"]:hover {
    border-color: rgba(255, 255, 255, 0.6) !important;
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5) !important;
}

/* ==========================================
   FIX FULLSCREEN & HIDE STREAMLIT DEFAULT UI
   ========================================== */
footer {
    display: none !important;
    visibility: hidden !important;
    height: 0px !important;
}

.block-container {
    padding-top: 2rem !important; 
    padding-bottom: 1rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* ==========================================
   5. PRINT MODE (HTML TO PDF OPTIMIZATION)
   ========================================== */
@media print {
    html, body, .stApp, div {
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
    }
    
    [data-testid="stSidebar"], 
    header[data-testid="stHeader"], 
    footer {
        display: none !important;
    }

    .stApp {
        background-image: none !important;
        background-color: #0b0b10 !important; 
    }
    .stApp::before { display: none !important; }
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
                # CATATAN: Pindahkan hardcode password ini ke st.secrets("ADMIN_PASSWORD") pas mau deploy ke production!
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
                    df_temp = pd.read_csv(uploaded_file)
                else:
                    df_temp = pd.read_excel(uploaded_file)
                    
                nama_kolom_baru = ['Timestamp', 'Nama', 'Kelas', 'Jurusan', 'Jenis_Kelamin'] + [f'P{i}' for i in range(1, 21)]
                
                # BUGFIX: Safeguard & Readable slicing
                if len(df_temp.columns) >= 25: 
                    df = df_temp.iloc[:, :25].copy()
                    df.columns = nama_kolom_baru
                else:
                    st.error(f"Format file tidak sesuai. Butuh minimal 25 kolom, tapi file cuma punya {len(df_temp.columns)}.")
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

        st.sidebar.header("2. Mode Konfigurasi PCA")
        mode_pca = st.sidebar.radio(
            "Pilih Target Reduksi Dimensi:",
            ["Mode Visualisasi 3D (3 Komponen)", "Mode Validasi Akademis (Target >70%)"]
        )

        st.sidebar.header("3. Konfigurasi Klastering")
        gunakan_pca = st.sidebar.toggle("Aktifkan Reduksi PCA", value=True)
        n_clusters = st.sidebar.slider("Jumlah Klaster (K)", 2, 5, 3)

        # ⚡ EXECUTE CACHED ML PIPELINE ⚡
        with st.spinner("Memproses Model AI..."):
            pca_data, var_explained, dim_baru, labels, centroids, skor_siluet = run_ml_pipeline(
                data_numeric, mode_pca, n_clusters, gunakan_pca
            )
        
        # Inject ML results back to DataFrame
        df['Cluster'] = labels
        if mode_pca == "Mode Visualisasi 3D (3 Komponen)":
            df['PC1'] = pca_data[:, 0]
            df['PC2'] = pca_data[:, 1]
            df['PC3'] = pca_data[:, 2]
            st.sidebar.info(f"Mode Visual Aktif.\nTotal Variansi: {var_explained:.2f}%")
            tampilkan_3d = True 
        else:
            df['PC1'] = pca_data[:, 0]
            df['PC2'] = pca_data[:, 1] if dim_baru > 1 else 0 
            st.sidebar.success(f"Mode Akademis Aktif.\nDimensi: {dim_baru}\nVariansi: {var_explained:.2f}%")
            st.sidebar.caption("Grafik 3D dinonaktifkan.")
            tampilkan_3d = False 

        # =====================================================================
        # MENU BANTUAN HITUNGAN MANUAL EXCEL
        # =====================================================================
        st.sidebar.markdown("---")
        tampilkan_excel = st.sidebar.toggle("🛠️ Mode Bantuan Excel")
        
        if tampilkan_excel:
            st.sidebar.success("💡 **Info Copy:** Klik tombol copy di sudut kanan atas tiap kotak kode di bawah, lalu langsung paste (CTRL+V) ke cell Excel. Data akan otomatis masuk ke dalam kolom!")
            
            st.sidebar.subheader("📍 Centroid Akhir")
            centroid_df = pd.DataFrame(centroids, columns=[f'PC{i+1}' for i in range(centroids.shape[1])])
            centroid_df.index.name = 'Klaster'
            st.sidebar.code(centroid_df.to_csv(sep='\t', decimal=','), language='text')

            st.sidebar.markdown("---")
            st.sidebar.subheader("📋 Koordinat Siswa")
            cols_to_copy = ['Nama', 'PC1', 'PC2', 'PC3'] if tampilkan_3d else ['Nama', 'PC1', 'PC2']
            st.sidebar.code(df[cols_to_copy].to_csv(index=False, sep='\t', decimal=','), language='text')
            
            st.sidebar.caption("Matikan toggle 'Mode Bantuan Excel' jika sudah selesai agar menu kembali rapi.")
        # =====================================================================
        
        if n_clusters >= 2:
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
            df_filtered['Fasilitas'] = df_filtered[DIMENSI_MAP['Fasilitas']].mean(axis=1)
            df_filtered['Kurikulum'] = df_filtered[DIMENSI_MAP['Kurikulum']].mean(axis=1)
            df_filtered['Guru'] = df_filtered[DIMENSI_MAP['Guru']].mean(axis=1)
            df_filtered['Lingkungan'] = df_filtered[DIMENSI_MAP['Lingkungan']].mean(axis=1)

            profile = df_filtered.groupby('Cluster')[['Fasilitas', 'Kurikulum', 'Guru', 'Lingkungan']].mean()
            counts = df_filtered['Cluster'].value_counts().sort_index()
            
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
                    fig_3d.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=600, font_color='white')
                    st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.info(f"Algoritma beroperasi di ruang {dim_baru} dimensi. Ditampilkan sebagai Proyeksi 2D Utama.")
                    
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
                for cluster in profile.index:
                    dim_otomatis = profile.loc[cluster].idxmin()
                    
                    dim_pilihan = st.selectbox(
                        f"Pilih Dimensi untuk dibedah pada Klaster {cluster}:",
                        options=['Fasilitas', 'Kurikulum', 'Guru', 'Lingkungan'],
                        index=['Fasilitas', 'Kurikulum', 'Guru', 'Lingkungan'].index(dim_otomatis),
                        key=f"select_dim_{cluster}" 
                    )
                    
                    skor_dimensi = profile.loc[cluster][dim_pilihan]
                    cols_dimensi = DIMENSI_MAP[dim_pilihan]
                    df_cluster = df_filtered[df_filtered['Cluster'] == cluster]
                    
                    rata_item = df_cluster[cols_dimensi].mean()
                    item_code = rata_item.idxmin()
                    skor_item = rata_item.min()
                    
                    isi_masalah = KAMUS[item_code]['masalah'] 
                    solusi_masalah = KAMUS[item_code]['solusi'] 

                    # BUGFIX: mode crash prevention on empty clusters
                    mode_kelas = df_cluster['Kelas'].mode()
                    kelas_terbanyak = int(mode_kelas[0]) if not mode_kelas.empty else "N/A"
                    jml_kelas = len(df_cluster[df_cluster['Kelas'] == kelas_terbanyak])
                    
                    mode_jur = df_cluster['Jurusan'].mode()
                    jurusan_terbanyak = mode_jur[0] if not mode_jur.empty else "N/A"
                    jml_jurusan = len(df_cluster[df_cluster['Jurusan'] == jurusan_terbanyak])

                    status, glass_color = get_status_color(skor_item)

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
                        with st.expander(f"Daftar Nama Target Investigasi (Klaster {cluster} - Kasus: {dim_pilihan})"):
                            df_kelas = df_cluster[df_cluster['Kelas'] == kelas_terbanyak][['Nama', 'Kelas', 'Jurusan', 'Jenis_Kelamin']].reset_index(drop=True)
                            df_kelas.index += 1 
                            st.markdown(f"**Target Kelas {kelas_terbanyak}:**")
                            st.dataframe(df_kelas, use_container_width=True)
                            st.markdown("---")
                            df_jurusan = df_cluster[df_cluster['Jurusan'] == jurusan_terbanyak][['Nama', 'Kelas', 'Jurusan', 'Jenis_Kelamin']].reset_index(drop=True)
                            df_jurusan.index += 1 
                            st.markdown(f"**Target Jurusan {jurusan_terbanyak}:**")
                            st.dataframe(df_jurusan, use_container_width=True)

                st.markdown("---")
                st.subheader("📥 Cetak Laporan Operasional (PDF)")
                st.info("💡 Klik tombol di bawah ini. Pastikan untuk mencentang opsi **'Background graphics' (Grafik Latar Belakang)** di pengaturan jendela *Print* agar tampilan kaca (Glass UI) tetap terlihat.")
                
                # Menggunakan helper function
                cetak_pdf_button(
                    label="🖨️ Cetak Dashboard / Save as PDF",
                    bg_gradient="linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(96, 165, 250, 0.2) 100%)",
                    border_color="rgba(96, 165, 250, 0.4)",
                    shadow_color="rgba(59, 130, 246, 0.4)"
                )

                st.markdown("---")
                st.subheader("Download Database Mentah")
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
                        height=400,
                        margin=dict(l=40, r=40, t=40, b=40),
                        polar=dict(
                            bgcolor='rgba(15, 15, 20, 0.6)',
                            radialaxis=dict(visible=True, gridcolor='rgba(255,255,255,0.1)', color='rgba(255,255,255,0.4)'),
                            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='rgba(255,255,255,0.8)')
                        ),
                        paper_bgcolor='rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)', 
                        font_color='white'
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
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor='rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)', 
                        font_color='white'
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)

                st.divider()

                st.subheader("Rapor Evaluasi 4 Dimensi Layanan")
                rata_global_series = pd.Series(rata_global).sort_values() 

                for dimensi, skor_dimensi in rata_global_series.items():
                    cols_dimensi = DIMENSI_MAP[dimensi]
                    rata_item_dimensi = df_filtered[cols_dimensi].mean()
                    item_terendah = rata_item_dimensi.idxmin()
                    skor_item_terendah = rata_item_dimensi.min()

                    isi_masalah, solusi_masalah = KAMUS[item_terendah]['masalah'], KAMUS[item_terendah]['solusi']
                    status, glass_color = get_status_color(skor_dimensi)

                    if skor_dimensi >= 4.0:
                        pesan_html = f"<b>Dimensi {dimensi} (Skor: {skor_dimensi:.2f}/5.00) - {status}</b><br><br><b>Saran Preventif:</b> Perhatikan poin <i>'{isi_masalah}'</i> (Skor: {skor_item_terendah:.2f}). <b>Tindakan:</b> {solusi_masalah}"
                    else:
                        pesan_html = f"<b>Dimensi {dimensi} (Skor: {skor_dimensi:.2f}/5.00) - {status}</b><br><br><b>Titik Terlemah:</b> <i>'{isi_masalah}'</i> (Skor: <b>{skor_item_terendah:.2f}</b>).<br><br><b>Tindakan:</b> {solusi_masalah}"

                    st.markdown(f'<div class="glass-alert {glass_color}">{pesan_html}</div>', unsafe_allow_html=True)

                st.markdown("---")
                st.subheader("📥 Cetak Laporan Eksekutif (PDF)")
                st.info("💡 Klik tombol di bawah ini untuk mencetak halaman Laporan Eksekutif ini menjadi PDF.")
                
                cetak_pdf_button(
                    label="📄 Simpan Halaman Eksekutif ke PDF",
                    bg_gradient="linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(52, 211, 153, 0.2) 100%)",
                    border_color="rgba(52, 211, 153, 0.4)",
                    shadow_color="rgba(16, 185, 129, 0.4)"
                )
