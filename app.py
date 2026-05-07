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

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sistem Penilaian Kepuasan SMK", layout="wide", initial_sidebar_state="expanded")

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
# HALAMAN 1: KUESIONER SISWA
# ==========================================
if menu == "Isi Kuesioner (Siswa)":
    st.title("🎓 Evaluasi Layanan SMKN 1 Dawuan")
    st.markdown("Silakan isi kuesioner berikut secara objektif sesuai dengan pengalaman kamu. Skala **1 (Sangat Tidak Puas)** hingga **5 (Sangat Puas)**.")

    with st.form(key='kuesioner_form'):
        st.subheader("Data Responden")
        col1, col2, col3 = st.columns(3)
        with col1:
            nama_siswa = st.text_input("Nama Lengkap (Opsional)")
        with col2:
            kelas = st.selectbox("Kelas", ["10", "11", "12"])
        with col3:
            jurusan = st.selectbox("Jurusan", ["TKJT", "Busana", "DKV", "TO", "ATU"])
        
        jenis_kelamin = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"], horizontal=True)
        st.divider()

        # --- DIMENSI 1: FASILITAS ---
        st.subheader("1. Fasilitas Sekolah")
        p1 = st.slider("Kelengkapan fasilitas lab/bengkel untuk praktik (P1)", 1, 5, 3)
        p2 = st.slider("Kondisi dan kelayakan peralatan belajar di kelas (P2)", 1, 5, 3)
        p3 = st.slider("Kenyamanan dan keamanan area sekolah secara umum (P3)", 1, 5, 3)
        p4 = st.slider("Akses dan kebersihan fasilitas umum (Toilet, Kantin) (P4)", 1, 5, 3)
        p5 = st.slider("Dukungan fasilitas terhadap kelancaran belajar (P5)", 1, 5, 3)
        st.divider()

        # --- DIMENSI 2: KURIKULUM ---
        st.subheader("2. Kurikulum dan Materi Pembelajaran")
        p6 = st.slider("Kesesuaian materi dengan kebutuhan industri/dunia kerja (P6)", 1, 5, 3)
        p7 = st.slider("Tingkat kemudahan dalam memahami materi yang diajarkan (P7)", 1, 5, 3)
        p8 = st.slider("Kesesuaian kurikulum dengan minat dan bakat siswa (P8)", 1, 5, 3)
        p9 = st.slider("Variasi metode pembelajaran yang digunakan (tidak membosankan) (P9)", 1, 5, 3)
        p10 = st.slider("Manfaat materi pelajaran untuk masa depan karir (P10)", 1, 5, 3)
        st.divider()

        # --- DIMENSI 3: GURU ---
        st.subheader("3. Kinerja dan Kompetensi Guru")
        p11 = st.slider("Penguasaan materi oleh guru saat mengajar di kelas (P11)", 1, 5, 3)
        p12 = st.slider("Kejelasan guru dalam menyampaikan penjelasan (P12)", 1, 5, 3)
        p13 = st.slider("Sikap keteladanan, kedisiplinan, dan etika guru (P13)", 1, 5, 3)
        p14 = st.slider("Kemudahan menghubungi guru saat mengalami kesulitan belajar (P14)", 1, 5, 3)
        p15 = st.slider("Ketepatan waktu guru dalam mengisi jam pelajaran (P15)", 1, 5, 3)
        st.divider()

        # --- DIMENSI 4: LINGKUNGAN ---
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
            "Nama": nama_siswa if nama_siswa else "Anonim",
            "Kelas": kelas, "Jurusan": jurusan, "Jenis_Kelamin": jenis_kelamin,
            "P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5,
            "P6": p6, "P7": p7, "P8": p8, "P9": p9, "P10": p10,
            "P11": p11, "P12": p12, "P13": p13, "P14": p14, "P15": p15,
            "P16": p16, "P17": p17, "P18": p18, "P19": p19, "P20": p20
        }
        df_baru = pd.DataFrame([data_responden])
        st.success("Terima kasih! Data evaluasi berhasil disimpan sementara.")
        st.balloons()
        # Note: Logika simpan ke Database (Supabase/MySQL) akan ditambahkan di sini nantinya

# ==========================================
# HALAMAN 2: DASHBOARD ADMIN (KODE LAMA)
# ==========================================
elif menu == "Dashboard Analisis (Admin)":
    st.title("📊 Analisis Kepuasan Siswa - SMK N 1 Dawuan")
    st.markdown("Sistem Analisis Kepuasan Siswa Berbasis **Machine Learning (PCA & K-Means Clustering)**.")

    # --- SIDEBAR: UPLOAD ---
    st.sidebar.header("1. Input Data")
    uploaded_file = st.sidebar.file_uploader("Unggah file Excel/CSV dari Google Forms", type=["csv", "xlsx"])

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
                st.error("Format file tidak sesuai! Pastikan ada kolom Timestamp, Nama, Kelas, Jurusan, JK, dan 20 Pertanyaan.")
                st.stop()
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")
            st.stop()

        kolom_nilai = [f'P{i}' for i in range(1, 21)]
        jml_awal = len(df)
        df = df.dropna(subset=kolom_nilai).reset_index(drop=True)
        jml_akhir = len(df)
        
        if jml_awal != jml_akhir:
            st.sidebar.warning(f"⚠️ Ditemukan {jml_awal - jml_akhir} data kuesioner tidak lengkap. Otomatis diabaikan.")

        data_numeric = df[kolom_nilai]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_numeric)

        # ==========================================
        # 🧠 LOGIKA SWITCH PCA (VISUALISASI VS AKADEMIS)
        # ==========================================
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
            
            st.sidebar.info(f"👁️ Mode Visual Aktif.\nTotal Variansi: **{variansi_terjelaskan:.2f}%**")
            tampilkan_3d = True 
            
        else:
            # Mesin otomatis mencari jumlah komponen untuk mencapai variansi minimal 72%
            pca = PCA(n_components=0.72) 
            pca_data = pca.fit_transform(scaled_data)
            variansi_terjelaskan = sum(pca.explained_variance_ratio_) * 100
            jumlah_dimensi_baru = pca.n_components_
            
            # Ekstrak setidaknya 2 komponen pertama untuk Proyeksi Scatter 2D
            df['PC1'] = pca_data[:, 0]
            df['PC2'] = pca_data[:, 1]
            
            st.sidebar.success(f"🎓 Mode Akademis Aktif.\nDimensi Terbentuk: **{jumlah_dimensi_baru} Komponen**\nTotal Variansi: **{variansi_terjelaskan:.2f}%**")
            st.sidebar.caption("⚠️ Grafik 3D dinonaktifkan karena dimensi ruang K-Means melebihi 3.")
            tampilkan_3d = False 

        # --- K-MEANS CLUSTERING ---
        st.sidebar.header("3. Konfigurasi Klastering")
        gunakan_pca = st.sidebar.toggle("🟢 Aktifkan Reduksi PCA", value=True)
        data_untuk_kmeans = pca_data if gunakan_pca else scaled_data 

        n_clusters = st.sidebar.slider("Jumlah Klaster (K)", 2, 5, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(data_untuk_kmeans)

        if n_clusters >= 2:
            skor_siluet = silhouette_score(data_untuk_kmeans, df['Cluster'])
            with st.sidebar.expander("🔬 Validasi Matematis (K-Means)"):
                st.write(f"Silhouette Score (k={n_clusters}): **{skor_siluet:.3f}**")
                st.caption("Semakin mendekati 1.0, semakin optimal pemisahan klaster.")
            
        # --- FILTER DATA ---
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
            st.warning("⚠️ Data tidak ditemukan dengan kombinasi filter tersebut.")
        else:
            df_filtered['Fasilitas'] = df_filtered[[f'P{i}' for i in range(1, 6)]].mean(axis=1)
            df_filtered['Kurikulum'] = df_filtered[[f'P{i}' for i in range(6, 11)]].mean(axis=1)
            df_filtered['Guru'] = df_filtered[[f'P{i}' for i in range(11, 16)]].mean(axis=1)
            df_filtered['Lingkungan'] = df_filtered[[f'P{i}' for i in range(16, 21)]].mean(axis=1)

            profile = df_filtered.groupby('Cluster')[['Fasilitas', 'Kurikulum', 'Guru', 'Lingkungan']].mean()
            counts = df_filtered['Cluster'].value_counts().sort_index()

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
                'P1': 'Inventarisasi lab dan ajukan pengadaan alat.', 'P2': 'Maintenance rutin dan perbarui teknologi.',
                'P3': 'Tingkatkan keamanan dan patroli sekolah.', 'P4': 'Perbaiki layout dan tambah fasilitas toilet/kantin.',
                'P5': 'Evaluasi penggunaan lab agar efektif.', 'P6': 'Undang praktisi industri dan sinkronisasi kurikulum.',
                'P7': 'Adakan tutor sebaya dan sederhanakan modul.', 'P8': 'Perkuat program konseling karir dari guru BK.',
                'P9': 'Training guru untuk metode belajar interaktif.', 'P10': 'Adakan seminar prospek karir/kunjungan industri.',
                'P11': 'Ikutkan guru dalam diklat/magang industri.', 'P12': 'Evaluasi cara mengajar dan kumpulkan feedback.',
                'P13': 'Tegakkan kode etik dan beri penghargaan guru teladan.', 'P14': 'Wajibkan jam konsultasi siswa di luar kelas.',
                'P15': 'Terapkan presensi ketat dan sanksi keterlambatan.', 'P16': 'Galakkan program kebersihan dan evaluasi OB.',
                'P17': 'Tindak tegas pelanggaran/bullying.', 'P18': 'Perbaiki fasilitas kelas & tegakkan tatib.',
                'P19': 'Adakan kegiatan kebersamaan lintas kelas.', 'P20': 'Kampanyekan budaya 5S secara masif.'
            }

            dimensi_map = {
                'Fasilitas': [f'P{i}' for i in range(1, 6)], 'Kurikulum': [f'P{i}' for i in range(6, 11)],
                'Guru': [f'P{i}' for i in range(11, 16)], 'Lingkungan': [f'P{i}' for i in range(16, 21)]
            }

            jml_pria = len(df_filtered[df_filtered['Jenis_Kelamin'].str.contains('Pria|Laki', case=False, na=False)])
            jml_wanita = len(df_filtered) - jml_pria
            
            tab1, tab2 = st.tabs(["📊 Dashboard Analisis", "📑 Laporan Eksekutif"])

            # ==========================================
            # TAB 1: DASHBOARD ANALISIS 
            # ==========================================
            with tab1:
                st.subheader("👥 Statistik Responden")
                cols = st.columns(len(df_filtered['Cluster'].unique()))
                for i, (cls, count) in enumerate(counts.items()):
                    cols[i].metric(f"Klaster {cls}", f"{count} Siswa")

                # Menerapkan logika kondisional untuk plot 3D dan 2D
                if tampilkan_3d:
                    fig_3d = px.scatter_3d(
                        df_filtered, x='PC1', y='PC2', z='PC3', color=df_filtered['Cluster'].astype(str),
                        hover_data=['Nama', 'Kelas', 'Jurusan'], title=f"Visualisasi 3D ({len(df_filtered)} Siswa)",
                        labels={'color': 'Klaster'}
                    )
                    fig_3d.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=600, font_color='white')
                    st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.info(f"🌌 Algoritma saat ini beroperasi di ruang **{pca.n_components_} dimensi** untuk mengamankan **{variansi_terjelaskan:.2f}%** informasi asli. Tampilan spasial dikonversi menjadi Proyeksi 2D Utama.")
                    
                    # --- PENGGANTI: PROYEKSI 2D SCATTER PLOT ---
                    st.markdown("**📉 Proyeksi 2D (Komponen Utama 1 vs Komponen Utama 2)**")
                    fig_2d = px.scatter(
                        df_filtered, x='PC1', y='PC2', color=df_filtered['Cluster'].astype(str),
                        hover_data=['Nama', 'Kelas', 'Jurusan'], 
                        labels={'color': 'Klaster'},
                        title="Bayangan Klaster di Dimensi Terbesar (PC1 & PC2)"
                    )
                    fig_2d.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
                    fig_2d.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                    st.plotly_chart(fig_2d, use_container_width=True)
                    st.caption("💡 *Grafik di atas adalah proyeksi (bayangan) 2D dari algoritma multi-dimensi. Posisi titik mewakili sebaran siswa secara garis besar.*")

                st.subheader("📈 Perbandingan Skor Rata-rata per Klaster")
                profile_reset = profile.reset_index()
                profile_reset['Label_Klaster'] = profile_reset['Cluster'].apply(lambda x: f"Klaster {x}<br>({counts[x]} Siswa)")
                
                fig_bar = px.bar(
                    profile_reset, x='Label_Klaster', y=['Fasilitas', 'Kurikulum', 'Guru', 'Lingkungan'],
                    barmode='group', labels={'value': 'Skor (1-5)', 'variable': 'Dimensi', 'Label_Klaster': 'Kelompok'},
                    color_discrete_sequence=['#3b82f6', '#f59e0b', '#10b981', '#ef4444']
                )
                fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_bar, use_container_width=True)

                st.subheader("🔍 Bedah Investigasi per Klaster")
                for cluster in profile.index:
                    dim_otomatis = profile.loc[cluster].idxmin()
                    
                    dim_pilihan = st.selectbox(
                        f"🔎 Pilih Dimensi untuk dibedah pada Klaster {cluster}:",
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
                        status, warna_alert = "Sangat Memuaskan 🌟", st.success
                    elif skor_item >= 3.0:
                        status, warna_alert = "Cukup Memuaskan 🔵", st.info
                    elif skor_item >= 2.0:
                        status, warna_alert = "Perlu Perbaikan 🟠", st.warning
                    else:
                        status, warna_alert = "Kritis (Perbaikan Segera!) 🔴", st.error

                    if skor_item >= 4.0:
                        pesan = f"**Klaster {cluster} (Dimensi: {dim_pilihan} - Status: {status})**\n\n✨ **Kondisi Aman:** Secara umum siswa di klaster ini sudah PUAS dengan **{dim_pilihan}**. Aspek terendah ada pada poin *'{isi_masalah}'* namun skornya masih sangat aman di angka **{skor_item:.2f}/5.00**.\n\n📈 **Saran:** Pertahankan kinerja. Preventif: {solusi_masalah}"
                        warna_alert(pesan)
                        with st.expander(f"📋 Lihat Daftar Mayoritas Siswa (Klaster {cluster})"):
                            st.write(f"Siswa di klaster ini tidak memerlukan investigasi mendalam untuk masalah {dim_pilihan}.")
                    else:
                        pesan = f"**Klaster {cluster} (Dimensi: {dim_pilihan} - Status: {status})**\n\n👉 **Akar Masalah:** *{isi_masalah}* (Skor: **{skor_item:.2f}/5.00**).\n\n🛠️ **Rekomendasi Solusi:** {solusi_masalah}\n\n🎯 **Target Investigasi:** Fokus pada **Kelas {kelas_terbanyak}** ({jml_kelas} anak) dan **Jurusan {jurusan_terbanyak}** ({jml_jurusan} anak)."
                        warna_alert(pesan)
                        with st.expander(f"📋 Klik untuk Daftar Nama Target Investigasi (Klaster {cluster} - Kasus: {dim_pilihan})"):
                            df_kelas = df_cluster[df_cluster['Kelas'] == kelas_terbanyak][['Nama', 'Kelas', 'Jurusan', 'Jenis_Kelamin']].reset_index(drop=True)
                            df_kelas.index += 1 
                            st.markdown(f"**🎯 1. Target Kelas {kelas_terbanyak}:**")
                            st.dataframe(df_kelas, use_container_width=True)
                            st.markdown("---")
                            df_jurusan = df_cluster[df_cluster['Jurusan'] == jurusan_terbanyak][['Nama', 'Kelas', 'Jurusan', 'Jenis_Kelamin']].reset_index(drop=True)
                            df_jurusan.index += 1 
                            st.markdown(f"**🎯 2. Target Jurusan {jurusan_terbanyak}:**")
                            st.dataframe(df_jurusan, use_container_width=True)

                st.markdown("---")
                st.subheader("📥 Cetak Laporan Operasional (PDF)")
                def buat_pdf_dashboard():
                    fig_bar.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", font_color="black")
                    fig_bar.write_image("temp_bar.png", engine="kaleido", width=1000, height=450)
                    
                    pdf = FPDF()
                    pdf.add_page()
                    
                    pdf.set_font("Arial", 'B', 16)
                    pdf.cell(0, 8, txt="LAPORAN OPERASIONAL KLASTER SISWA", ln=True, align='C')
                    pdf.set_font("Arial", '', 11)
                    pdf.cell(0, 6, txt=f"SMK N 1 DAWUAN | Dicetak melalui Sistem Analisis Kepuasan", ln=True, align='C')
                    pdf.line(10, 25, 200, 25) 
                    pdf.ln(10)
                    
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 8, txt="1. Demografi Responden:", ln=True)
                    pdf.set_font("Arial", '', 11)
                    pdf.cell(0, 6, txt=f"- Total Siswa Dianalisis: {len(df_filtered)} Siswa", ln=True)
                    pdf.cell(0, 6, txt=f"- Jenis Kelamin: Laki-laki ({jml_pria}), Perempuan ({jml_wanita})", ln=True)
                    pdf.cell(0, 6, txt=f"- Jumlah Klaster Terbentuk: {n_clusters} Kelompok", ln=True)
                    pdf.ln(5)

                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 8, txt="2. Grafik Perbandingan Antar Klaster:", ln=True)
                    pdf.image("temp_bar.png", x=10, w=190)
                    pdf.ln(2)
                    
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, txt="3. Rincian Temuan & Prioritas Perbaikan:", ln=True)
                    
                    for cluster in profile.index:
                        dim_terendah = st.session_state.get(f"select_dim_{cluster}", profile.loc[cluster].idxmin())
                        skor_terendah = profile.loc[cluster][dim_terendah]
                        jml_siswa = len(df_filtered[df_filtered['Cluster'] == cluster])
                        
                        if skor_terendah >= 4.0: status_teks, r, g, b = "Sangat Memuaskan (Aman)", 0, 150, 0 
                        elif skor_terendah >= 3.0: status_teks, r, g, b = "Cukup (Perlu Peningkatan)", 0, 0, 200 
                        elif skor_terendah >= 2.0: status_teks, r, g, b = "Kurang (Perlu Perbaikan)", 200, 100, 0 
                        else: status_teks, r, g, b = "Kritis (Perbaikan Segera!)", 220, 0, 0 

                        pdf.set_font("Arial", 'B', 11)
                        pdf.set_text_color(0, 0, 0)
                        pdf.cell(25, 8, txt=f"> Klaster {cluster}: ", ln=False)
                        pdf.set_text_color(r, g, b) 
                        pdf.cell(0, 8, txt=status_teks, ln=True)
                        
                        pdf.set_text_color(0, 0, 0) 
                        pdf.set_font("Arial", '', 10)
                        pdf.cell(0, 6, txt=f"   - Jumlah Siswa: {jml_siswa} Orang", ln=True)
                        pdf.cell(0, 6, txt=f"   - Prioritas Utama: Dimensi {dim_terendah} (Skor: {skor_terendah:.2f}/5.00)", ln=True)
                        pdf.ln(2)

                    pdf.output("temp_dashboard.pdf")
                    with open("temp_dashboard.pdf", "rb") as f: pdf_bytes_dash = f.read()
                    for file in ["temp_bar.png", "temp_dashboard.pdf"]:
                        if os.path.exists(file): os.remove(file)
                    
                    fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                    return pdf_bytes_dash

                if st.button("📊 Proses PDF Operasional"):
                    with st.spinner("Menyiapkan dokumen elegan..."):
                        st.session_state['pdf_dash_ready'] = buat_pdf_dashboard()
                        st.success("PDF siap diunduh!")

                if 'pdf_dash_ready' in st.session_state:
                    st.download_button(label="⬇️ Download PDF Operasional", data=st.session_state['pdf_dash_ready'], file_name="Laporan_Operasional_Klaster.pdf", mime="application/pdf")

                st.markdown("---")
                st.subheader("💾 Download Database Mentah")
                csv = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Database Lengkap (.csv)", csv, "Database_Kepuasan_SMK.csv", "text/csv")

            # ==========================================
            # TAB 2: LAPORAN EKSEKUTIF 
            # ==========================================
            with tab2:
                st.header("📑 Executive Summary")
                st.markdown(f"Laporan ini merangkum hasil analisis kepuasan dari **{len(df_filtered)} responden**.")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🕸️ Radar Chart Kinerja Global")
                    rata_global = {
                        'Fasilitas': df_filtered['Fasilitas'].mean(),
                        'Kurikulum': df_filtered['Kurikulum'].mean(),
                        'Guru': df_filtered['Guru'].mean(),
                        'Lingkungan': df_filtered['Lingkungan'].mean()
                    }
                    df_radar = pd.DataFrame(dict(Skor=list(rata_global.values()), Dimensi=list(rata_global.keys())))
                    fig_radar = px.line_polar(df_radar, r='Skor', theta='Dimensi', line_close=True, range_r=[0,5], markers=True)
                    fig_radar.update_traces(fill='toself', line_color='cyan')
                    
                    fig_radar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                    st.plotly_chart(fig_radar, use_container_width=True)

                with col2:
                    klaster_terburuk = profile.mean(axis=1).idxmin()
                    df_kritis = df_filtered[df_filtered['Cluster'] == klaster_terburuk]
                    st.subheader(f"🍩 Proporsi Jurusan Kritis (Klaster {klaster_terburuk})")
                    
                    fig_donut = px.pie(df_kritis, names='Jurusan', hole=0.4)
                    
                    fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                    st.plotly_chart(fig_donut, use_container_width=True)

                st.divider()

                st.subheader("📋 Rapor Evaluasi 4 Dimensi Layanan")
                rata_global_series = pd.Series(rata_global).sort_values() 

                for dimensi, skor_dimensi in rata_global_series.items():
                    cols_dimensi = dimensi_map[dimensi]
                    rata_item_dimensi = df_filtered[cols_dimensi].mean()
                    item_terendah = rata_item_dimensi.idxmin()
                    skor_item_terendah = rata_item_dimensi.min()

                    isi_masalah, solusi_masalah = kamus_masalah[item_terendah], kamus_solusi[item_terendah]

                    if skor_dimensi >= 4.0:
                        status, warna_alert = "Sangat Memuaskan 🌟", st.success
                        pesan = f"**Dimensi {dimensi} (Skor: {skor_dimensi:.2f}/5.00) - {status}**\n\n💡 **Saran Preventif:** Perhatikan poin *'{isi_masalah}'* (Skor: {skor_item_terendah:.2f}). **Tindakan:** {solusi_masalah}"
                    else:
                        if skor_dimensi >= 3.0: status, warna_alert = "Cukup Memuaskan 🔵", st.info
                        elif skor_dimensi >= 2.0: status, warna_alert = "Perlu Perbaikan 🟠", st.warning
                        else: status, warna_alert = "Kritis (Perbaikan Segera!) 🔴", st.error
                        pesan = f"**Dimensi {dimensi} (Skor: {skor_dimensi:.2f}/5.00) - {status}**\n\n🚨 **Titik Terlemah:** *'{isi_masalah}'* (Skor: **{skor_item_terendah:.2f}**).\n\n🛠️ **Tindakan:** {solusi_masalah}"
                    
                    warna_alert(pesan)

                st.markdown("---")
                st.subheader("📥 Cetak Laporan Eksekutif (PDF)")
                def buat_pdf():
                    fig_radar.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", font_color="black")
                    fig_radar.write_image("temp_radar.png", engine="kaleido", width=600, height=400)
                    
                    fig_donut.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", font_color="black")
                    fig_donut.write_image("temp_donut.png", engine="kaleido", width=500, height=400)
                    
                    pdf = FPDF()
                    pdf.add_page()
                    
                    pdf.set_font("Arial", 'B', 18)
                    pdf.cell(0, 10, txt="EXECUTIVE SUMMARY: KEPUASAN SISWA", ln=True, align='C')
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(0, 8, txt="SMK NEGERI 1 DAWUAN", ln=True, align='C')
                    pdf.line(10, 30, 200, 30)
                    pdf.ln(5)
                    
                    pdf.set_font("Arial", '', 11)
                    teks_demo = f"Laporan ini merupakan hasil komputasi algoritma Machine Learning terhadap {len(df_filtered)} siswa."
                    pdf.multi_cell(0, 6, txt=teks_demo)
                    pdf.ln(5)
                    
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 8, txt="A. Pemetaan Kinerja & Profiling Kritis:", ln=True)
                    pdf.image("temp_radar.png", x=10, y=55, w=100)
                    pdf.image("temp_donut.png", x=110, y=55, w=90)
                    pdf.ln(70) 
                    
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 8, txt="B. Rapor Evaluasi 4 Dimensi Layanan:", ln=True)
                    
                    for dimensi, skor_dimensi in rata_global_series.items():
                        if skor_dimensi >= 4.0: status_pdf, r, g, b = "SANGAT MEMUASKAN", 0, 150, 0
                        elif skor_dimensi >= 3.0: status_pdf, r, g, b = "CUKUP MEMUASKAN", 0, 0, 200
                        elif skor_dimensi >= 2.0: status_pdf, r, g, b = "PERLU PERBAIKAN", 200, 100, 0
                        else: status_pdf, r, g, b = "KRITIS", 220, 0, 0

                        pdf.set_font("Arial", 'B', 11)
                        pdf.set_text_color(0, 0, 0)
                        pdf.cell(40, 7, txt=f"- {dimensi}:", ln=False)
                        pdf.cell(30, 7, txt=f"Skor {skor_dimensi:.2f}", ln=False)
                        
                        pdf.set_text_color(r, g, b)
                        pdf.cell(0, 7, txt=f"[{status_pdf}]", ln=True)
                    
                    pdf.set_text_color(0, 0, 0)
                    pdf.ln(5)
                    
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 8, txt="C. Fokus Perbaikan & Solusi Manajerial:", ln=True)
                    
                    for dimensi, skor_dimensi in rata_global_series.items():
                        if skor_dimensi < 4.0: 
                            cols_dimensi = dimensi_map[dimensi]
                            rata_item_dimensi = df_filtered[cols_dimensi].mean()
                            item_terendah = rata_item_dimensi.idxmin()
                            
                            teks_masalah = kamus_masalah[item_terendah]
                            teks_solusi = kamus_solusi[item_terendah]
                            
                            pdf.set_font("Arial", 'B', 11)
                            pdf.cell(0, 7, txt=f">> Masalah Utama pada {dimensi}:", ln=True)
                            pdf.set_font("Arial", '', 10)
                            pdf.multi_cell(0, 5, txt=f"Keluhan Siswa: {teks_masalah}")
                            pdf.ln(2)
                            pdf.multi_cell(0, 5, txt=f"Saran Tindakan: {teks_solusi}")
                            pdf.ln(3)

                    pdf.output("temp_laporan.pdf")
                    with open("temp_laporan.pdf", "rb") as f: pdf_bytes = f.read()
                    for file in ["temp_radar.png", "temp_donut.png", "temp_laporan.pdf"]:
                        if os.path.exists(file): os.remove(file)
                    
                    fig_radar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                    fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                    return pdf_bytes

                if st.button("📄 Proses PDF Eksekutif"):
                    with st.spinner("Meracik laporan eksekutif lengkap..."):
                        st.session_state['pdf_ready'] = buat_pdf()
                        st.success("PDF Eksekutif berhasil dibuat!")

                if 'pdf_ready' in st.session_state:
                    st.download_button(label="⬇️ Download Laporan Kepsek (PDF)", data=st.session_state['pdf_ready'], file_name="Laporan_Manajerial_SMK.pdf", mime="application/pdf")

else:
    st.info("Silakan pilih menu di sidebar.")
