import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import streamlit.components.v1 as components
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from supabase import create_client

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sistem Penilaian Kepuasan SMK", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# INISIALISASI SESSION STATE
# ==========================================
if 'sudah_login' not in st.session_state: st.session_state['sudah_login'] = False
if 'data_diri' not in st.session_state: st.session_state['data_diri'] = {}
if 'admin_logged_in' not in st.session_state: st.session_state['admin_logged_in'] = False

# ==========================================
# KONEKSI & OPTIMASI DATABASE SUPABASE
# ==========================================
@st.cache_resource
def init_connection():
    try:
        return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except Exception:
        return None

supabase = init_connection()

@st.cache_data(ttl=600) # Cache data selama 10 menit biar enteng
def fetch_supabase_data():
    if supabase:
        response = supabase.table("kuesioner").select("*").execute()
        return pd.DataFrame(response.data)
    return pd.DataFrame()

# ==========================================
# CUSTOM CSS: PURE LIQUID GLASS KUBE.IO
# ==========================================
liquid_glass_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* 1. Background Dinamis */
.stApp {
    background-image: url('https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=2564&auto=format&fit=crop'); 
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-color: rgba(10, 10, 15, 0.7); 
    background-blend-mode: overlay;
    font-family: 'Inter', sans-serif !important;
}

/* 2. LIQUID GLASS CONTAINERS (Forms, Expanders, Sidebar) */
[data-testid="stForm"], 
[data-testid="metric-container"], 
[data-testid="stExpander"],
[data-testid="stSidebar"],
[data-testid="stFileUploadDropzone"],
.glass-alert {
    background: rgba(255, 255, 255, 0.05) !important; 
    backdrop-filter: blur(20px) saturate(150%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(150%) !important;
    border-radius: 24px !important; 
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-left: 1px solid rgba(255, 255, 255, 0.3) !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important; 
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
    padding: 20px;
}

[data-testid="stForm"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4) !important;
}

/* Teks Putih Bersih */
h1, h2, h3, h4, p, label, span, li, div[data-testid="stMarkdownContainer"] {
    color: rgba(255, 255, 255, 0.95) !important;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5) !important;
}

/* 3. LIQUID GLASS BUTTONS */
div.stButton > button, div.stDownloadButton > button, [data-testid="baseButton-formSubmit"] {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.5) !important;
    border-left: 1px solid rgba(255, 255, 255, 0.5) !important;
    border-radius: 16px !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
}

div.stButton > button:hover, div.stDownloadButton > button:hover, [data-testid="baseButton-formSubmit"]:hover {
    background: rgba(255, 255, 255, 0.2) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3), inset 0 1px 2px rgba(255, 255, 255, 0.6) !important;
}

/* 4. LIQUID GLASS TOGGLE SWITCH */
[data-testid="stCheckbox"] > label > div[data-baseweb="checkbox"] > div {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    box-shadow: inset 0 2px 5px rgba(0,0,0,0.3) !important;
}
[data-testid="stCheckbox"] > label > div[data-baseweb="checkbox"] > div > div {
    background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,255,255,0.5)) !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.4) !important;
}

/* KOTAK KLASTER WARNA (Tetap Glass) */
.glass-green { background: rgba(16, 185, 129, 0.2) !important; border-top: 1px solid rgba(16, 185, 129, 0.5) !important; }
.glass-blue { background: rgba(59, 130, 246, 0.2) !important; border-top: 1px solid rgba(59, 130, 246, 0.5) !important; }
.glass-orange { background: rgba(245, 158, 11, 0.2) !important; border-top: 1px solid rgba(245, 158, 11, 0.5) !important; }
.glass-red { background: rgba(239, 68, 68, 0.2) !important; border-top: 1px solid rgba(239, 68, 68, 0.5) !important; }

/* FIX UI */
footer { display: none !important; }
.block-container { padding: 2rem 1rem 1rem 1rem !important; }
.stApp { min-height: -webkit-fill-available !important; }

/* PRINT MODE */
@media print {
    html, body, .stApp, div { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
    [data-testid="stSidebar"], header[data-testid="stHeader"], footer { display: none !important; }
    .stApp { background-image: none !important; background-color: #0b0b10 !important; }
}
</style>
"""
st.markdown(liquid_glass_css, unsafe_allow_html=True)

# --- MENU NAVIGASI ---
st.sidebar.title("Navigasi Sistem")
menu = st.sidebar.selectbox("Pilih Halaman", ["Isi Kuesioner (Siswa)", "Dashboard Analisis (Admin)"])
st.sidebar.markdown("---")

# ==========================================
# HALAMAN 1: KUESIONER SISWA
# ==========================================
if menu == "Isi Kuesioner (Siswa)":
    
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
            p = [st.slider(f"P{i}", 1, 5, 3, label_visibility="collapsed") for i in range(1, 6)] # Optimized
            
            st.subheader("2. Kurikulum dan Materi Pembelajaran")
            p.extend([st.slider(f"P{i}", 1, 5, 3, label_visibility="collapsed") for i in range(6, 11)])
            
            st.subheader("3. Kinerja dan Kompetensi Guru")
            p.extend([st.slider(f"P{i}", 1, 5, 3, label_visibility="collapsed") for i in range(11, 16)])
            
            st.subheader("4. Lingkungan Sekolah")
            p.extend([st.slider(f"P{i}", 1, 5, 3, label_visibility="collapsed") for i in range(16, 21)])
            
            submit_button = st.form_submit_button(label='Kirim Evaluasi')

        if submit_button:
            data_responden = {
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **st.session_state['data_diri'],
                **{f"P{i+1}": val for i, val in enumerate(p)}
            }
            
            if supabase:
                try:
                    supabase.table("kuesioner").insert(data_responden).execute()
                    st.success("Terima kasih. Data evaluasi berhasil disimpan.")
                    st.cache_data.clear() # Clear cache agar data admin otomatis terupdate
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
    
    if not st.session_state['admin_logged_in']:
        st.title("Akses Terbatas Manajemen")
        col_space1, col_login, col_space2 = st.columns([1, 2, 1])
        with col_login:
            with st.form(key='admin_login_form'):
                st.subheader("Login Sistem")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button(label='Masuk Dashboard'):
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

    # --- DATA SOURCE ---
    st.sidebar.header("1. Sumber Data")
    sumber_data = st.sidebar.radio("Metode pengambilan data:", ["Database (Otomatis)", "Upload Manual (Excel/CSV)"])

    df = None 
    if sumber_data == "Database (Otomatis)":
        df_raw = fetch_supabase_data()
        if not df_raw.empty:
            df = df_raw.copy()
            st.info(f"Berhasil menarik {len(df)} data kuesioner dari database.")
        else:
            st.warning("Database kosong atau koneksi gagal.")
    else:
        uploaded_file = st.sidebar.file_uploader("Unggah file", type=["csv", "xlsx"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                df = df.iloc[:, :25]
                df.columns = ['Timestamp', 'Nama', 'Kelas', 'Jurusan', 'Jenis_Kelamin'] + [f'P{i}' for i in range(1, 21)]
            except Exception as e:
                st.error(f"Error membaca file: {e}")

    # --- ML PROCESSING ---
    if df is not None:
        kolom_nilai = [f'P{i}' for i in range(1, 21)]
        df = df.dropna(subset=kolom_nilai).reset_index(drop=True)
        
        scaled_data = StandardScaler().fit_transform(df[kolom_nilai])

        st.sidebar.header("2. Mode Konfigurasi PCA")
        mode_pca = st.sidebar.radio("Target Reduksi Dimensi:", ["Visualisasi 3D (3 Komponen)", "Validasi Akademis (>70%)"])

        if mode_pca == "Visualisasi 3D (3 Komponen)":
            pca = PCA(n_components=3)
            pca_data = pca.fit_transform(scaled_data)
            df['PC1'], df['PC2'], df['PC3'] = pca_data[:, 0], pca_data[:, 1], pca_data[:, 2]
            tampilkan_3d = True 
        else:
            pca = PCA(n_components=0.72) 
            pca_data = pca.fit_transform(scaled_data)
            df['PC1'] = pca_data[:, 0]
            df['PC2'] = pca_data[:, 1] if pca.n_components_ > 1 else 0 
            tampilkan_3d = False 

        st.sidebar.header("3. Konfigurasi Klastering")
        gunakan_pca = st.sidebar.toggle("Aktifkan Reduksi PCA", value=True)
        data_untuk_kmeans = pca_data if gunakan_pca else scaled_data 

        n_clusters = st.sidebar.slider("Jumlah Klaster (K)", 2, 5, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(data_untuk_kmeans)

        # Mode Bantuan Excel
        st.sidebar.markdown("---")
        if st.sidebar.toggle("🛠️ Mode Bantuan Excel"):
            st.sidebar.success("Copy-Paste langsung ke cell Excel.")
            centroid_df = pd.DataFrame(kmeans.cluster_centers_, columns=[f'PC{i+1}' for i in range(kmeans.cluster_centers_.shape[1])])
            st.sidebar.code(centroid_df.to_csv(sep='\t', decimal=','), language='text')
            st.sidebar.code(df[['Nama', 'PC1', 'PC2', 'PC3']].to_csv(index=False, sep='\t', decimal=',') if tampilkan_3d else df[['Nama', 'PC1', 'PC2']].to_csv(index=False, sep='\t', decimal=','), language='text')

        if n_clusters >= 2:
            skor_siluet = silhouette_score(data_untuk_kmeans, df['Cluster'])
            st.sidebar.metric(label=f"Silhouette Score (K={n_clusters})", value=f"{skor_siluet:.3f}")

        # --- FILTER DATA ---
        st.sidebar.header("4. Filter Data")
        pilih_jurusan = st.sidebar.multiselect("Jurusan:", df['Jurusan'].unique(), default=df['Jurusan'].unique())
        pilih_kelas = st.sidebar.multiselect("Kelas:", sorted(df['Kelas'].unique()), default=sorted(df['Kelas'].unique()))
        
        df_filtered = df[(df['Jurusan'].isin(pilih_jurusan)) & (df['Kelas'].isin(pilih_kelas))].copy()

        if not df_filtered.empty:
            df_filtered['Fasilitas'] = df_filtered[[f'P{i}' for i in range(1, 6)]].mean(axis=1)
            df_filtered['Kurikulum'] = df_filtered[[f'P{i}' for i in range(6, 11)]].mean(axis=1)
            df_filtered['Guru'] = df_filtered[[f'P{i}' for i in range(11, 16)]].mean(axis=1)
            df_filtered['Lingkungan'] = df_filtered[[f'P{i}' for i in range(16, 21)]].mean(axis=1)

            profile = df_filtered.groupby('Cluster')[['Fasilitas', 'Kurikulum', 'Guru', 'Lingkungan']].mean()
            counts = df_filtered['Cluster'].value_counts().sort_index()

            tab1, tab2 = st.tabs(["Dashboard Analisis", "Laporan Eksekutif"])

            with tab1:
                st.subheader("Statistik Responden")
                cols = st.columns(len(counts))
                for i, (cls, count) in enumerate(counts.items()):
                    cols[i].metric(f"Klaster {cls}", f"{count} Siswa")

                if tampilkan_3d:
                    fig_3d = px.scatter_3d(df_filtered, x='PC1', y='PC2', z='PC3', color=df_filtered['Cluster'].astype(str))
                    fig_3d.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                    st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    fig_2d = px.scatter(df_filtered, x='PC1', y='PC2', color=df_filtered['Cluster'].astype(str))
                    fig_2d.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                    st.plotly_chart(fig_2d, use_container_width=True)

                st.markdown("---")
                html_btn = """
                <button onclick="window.parent.print()" style="background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.4); border-radius: 12px; color: white; padding: 12px; width: 100%; cursor: pointer;">🖨️ Cetak Dashboard ke PDF</button>
                """
                components.html(html_btn, height=60)
                st.download_button("Download Data (.csv)", df_filtered.to_csv(index=False).encode('utf-8'), "Data.csv", "text/csv")

            with tab2:
                st.header("Executive Summary")
                st.markdown(f"Laporan merangkum hasil analisis kepuasan dari **{len(df_filtered)} responden**.")
                # Ringkasan Eksekutif (Bisa lu kembangin sesuai kamus masalah di kode lama lu, sengaja disingkat biar ringan)
                st.dataframe(profile.style.highlight_min(axis=1, color='lightcoral').highlight_max(axis=1, color='lightgreen'))
