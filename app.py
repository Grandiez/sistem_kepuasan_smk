import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sistem Penilaian Kepuasan SMK", layout="wide")

st.title("📊 Analisis Kepuasan Siswa - SMK N 1 Dawuan")
st.markdown("Sistem analisis kepuasan siswa menggunakan metode **PCA & K-Means Clustering**.")

# --- SIDEBAR: UPLOAD ---
st.sidebar.header("1. Input Data")
uploaded_file = st.sidebar.file_uploader("Unggah file Excel/CSV dari Google Forms", type=["csv", "xlsx"])

if uploaded_file is not None:
    # --- PROSES 1: MEMBACA & MEMBERSIHKAN DATA ---
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        # STEP KRUSIAL: RENAMING KOLOM
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

    # --- PROSES 2: PCA & K-MEANS ---
    kolom_nilai = [f'P{i}' for i in range(1, 21)]
    
    # Pembersih Data Kosong Otomatis
    jml_awal = len(df)
    df = df.dropna(subset=kolom_nilai).reset_index(drop=True)
    jml_akhir = len(df)
    
    if jml_awal != jml_akhir:
        st.warning(f"⚠️ Ditemukan {jml_awal - jml_akhir} data kuesioner yang tidak lengkap (ada sel kosong). Data otomatis diabaikan agar sistem tidak error.")

    data_numeric = df[kolom_nilai]
    
    # Scaling & PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_numeric)

    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(scaled_data)
    
    # Koordinat X, Y, Z untuk grafik 3D
    df['PC1'] = pca_data[:, 0]
    df['PC2'] = pca_data[:, 1]
    df['PC3'] = pca_data[:, 2]

    # --- KONFIGURASI ALGORITMA ---
    st.sidebar.header("2. Konfigurasi Algoritma")
    gunakan_pca = st.sidebar.toggle("🟢 Aktifkan Reduksi PCA", value=True)
    
    if gunakan_pca:
        st.sidebar.success("PCA Aktif: 20 pertanyaan diperas jadi 3 Komponen (Sumbu X, Y, Z) untuk K-Means.")
        data_untuk_kmeans = pca_data 
    else:
        st.sidebar.error("PCA Mati: K-Means memproses 20 Dimensi Mentah!")
        data_untuk_kmeans = scaled_data 

    # Clustering K-Means
    n_clusters = st.sidebar.slider("Jumlah Klaster (K)", 2, 5, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(data_untuk_kmeans)

    # --- PROSES 3: FILTER TAMPILAN ---
    st.sidebar.header("3. Filter Data")
    
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
    ]

    if df_filtered.empty:
        st.warning("⚠️ Data tidak ditemukan dengan kombinasi filter tersebut.")
    else:
        # --- DASHBOARD UTAMA ---
        st.subheader("👥 Statistik Responden")
        cols = st.columns(len(df_filtered['Cluster'].unique()))
        counts = df_filtered['Cluster'].value_counts().sort_index()
        for i, (cls, count) in enumerate(counts.items()):
            cols[i].metric(f"Klaster {cls}", f"{count} Siswa")

        # Visualisasi 3D
        st.subheader("🌐 Peta Sebaran Siswa")
        fig = px.scatter_3d(
            df_filtered, x='PC1', y='PC2', z='PC3',
            color=df_filtered['Cluster'].astype(str),
            hover_data=['Nama', 'Kelas', 'Jurusan', 'Jenis_Kelamin'],
            title=f"Visualisasi 3D (Total: {len(df_filtered)} Siswa)",
            labels={'color': 'Klaster'}
        )
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

        # Profiling Dimensi
        df_filtered['Fasilitas'] = df_filtered[[f'P{i}' for i in range(1, 6)]].mean(axis=1)
        df_filtered['Kurikulum'] = df_filtered[[f'P{i}' for i in range(6, 11)]].mean(axis=1)
        df_filtered['Guru'] = df_filtered[[f'P{i}' for i in range(11, 16)]].mean(axis=1)
        df_filtered['Lingkungan'] = df_filtered[[f'P{i}' for i in range(16, 21)]].mean(axis=1)

        profile = df_filtered.groupby('Cluster')[['Fasilitas', 'Kurikulum', 'Guru', 'Lingkungan']].mean()
        
        # Bar Chart Grouped
        st.subheader("📈 Perbandingan Skor Rata-rata")
        profile_reset = profile.reset_index()
        profile_reset['Label_Klaster'] = profile_reset['Cluster'].apply(lambda x: f"Klaster {x}<br>({counts[x]} Siswa)")
        
        fig_bar = px.bar(
            profile_reset, x='Label_Klaster', 
            y=['Fasilitas', 'Kurikulum', 'Guru', 'Lingkungan'],
            barmode='group',
            labels={'value': 'Skor (1-5)', 'variable': 'Aspek', 'Label_Klaster': 'Kelompok (Klaster)'}
        )
        fig_bar.update_layout(legend_title_text='Dimensi Layanan', xaxis_title="Kelompok (Klaster) & Jumlah Siswa")
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- KAMUS MASALAH ---
        kamus_masalah = {
            'P1': 'Fasilitas pendukung (lab, perpus, alat praktik) dirasa kurang lengkap saat dibutuhkan',
            'P2': 'Kondisi peralatan dan sarana belajar banyak yang rusak atau tertinggal (kurang modern)',
            'P3': 'Area sekolah dirasa kurang memberikan rasa aman dan nyaman',
            'P4': 'Lokasi fasilitas (kantin, toilet) kurang memadai atau sulit dijangkau',
            'P5': 'Fasilitas yang ada belum maksimal membantu efektivitas belajar siswa',
            'P6': 'Materi yang diajarkan dirasa kurang relevan dengan kebutuhan industri saat ini',
            'P7': 'Siswa kesulitan memahami materi pelajaran yang disampaikan',
            'P8': 'Kurikulum dirasa belum sesuai dengan minat dan bakat mayoritas siswa',
            'P9': 'Metode pembelajaran terasa membosankan dan kurang bervariasi',
            'P10': 'Siswa merasa materi yang dipelajari kurang memberikan manfaat bagi masa depannya',
            'P11': 'Guru dirasa kurang menguasai materi secara mendalam di bidangnya',
            'P12': 'Penjelasan guru sering berbelit-belit dan sulit dimengerti',
            'P13': 'Guru dirasa kurang memberikan teladan yang baik (sikap, disiplin, etika)',
            'P14': 'Guru kurang terbuka atau sulit dihubungi saat siswa mengalami kesulitan',
            'P15': 'Guru sering terlambat, jam kosong, atau kurang profesional dalam mengajar',
            'P16': 'Kebersihan dan kerapian lingkungan sekolah kurang terjaga dengan baik',
            'P17': 'Lingkungan sekolah dirasa kurang aman dan sering ada gangguan',
            'P18': 'Suasana kelas kurang kondusif sehingga mengganggu konsentrasi belajar',
            'P19': 'Hubungan sosial antar warga sekolah (guru, siswa, staf) kurang harmonis',
            'P20': 'Budaya sopan santun belum diterapkan secara nyata dalam keseharian'
        }

        # --- KAMUS SOLUSI ---
        kamus_solusi = {
            'P1': 'Lakukan inventarisasi lab/bengkel. Ajukan pengadaan alat praktik yang kurang ke yayasan/Dinas, atau buat jadwal penggunaan silang antar kelas.',
            'P2': 'Jadwalkan maintenance rutin untuk PC lab atau mesin bengkel. Perbaiki segera alat yang rusak dan sisihkan anggaran untuk pembaruan teknologi.',
            'P3': 'Tingkatkan patroli keamanan sekolah, pasang CCTV di titik rawan, dan pastikan pencahayaan serta sirkulasi udara di area sekolah memadai.',
            'P4': 'Buat penunjuk arah yang jelas, tambah jumlah toilet jika rasio siswa tidak seimbang, dan atur ulang layout kantin agar tidak berdesakan.',
            'P5': 'Evaluasi kembali kesesuaian modul ajar dengan alat yang ada. Berikan pelatihan/orientasi penggunaan alat praktik baru kepada siswa.',
            'P6': 'Undang praktisi industri (guru tamu) secara rutin dan sinkronisasi silabus dengan mitra Dudika (Dunia Usaha Dunia Industri).',
            'P7': 'Adakan program remedial atau tutor sebaya. Dorong guru untuk menyederhanakan modul ajar menjadi lebih visual dan lebih banyak praktik.',
            'P8': 'Sediakan layanan konseling karir dari guru BK yang lebih intensif, dan perbanyak pilihan ekstrakurikuler peminatan keahlian.',
            'P9': 'Adakan in-house training pedagogik bagi guru untuk menggunakan metode interaktif (kuis digital, project-based learning, simulasi).',
            'P10': 'Sering adakan seminar karir atau kunjungan industri (industrial visit) agar siswa melihat langsung prospek kerja dari jurusan mereka.',
            'P11': 'Fasilitasi guru produktif untuk mengikuti diklat, sertifikasi profesi, atau program magang industri agar ilmunya terus up-to-date.',
            'P12': 'Lakukan evaluasi peer-teaching antar guru dan kumpulkan feedback anonim dari siswa tiap pertengahan semester terkait cara mengajar.',
            'P13': 'Tegakkan kode etik guru dengan tegas oleh Kepala Sekolah/Yayasan, dan berikan penghargaan bagi guru teladan sebagai motivasi.',
            'P14': 'Wajibkan guru menyisihkan waktu untuk jam konsultasi di luar jam pelajaran, dan latih guru memiliki pendekatan empati kepada siswa.',
            'P15': 'Terapkan sistem presensi digital terintegrasi untuk guru dan berikan teguran tertulis bagi guru yang sering terlambat/jam kosong.',
            'P16': 'Galakkan kembali program Jumat Bersih, perbanyak tempat sampah terpilah, dan evaluasi kinerja petugas kebersihan (OB) sekolah.',
            'P17': 'Tindak tegas secara kedisiplinan (SP) terhadap kasus perundungan (bullying) antar siswa, dan perketat akses masuk gerbang sekolah.',
            'P18': 'Perbaiki fasilitas kelas yang rusak (kipas/AC mati, lampu redup) dan wajibkan wali kelas membuat kesepakatan tata tertib kelas yang tegas.',
            'P19': 'Adakan acara kebersamaan lintas kelas/jurusan (class meeting) dan mediasi segera jika ada indikasi kubu-kubuan antar siswa.',
            'P20': 'Kampanyekan kembali budaya 5S (Senyum, Salam, Sapa, Sopan, Santun) secara masif, dan jadikan guru serta staf sebagai role model utama.'
        }

        dimensi_map = {
            'Fasilitas': [f'P{i}' for i in range(1, 6)],
            'Kurikulum': [f'P{i}' for i in range(6, 11)],
            'Guru': [f'P{i}' for i in range(11, 16)],
            'Lingkungan': [f'P{i}' for i in range(16, 21)]
        }

# --- REKOMENDASI MANAJERIAL & PROFILING KELAS/JURUSAN ---
        st.subheader("💡 Rekomendasi & Target Investigasi")
        
        for cluster in profile.index:
            dimensi_terendah = profile.loc[cluster].idxmin()
            skor_dimensi = profile.loc[cluster].min()
            
            cols_dimensi = dimensi_map[dimensi_terendah]
            df_cluster = df_filtered[df_filtered['Cluster'] == cluster]
            
            rata_item = df_cluster[cols_dimensi].mean()
            item_code = rata_item.idxmin()
            skor_item = rata_item.min()
            
            isi_masalah = kamus_masalah[item_code] 
            solusi_masalah = kamus_solusi[item_code] 

            kelas_terbanyak = int(df_cluster['Kelas'].mode()[0])
            jml_kelas = len(df_cluster[df_cluster['Kelas'] == kelas_terbanyak])
            jurusan_terbanyak = df_cluster['Jurusan'].mode()[0]
            jml_jurusan = len(df_cluster[df_cluster['Jurusan'] == jurusan_terbanyak])

            # --- LOGIKA RENTANG SKALA LIKERT BARU ---
            if skor_item >= 4.0:
                status = "Sangat Memuaskan 🌟"
                warna_alert = st.success
            elif skor_item >= 3.0:
                status = "Cukup Memuaskan (Perlu Peningkatan) 🔵"
                warna_alert = st.info
            elif skor_item >= 2.0:
                status = "Kurang Memuaskan (Perlu Perbaikan) 🟠"
                warna_alert = st.warning
            else:
                status = "Sangat Kurang / Kritis (Perbaikan Segera!) 🔴"
                warna_alert = st.error

            # --- TAMPILAN BERDASARKAN STATUS KEPARAHAN ---
            if skor_item >= 4.0:
                # JIKA SUDAH PUAS (Gak perlu dicari tersangkanya)
                pesan = (
                    f"**Klaster {cluster} (Status: {status})**\n\n"
                    f"✨ **Kondisi Aman:** Secara umum siswa di klaster ini sudah **SANGAT PUAS**. Aspek terendah ada pada dimensi **{dimensi_terendah}** (potensi: *{isi_masalah}*), namun skornya masih sangat aman di angka **{skor_item:.2f}/5.00**.\n\n"
                    f"📈 **Saran:** Pertahankan kinerja saat ini. Sebagai tindakan preventif: {solusi_masalah}"
                )
                warna_alert(pesan)
                
                # Menu lipat khusus yang sudah puas
                with st.expander(f"📋 Klik untuk melihat Daftar Mayoritas Siswa di Klaster {cluster} (Tidak Butuh Investigasi)"):
                    st.write("Karena klaster ini bersatus 'Sangat Memuaskan', Anda tidak perlu melakukan investigasi mendalam.")
            
            else:
                # JIKA ADA MASALAH (Skor di bawah 4.0)
                pesan = (
                    f"**Klaster {cluster} (Status: {status} - Prioritas Perbaikan: {dimensi_terendah})**\n\n"
                    f"👉 **Akar Masalah:** *{isi_masalah}* (Skor Kepuasan: **{skor_item:.2f}/5.00**).\n\n"
                    f"🛠️ **Rekomendasi Solusi:** {solusi_masalah}\n\n"
                    f"🎯 **Target Investigasi:** Mayoritas keluhan berasal dari **Kelas {kelas_terbanyak}** ({jml_kelas} siswa) dan **Jurusan {jurusan_terbanyak}** ({jml_jurusan} siswa).\n"
                )
                warna_alert(pesan)
                
                # Menu lipat target investigasi (tetap sama)
                with st.expander(f"📋 Klik untuk melihat Daftar Nama Target Investigasi (Klaster {cluster})"):
                    df_kelas = df_cluster[df_cluster['Kelas'] == kelas_terbanyak][['Nama', 'Kelas', 'Jurusan', 'Jenis_Kelamin']].reset_index(drop=True)
                    df_kelas.index = df_kelas.index + 1 
                    
                    st.markdown(f"**🎯 1. Daftar {jml_kelas} Siswa dari Kelas {kelas_terbanyak}:**")
                    st.dataframe(df_kelas, use_container_width=True)
                    
                    st.markdown("---") 
                    
                    df_jurusan = df_cluster[df_cluster['Jurusan'] == jurusan_terbanyak][['Nama', 'Kelas', 'Jurusan', 'Jenis_Kelamin']].reset_index(drop=True)
                    df_jurusan.index = df_jurusan.index + 1 
                    
                    st.markdown(f"**🎯 2. Daftar {jml_jurusan} Siswa dari Jurusan {jurusan_terbanyak}:**")
                    st.dataframe(df_jurusan, use_container_width=True)

        # Download Button
        st.markdown("---")
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Laporan Excel", csv, "Laporan_Analisis_SMK.csv", "text/csv")

else:
    st.info("Silakan unggah file Excel (.xlsx) atau CSV hasil Google Forms.")