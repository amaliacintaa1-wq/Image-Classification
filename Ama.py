@st.cache_resource
def load_cnn_model():
    """Fungsi untuk mengunduh model besar dari Google Drive dengan menembus konfirmasi virus"""
    if not os.path.exists(LOCAL_MODEL_PATH):
        with st.spinner("⏳ Mengunduh file model besar dari Google Drive... Harap tunggu (bisa memakan waktu beberapa menit)."):
            try:
                # Menggunakan session agar cookies konfirmasi virus tersimpan
                session = requests.Session()
                
                # Request pertama untuk memicu halaman konfirmasi virus
                # Ambil ID file dari URL asli Anda
                file_id = "1EyqQzeh3nK0jlQ8y3ej2Is1EP3S4PNz5"
                download_url = "https://docs.google.com/uc?export=download"
                
                response = session.get(download_url, params={'id': file_id}, stream=True)
                
                # Fungsi internal untuk mencari token konfirmasi di halaman HTML Google
                def get_confirm_token(res):
                    for key, value in res.cookies.items():
                        if key.startswith('download_warning'):
                            return value
                    for line in res.iter_lines():
                        line = line.decode('utf-8')
                        if 'download_warning' in line:
                            # Ekstrak token jika ada di text HTML
                            import re
                            match = re.search(r'confirm=([A-Za-z0-9_]+)', line)
                            if match:
                                return match.group(1)
                    return None

                token = get_confirm_token(response)
                
                # Jika token ditemukan, lakukan request kedua dengan menyertakan token konfirmasi
                if token:
                    params = {'id': file_id, 'confirm': token}
                    response = session.get(download_url, params=params, stream=True)
                
                response.raise_for_status()
                
                # Mulai menulis file asli ke lokal
                with open(LOCAL_MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            
                st.success("✅ Model besar berhasil diunduh dan disimpan!")
            except Exception as e:
                st.error(f"❌ Gagal bypass proteksi Google Drive: {e}")
                return None
                
    # 2. Muat model dari lokal jika sudah berhasil diunduh
    try:
        return load_model(LOCAL_MODEL_PATH)
    except Exception as e:
        st.error(f"❌ File hasil download rusak atau tidak lengkap. Gagal memuat model: {e}")
        # Jika gagal, hapus file yang rusak agar di-download ulang saat refresh
        if os.path.exists(LOCAL_MODEL_PATH):
            os.remove(LOCAL_MODEL_PATH)
        return None
