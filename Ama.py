import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import Image
import gdown  # Menggunakan gdown sebagai pengganti urllib

# =========================================================
# KONFIGURASI HALAMAN & MODEL
# =========================================================
st.set_page_config(page_title="Klasifikasi Dog vs Cat", page_icon="🐾", layout="centered")

MODEL_PATH = "model_pet_cnn.h5"  
IMG_SIZE = (277, 277)

@st.cache_resource
def load_cnn_model():
    """Fungsi untuk mendownload model menggunakan gdown jika file belum ada atau rusak"""
    # Jika file tidak ada ATAU ukurannya terlalu kecil (berarti corrupt/hanya teks HTML)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100000:
        with st.spinner("Mengunduh file model .h5 dari Google Drive... Mohon tunggu sebentar..."):
            DRIVE_ID = "1EyqQzeh3nK0jlQ8y3ej2Is1EP3S4PNz5" 
            URL = f"https://drive.google.com/uc?id={DRIVE_ID}"
            
            try:
                # Menghapus file lama jika ada yang corrupt sebelumnya
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
                    
                # Download menggunakan gdown yang dijamin aman untuk file besar
                gdown.download(URL, MODEL_PATH, quiet=False)
                st.success("Model .h5 berhasil diunduh dengan sempurna!")
            except Exception as e:
                st.error(f"Gagal mengunduh model dari Drive: {e}")
                return None
                
    if os.path.exists(MODEL_PATH):
        try:
            return load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Gagal memuat model. File kemungkinan rusak. Mencoba download ulang pada refresh berikutnya. Error: {e}")
            # Hapus file rusak agar di-download ulang saat direfresh
            os.remove(MODEL_PATH) 
            return None
    return None

# Load model
model = load_cnn_model()

# =========================================================
# ANTARMUKA PENGGUNA (UI)
# =========================================================
st.title("🐾 Klasifikasi Gambar: Kucing vs Anjing")
st.write("Unggah gambar kucing atau anjing untuk mengetahui hasil prediksi beserta tingkat keyakinannya (confidence).")

if model is None:
    st.error(f"❌ File model gagal dimuat! Pastikan aplikasi di-refresh kembali.")
else:
    # Komponen Unggah Gambar
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Menampilkan gambar yang diunggah
        img_display = Image.open(uploaded_file)
        st.image(img_display, caption="Gambar yang Diunggah", use_container_width=True)
        
        st.write("⏳ Sedang memproses dan memprediksi...")

        try:
            # Preprocessing Gambar sesuai dengan spesifikasi model
            img_resized = img_display.resize(IMG_SIZE)
            img_array = image.img_to_array(img_resized)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Melakukan Prediksi
            prediction = model.predict(img_array)
            confidence = prediction[0][0]

            # Menampilkan Hasil Klasifikasi
            st.subheader("📊 Hasil Prediksi:")
            
            if confidence > 0.5:
                st.success(f"### **Prediksi : ANJING (DOG)**")
                st.info(f"**Confidence : {confidence * 100:.2f}%**")
            else:
                st.success(f"### **Prediksi : KUCING (CAT)**")
                st.info(f"**Confidence : {(1 - confidence) * 100:.2f}%**")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
