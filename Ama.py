import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
import requests
from PIL import Image

# =========================================================
# KONFIGURASI HALAMAN & MODEL
# =========================================================
st.set_page_config(page_title="Klasifikasi Dog vs Cat", page_icon="🐾", layout="centered")

# Path lokal tempat menyimpan model setelah diunduh
LOCAL_MODEL_PATH = "model_pet_cnn.keras"

# URL download langsung (Direct Link) untuk Google Drive Anda
# Catatan: Struktur URL diubah dari '/view?usp=...' menjadi '/uc?export=download&id=...'
DRIVE_MODEL_URL = "https://docs.google.com/uc?export=download&id=1EyqQzeh3nK0jlQ8y3ej2Is1EP3S4PNz5"

IMG_SIZE = (277, 277)

@st.cache_resource
def load_cnn_model():
    """Fungsi untuk mengunduh model dari Drive jika belum ada, lalu memuatnya dengan cache"""
    # 1. Cek apakah file sudah diunduh sebelumnya secara lokal
    if not os.path.exists(LOCAL_MODEL_PATH):
        with st.spinner("⏳ Mengunduh file model dari Google Drive... Harap tunggu beberapa saat."):
            try:
                response = requests.get(DRIVE_MODEL_URL, stream=True)
                response.raise_for_status()  # Memastikan tidak ada error HTTP
                
                with open(LOCAL_MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                st.success("✅ Model berhasil diunduh!")
            except Exception as e:
                st.error(f"❌ Gagal mengunduh model dari Google Drive: {e}")
                return None
                
    # 2. Jika file sudah ada di lokal, muat ke dalam aplikasi
    try:
        return load_model(LOCAL_MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Gagal memuat file model: {e}")
        return None

# Load model
model = load_cnn_model()

# =========================================================
# ANTARMUKA PENGGUNA (UI)
# =========================================================
st.title("🐾 Klasifikasi Gambar: Kucing vs Anjing")
st.write("Unggah gambar kucing atau anjing untuk mengetahui hasil prediksi beserta tingkat keyakinannya (confidence).")

if model is None:
    st.error(f"❌ Model tidak dapat dimuat. Pastikan tautan Google Drive Anda valid dan diatur ke status 'Siapa saja yang memiliki link dapat melihat' (Public Access).")
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
            # 1. Resize ke 277x277
            img_resized = img_display.resize(IMG_SIZE)
            # 2. Konversi ke array & normalisasi (rescale 1./255)
            img_array = image.img_to_array(img_resized)
            img_array = img_array / 255.0
            # 3. Menambahkan dimensi batch (expand dims)
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
