import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import Image
import urllib.request

# =========================================================
# KONFIGURASI HALAMAN & MODEL
# =========================================================
st.set_page_config(page_title="Klasifikasi Dog vs Cat", page_icon="🐾", layout="centered")

# Menggunakan format .h5 agar aman dari error versi Keras
MODEL_PATH = "model_pet_cnn.h5"  
IMG_SIZE = (277, 277)

@st.cache_resource
def load_cnn_model():
    """Fungsi untuk mendownload model .h5 dari Google Drive jika belum ada di server"""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Mengunduh file model .h5 dari cloud... Mohon tunggu sebentar..."):
            # ⚠️ GANTI TEKS DI BAWAH INI DENGAN ID FILE GOOGLE DRIVE KAMU YANG BARU (.h5)
            DRIVE_ID = "PASTE_ID_FILE_GOOGLE_DRIVE_KAMU_DI_SINI" 
            
            URL = f"https://docs.google.com/uc?export=download&id={DRIVE_ID}"
            try:
                urllib.request.urlretrieve(URL, MODEL_PATH)
                st.success("Model .h5 berhasil diunduh!")
            except Exception as e:
                st.error(f"Gagal mengunduh model dari Drive: {e}")
                return None
                
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    else:
        return None

# Load model
model = load_cnn_model()

# =========================================================
# ANTARMUKA PENGGUNA (UI)
# =========================================================
st.title("🐾 Klasifikasi Gambar: Kucing vs Anjing")
st.write("Unggah gambar kucing atau anjing untuk mengetahui hasil prediksi beserta tingkat keyakinannya (confidence).")

if model is None:
    st.error(f"❌ File model gagal dimuat! Pastikan ID Google Drive benar dan akses file di-set ke 'Anyone with the link'.")
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
