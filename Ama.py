import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import Image

# =========================================================
# KONFIGURASI HALAMAN & MODEL
# =========================================================
st.set_page_config(page_title="Klasifikasi Dog vs Cat", page_icon="🐾", layout="centered")

MODEL_PATH = "https://drive.google.com/file/d/1EyqQzeh3nK0jlQ8y3ej2Is1EP3S4PNz5/view?usp=drive_link"  # Sesuaikan dengan lokasi model Anda
IMG_SIZE = (277, 277)

@st.cache_resource
def load_cnn_model():
    """Fungsi untuk memuat model dengan cache agar tidak di-load berulang kali"""
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
    st.error(f"❌ File model `{MODEL_PATH}` tidak ditemukan! Harap latih model terlebih dahulu atau pastikan file berada di direktori yang sama dengan `app.py`.")
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
