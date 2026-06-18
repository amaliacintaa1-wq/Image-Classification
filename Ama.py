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

# Model dibaca langsung secara lokal dari folder repository GitHub Anda
MODEL_PATH = "model_pet_cnn.keras"
IMG_SIZE = (277, 277)

@st.cache_resource
def load_cnn_model():
    """Fungsi mendeteksi dan memuat model lokal di server Streamlit"""
    if os.path.exists(MODEL_PATH):
        try:
            return load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"❌ Gagal memuat file model lokal: {e}")
            return None
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
    st.error(f"❌ File `{MODEL_PATH}` tidak ditemukan di repositori GitHub Anda! Pastikan file model sudah di-upload menggunakan Git LFS.")
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
