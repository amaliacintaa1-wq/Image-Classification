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

MODEL_PATH = "model_pet_cnn.keras"
IMG_SIZE = (277, 277)

# Penyelamat dari error 'quantization_config' di Streamlit
class CustomDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

@st.cache_resource
def load_cnn_model():
    """Mendownload model dari Google Drive jika belum ada di server Streamlit"""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Mengunduh file model dari cloud... Mohon tunggu sebentar..."):
            # ⚠️ GANTI TEKS DI BAWAH INI DENGAN ID FILE GOOGLE DRIVE KAMU
            DRIVE_ID = "1EyqQzeh3nK0jlQ8y3ej2Is1EP3S4PNz5" 
            
            URL = f"https://docs.google.com/uc?export=download&id={DRIVE_ID}"
            try:
                urllib.request.urlretrieve(URL, MODEL_PATH)
                st.success("Model berhasil diunduh!")
            except Exception as e:
                st.error(f"Gagal mengunduh model: {e}")
                return None
                
    # Memuat model menggunakan CustomDense agar tidak crash
    return load_model(MODEL_PATH, custom_objects={'Dense': CustomDense})

# Load model
model = load_cnn_model()

# =========================================================
# ANTARMUKA PENGGUNA (UI)
# =========================================================
st.title("🐾 Klasifikasi Gambar: Kucing vs Anjing")
st.write("Unggah gambar kucing atau anjing untuk mengetahui hasil prediksi beserta tingkat keyakinannya (confidence).")

if model is None:
    st.error("❌ Model gagal dimuat. Pastikan ID Google Drive benar dan file diset ke 'Anyone with the link'.")
else:
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img_display = Image.open(uploaded_file)
        st.image(img_display, caption="Gambar yang Diunggah", use_container_width=True)
        
        st.write("⏳ Sedang memproses dan memprediksi...")

        try:
            img_resized = img_display.resize(IMG_SIZE)
            img_array = image.img_to_array(img_resized)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            confidence = prediction[0][0]

            st.subheader("📊 Hasil Prediksi:")
            
            if confidence > 0.5:
                st.success(f"### **Prediksi : ANJING (DOG)**")
                st.info(f"**Confidence : {confidence * 100:.2f}%**")
            else:
                st.success(f"### **Prediksi : KUCING (CAT)**")
                st.info(f"**Confidence : {(1 - confidence) * 100:.2f}%**")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
