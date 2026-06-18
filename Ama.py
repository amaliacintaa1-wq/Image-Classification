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

LOCAL_MODEL_PATH = "model_pet_cnn.keras"

# GANTI LINK DI BAWAH INI dengan link hasil Copy Link Address dari GitHub Release Anda tadi!
MODEL_URL = "https://github.com/USERNAME/NAMA_REPO/releases/download/v1.0.0/model_pet_cnn.keras"

IMG_SIZE = (277, 277)

@st.cache_resource
def load_cnn_model():
    """Mengunduh model langsung dari URL publik yang stabil (GitHub/HuggingFace)"""
    if not os.path.exists(LOCAL_MODEL_PATH):
        with st.spinner("⏳ Sedang mengunduh file model untuk pertama kali... Mohon tunggu."):
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                
                with open(LOCAL_MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                st.success("✅ Model berhasil diunduh secara lokal!")
            except Exception as e:
                st.error(f"❌ Gagal mengunduh file model: {e}")
                return None
                
    try:
        return load_model(LOCAL_MODEL_PATH)
    except Exception as e:
        st.error(f"❌ File model rusak saat dimuat. Menghapus cache agar bisa di-download ulang saat halaman di-refresh. Detail Error: {e}")
        if os.path.exists(LOCAL_MODEL_PATH):
            os.remove(LOCAL_MODEL_PATH)
        return None

# Load model
model = load_cnn_model()

# =========================================================
# ANTARMUKA PENGGUNA (UI)
# =========================================================
st.title("🐾 Klasifikasi Gambar: Kucing vs Anjing")
st.write("Unggah gambar kucing atau anjing untuk mengetahui hasil prediksi beserta tingkat keyakinannya (confidence).")

if model is None:
    st.error("❌ Model gagal dimuat. Pastikan URL model pada kode Anda sudah benar dan aktif.")
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
