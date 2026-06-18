import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import Image
import gdown

# =========================================================
# TRIK BYPASS ERROR DESERIALISASI (UNTUK VERSI TENSORFLOW JADUL)
# =========================================================
# Fungsi ini memaksa Keras mengabaikan parameter quantization_config yang bikin crash
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="Custom")
class CustomDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)  # Hapus parameter penyebab error
        super().__init__(*args, **kwargs)

# =========================================================
# KONFIGURASI HALAMAN & MODEL
# =========================================================
st.set_page_config(page_title="Klasifikasi Dog vs Cat", page_icon="🐾", layout="centered")

MODEL_PATH = "model_pet_cnn.keras"
IMG_SIZE = (277, 277)

# Ganti teks di bawah ini dengan ID file Google Drive kamu
DRIVE_FILE_ID = "1EyqQzeh3nK0jlQ8y3ej2Is1EP3S4PNz5" 

@st.cache_resource
def load_cnn_model():
    """Fungsi mendownload model dari Drive dan memuatnya menggunakan Custom Objects"""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Mendownload file model dari Google Drive (ini hanya dilakukan sekali)..."):
            url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
            try:
                gdown.download(url, MODEL_PATH, quiet=False)
            except Exception as e:
                st.error(f"❌ Gagal mendownload model dari Google Drive: {e}")
                return None
                
    if os.path.exists(MODEL_PATH):
        try:
            # Muat model dengan menyisipkan CustomDense untuk menggantikan layer Dense bawaan yang error
            return load_model(MODEL_PATH, custom_objects={'Dense': CustomDense})
        except Exception as e:
            st.error(f"❌ Gagal memuat file model: {e}")
            return None
    else:
        return None

# Load model secara otomatis
model = load_cnn_model()

# =========================================================
# ANTARMUKA PENGGUNA (UI)
# =========================================================
st.title("🐾 Klasifikasi Gambar: Kucing vs Anjing")
st.write("Unggah gambar kucing atau anjing untuk mengetahui hasil prediksi beserta tingkat keyakinannya (confidence).")

if model is None:
    st.error(f"❌ Koneksi model gagal! Pastikan ID Google Drive benar dan akses file disetel ke 'Siapa saja yang memiliki link'.")
else:
    # Komponen Unggah Gambar
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Menampilkan gambar yang diunggah
        img_display = Image.open(uploaded_file)
        st.image(img_display, caption="Gambar yang Diunggah", use_container_width=True)
        
        # Menggunakan st.spinner agar transisi visual lebih halus dan profesional
        with st.spinner("⏳ Sedang memproses dan memprediksi..."):
            try:
                # Preprocessing Gambar sesuai dengan spesifikasi model
                # 1. Konversi ke RGB untuk mengantisipasi gambar PNG transparan (RGBA)
                img_rgb = img_display.convert('RGB')
                
                # 2. Resize ke 277x277
                img_resized = img_rgb.resize(IMG_SIZE)
                
                # 3. Konversi ke array & normalisasi (rescale 1./255)
                img_array = image.img_to_array(img_resized)
                img_array = img_array / 255.0
                
                # 4. Menambahkan dimensi batch (expand dims)
                img_array = np.expand_dims(img_array, axis=0)

                # Melakukan Prediksi
                prediction = model.predict(img_array)
                confidence = prediction[0][0]

                # Menampilkan Hasil Klasifikasi
                st.subheader("📊 Hasil Prediksi:")
                
                # Asumsi: output model mendekati 1 untuk Anjing dan 0 untuk Kucing
                if confidence > 0.5:
                    st.success("### **Prediksi : ANJING (DOG)**")
                    st.info(f"**Confidence : {confidence * 100:.2f}%**")
                else:
                    st.success("### **Prediksi : KUCING (CAT)**")
                    st.info(f"**Confidence : {(1 - confidence) * 100:.2f}%**")
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
