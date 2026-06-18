import streamlit as st
import numpy as np
import os
from PIL import Image

# =========================================================
# KONFIGURASI HALAMAN & MODEL LITE
# =========================================================
st.set_page_config(page_title="Klasifikasi Dog vs Cat", page_icon="🐾", layout="centered")

# Menggunakan model tflite yang jauh lebih ringan untuk server cloud
MODEL_PATH = "model_pet_cnn.tflite"
IMG_SIZE = (277, 277)

@st.cache_resource
def load_tflite_model():
    """Memuat TFLite Runtime Interpreter secara lokal"""
    if os.path.exists(MODEL_PATH):
        try:
            # Import tflite runtime di dalam fungsi agar menghemat memori saat startup
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            st.error(f"❌ Gagal memuat file TFLite: {e}")
            return None
    return None

interpreter = load_tflite_model()

# =========================================================
# ANTARMUKA PENGGUNA (UI)
# =========================================================
st.title("🐾 Klasifikasi Gambar: Kucing vs Anjing (Lite)")
st.write("Aplikasi ini menggunakan model terkompresi agar berjalan lancar di server cloud.")

if interpreter is None:
    st.error(f"❌ File `{MODEL_PATH}` tidak ditemukan di repositori GitHub Anda!")
else:
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img_display = Image.open(uploaded_file)
        st.image(img_display, caption="Gambar yang Diunggah", use_container_width=True)
        st.write("⏳ Sedang memproses...")

        try:
            # 1. Preprocessing Gambar secara manual (tanpa fungsi tf.keras.preprocessing)
            img_resized = img_display.resize(IMG_SIZE)
            img_array = np.array(img_resized, dtype=np.float32)
            img_array = img_array / 255.0  # Normalisasi
            img_array = np.expand_dims(img_array, axis=0)  # Tambah dimensi batch

            # 2. Jalankan Prediksi menggunakan TFLite Interpreter
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            
            prediction = interpreter.get_tensor(output_details[0]['index'])
            confidence = prediction[0][0]

            # 3. Tampilkan Hasil
            st.subheader("📊 Hasil Prediksi:")
            if confidence > 0.5:
                st.success(f"### **Prediksi : ANJING (DOG)**")
                st.info(f"**Confidence : {confidence * 100:.2f}%**")
            else:
                st.success(f"### **Prediksi : KUCING (CAT)**")
                st.info(f"**Confidence : {(1 - confidence) * 100:.2f}%**")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
