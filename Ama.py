import streamlit as st
import numpy as np
import os
from PIL import Image

# =========================================================
# KONFIGURASI HALAMAN & MODEL LITE
# =========================================================
st.set_page_config(page_title="Klasifikasi Dog vs Cat", page_icon="🐾", layout="centered")

MODEL_PATH = "model_pet_cnn.tflite"

@st.cache_resource
def load_tflite_model():
    """Memuat TFLite Runtime Interpreter dengan aman"""
    if os.path.exists(MODEL_PATH):
        try:
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
st.write("Aplikasi telah disesuaikan agar otomatis mengikuti dimensi model Anda.")

if interpreter is None:
    st.error(f"❌ File `{MODEL_PATH}` tidak ditemukan di repositori GitHub Anda!")
else:
    # Mengambil detail input otomatis dari model yang Anda buat
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Mendapatkan ukuran target asli (misal: 277x277) langsung dari model
    input_shape = input_details[0]['shape']  # Biasanya [1, 277, 277, 3]
    target_height = input_shape[1]
    target_width = input_shape[2]

    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img_display = Image.open(uploaded_file)
        st.image(img_display, caption="Gambar yang Diunggah", use_container_width=True)
        st.write("⏳ Sedang memproses...")

        try:
            # 1. Preprocessing Gambar disesuaikan dengan dimensi model secara dinamis
            img_resized = img_display.resize((target_width, target_height))
            img_array = np.array(img_resized, dtype=np.float32)
            
            # Normalisasi 1./255 seperti pada training notebook Anda
            img_array = img_array / 255.0  
            
            # Menambahkan dimensi batch agar shape menjadi [1, H, W, C]
            img_tensor = np.expand_dims(img_array, axis=0)

            # 2. Jalankan Prediksi
            interpreter.set_tensor(input_details[0]['index'], img_tensor)
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
            st.error(f"Terjadi kesalahan shape/tipe data saat prediksi: {e}")
