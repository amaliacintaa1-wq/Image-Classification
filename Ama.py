import streamlit as st
import numpy as np
import os
from PIL import Image
import tflite_runtime.interpreter as tflite

# =========================================================
# KONFIGURASI HALAMAN & MODEL LITE (ULTRA LIGHTWEIGHT)
# =========================================================
st.set_page_config(page_title="Klasifikasi Dog vs Cat", page_icon="🐾", layout="centered")

MODEL_PATH = "model_pet_cnn.tflite"

@st.cache_resource
def load_tflite_model():
    """Memuat TFLite Runtime Interpreter tanpa library TensorFlow utuh"""
    if os.path.exists(MODEL_PATH):
        try:
            # Memakai tflite_runtime yang sangat hemat RAM
            interpreter = tflite.Interpreter(model_path=MODEL_PATH)
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
st.title("🐾 Klasifikasi Gambar: Kucing vs Anjing (Lite Runtime)")
st.write("Aplikasi menggunakan tflite-runtime agar super ringan dan anti-crash di server cloud.")

if interpreter is None:
    st.error(f"❌ File `{MODEL_PATH}` tidak ditemukan di repositori GitHub Anda!")
else:
    # Mengambil detail input otomatis dari model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Mendapatkan ukuran target asli (277, 277) dari model Anda
    input_shape = input_details[0]['shape']  # [1, 277, 277, 3]
    target_height = input_shape[1]
    target_width = input_shape[2]

    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img_display = Image.open(uploaded_file).convert('RGB')
        st.image(img_display, caption="Gambar yang Diunggah", use_container_width=True)
        st.write("⏳ Sedang memproses...")

        try:
            # 1. Preprocessing Gambar (Resize sesuai target model)
            img_resized = img_display.resize((target_width, target_height))
            img_array = np.array(img_resized, dtype=np.float32)
            
            # Normalisasi 1./255 seperti pada training notebook Anda
            img_array = img_array / 255.0  
            
            # Menambahkan dimensi batch agar shape menjadi [1, H, W, C]
            img_tensor = np.expand_dims(img_array, axis=0)

            # 2. Jalankan Prediksi
            interpreter.set_tensor(input_details[0]['index'], img_tensor)
            interpreter.invoke()
            
            # Mengambil hasil akhir dan mengubahnya ke float desimal tunggal
            prediction = interpreter.get_tensor(output_details[0]['index'])
            confidence = float(np.squeeze(prediction))

            # 3. Tampilkan Hasil Klasifikasi sesuai indeks ({'cat': 0, 'dog': 1})
            st.subheader("📊 Hasil Prediksi:")
            if confidence > 0.5:
                st.success(f"### **Prediksi : ANJING (DOG)**")
                st.info(f"**Confidence : {confidence * 100:.2f}%**")
            else:
                st.success(f"### **Prediksi : KUCING (CAT)**")
                st.info(f"**Confidence : {(1.0 - confidence) * 100:.2f}%**")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data/prediksi: {e}")
