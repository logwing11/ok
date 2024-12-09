import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import streamlit_webrtc as webrtc
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2

# Load Model dalam format .keras
model = load_model("nasnet_model.keras")

# Mendefinisikan class names (disesuaikan dengan training dataset)
class_names = ["Biodegradable", "Non-Biodegradable"]


# Fungsi untuk Prediksi Gambar
def predict_image(img):
    img = img.resize((224, 224))  # Resize gambar sesuai input model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalisasi
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[
        0
    ]  # Mengambil indeks kelas dengan probabilitas tertinggi
    confidence = prediction[0][predicted_class] * 100  # Mengambil confidence score
    return class_names[predicted_class], confidence


# Kelas untuk mengubah frame video dari kamera
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Mengambil gambar dari frame kamera
        img = frame.to_image()

        # Melakukan prediksi pada gambar
        label, confidence = predict_image(img)

        # Konversi frame menjadi numpy array untuk memodifikasi
        frame_np = frame.to_ndarray(format="bgr24")

        # Menambahkan teks ke frame dengan hasil prediksi
        cv2.putText(
            frame_np,
            f"{label}: {confidence:.2f}%",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Mengembalikan frame dengan teks yang sudah ditambahkan
        return frame_np


def show_ns():
    # Streamlit Interface
    st.title("Klasifikasi Biodegradable dan Non-Biodegradable")
    st.write(
        "Upload gambar atau gunakan kamera untuk memprediksi apakah gambar termasuk **Biodegradable** atau **Non-Biodegradable**."
    )

    # Menu Pilihan
    menu = st.selectbox(
        "Pilih metode klasifikasi:", ["Upload Gambar", "Gunakan Kamera"]
    )

    if menu == "Upload Gambar":
        # File Uploader untuk mengunggah gambar
        uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Gambar yang diunggah", use_column_width=True)
            st.write("Memproses prediksi...")

            # Prediksi gambar yang diunggah
            label, confidence = predict_image(img)

            # Menampilkan hasil prediksi
            st.write(f"**Hasil Prediksi: {label}**")
            st.write(f"**Confidence: {confidence:.2f}%**")

    elif menu == "Gunakan Kamera":
        # Menggunakan streamlit_webrtc untuk mengakses kamera
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
