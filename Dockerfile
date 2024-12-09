# Gunakan image dasar TensorFlow
FROM tensorflow/tensorflow:latest

# Set environment variable
ENV PYTHONUNBUFFERED=1

# Buat direktori kerja di dalam container
WORKDIR /app

# Salin file requirements.txt ke dalam direktori kerja (jika ada)
COPY requirements.txt /app/

# Install dependencies yang diperlukan
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Salin semua file proyek ke dalam direktori kerja
COPY . /app/

# Expose port yang digunakan oleh Streamlit (default 8501)
EXPOSE 8501

# Perintah untuk menjalankan aplikasi Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]



