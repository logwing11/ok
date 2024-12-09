import streamlit as st

# Fungsi untuk menampilkan halaman-halaman aplikasi
from nasnet import show_ns
from mobilenet import show_mnv
from densenet import show_dns


# Menambahkan judul dan logo di bagian atas aplikasi
st.sidebar.title("PILIH MODEL")

# Tambahkan menu-menu ke sidebar
menu_selection = st.sidebar.radio(
    "",
    [
        "DenseNet",
        "Inception",
        "MobileNet",
        "Nasnet",
        "Resnet",
        "VGG",
        "Xception",
    ],
)

# Logika untuk menavigasi ke halaman yang dipilih
if menu_selection == "Nasnet":
    show_ns()
elif menu_selection == "MobileNet":
    show_mnv()
elif menu_selection == "DenseNet":
    show_dns()
