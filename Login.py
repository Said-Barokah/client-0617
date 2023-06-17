import streamlit as st

# Informasi pengguna yang valid
def authent(username, password):
    import pandas as pd
    df_user = pd.read_csv('data/user.csv')
    if username == df_user['nama'][0] and password ==str(df_user['password'][0]):
        return True
    return False

def login():
    st.title("Login Page")
    st.write("Silakan masuk menggunakan username dan password Anda.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if authent(username, password):
            st.session_state.login_status = True
            st.success("Login berhasil! dan klik tombol login sekali lagi untuk masuk utama")
            # Tandai status login sebagai berhasil
        else:
            st.error("Username atau password salah.")

def home():
    st.title("Halaman Utama")
    st.write("Selamat datang di halaman utama! Hanya pengguna yang berhasil login yang dapat mengakses halaman ini.")
if 'login_status' not in st.session_state:
    st.session_state.login_status = False

if  not st.session_state.login_status:
    login()
else:
    from halaman import upload_data,feature_selection,classification
    page_names_to_funcs = {
            "Upload Data" : upload_data.app,
            "Seleksi Fitur": feature_selection.app,
            "Klasifikasi" : classification.app
    }

    demo_name = st.sidebar.selectbox("halaman", page_names_to_funcs.keys())
    page_names_to_funcs[demo_name]()
            
# main()
