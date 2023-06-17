import streamlit as st
import pandas as pd
import time
import os

# Specify the file path

def app():
    st.title('Aplikasi Klasifikasi dengan C45 dan Fitur Seleksi PSO')
    if (os.path.exists("data/main_data.csv")):
        st.text('Data master')
        import numpy as np
        df = pd.read_csv('data/data master.csv')
        df.index = np.arange(1,len(df)+1)
        st.write(df)
        if(os.path.exists("data/definisi atribut.csv")):
            st.text('Defini atribut')
            df_atr = pd.read_csv('data/definisi atribut.csv')
            st.write(df_atr)
    data = st.file_uploader("upload data berformat csv untuk diklasifikasikan", type=['csv'])
    
    if data is not None:
            dataframe = pd.read_csv(data,lineterminator='\n')
            st.write(dataframe)

            label = st.selectbox("Pilih Kolom yang akan dijadikan label atau class :",
            list(dataframe.columns))
            label_column = pd.DataFrame(data={ 'label': [label]})
            if st.button('simpan data') :
                label_column.to_csv('data/meta/label_data.csv',index=False)
                # if os.path.exists("data/data_branch.csv"):
                #     os.remove("data/data_branch.csv")
                # if os.path.exists("data/tf_idf.csv"):
                #     os.remove("data/tf_idf.csv")
                # dataframe = dataframe[[column_data['column'][0],column_data['label'][0]]]
                if os.path.exists("data/main_data.csv"):
                    os.remove("data/main_data.csv")
                    st.warning('data diperbarui')
                dataframe.to_csv('data/main_data.csv',index=False)
                with st.spinner('tunggu sebentar ...'):
                    time.sleep(1)
                st.success('data berhasil disimpan')
                st.info('column ' + label_column['label'][0] + ' akan dijadikan label')
