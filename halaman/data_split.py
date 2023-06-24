from cgitb import text
import streamlit as st
import pandas as pd
import time
from sklearn.model_selection import train_test_split, cross_val_score

def app() :
    st.subheader('Split Data menjadi Data Training dan Data Testing')
    data = pd.read_csv('data/data master.csv',lineterminator='\r')
    test_size = (st.number_input('Data test Sebanyak',min_value=0.0,max_value=1.0,value=0.2,step=0.1,key='test_size'))
    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['kelas']), data[['kelas']], test_size=test_size, random_state=1221)
    df_test_size = pd.DataFrame(data={ 'test size': [test_size]})
    df_test_size.to_csv('data/meta/test_size.csv',index=False)
    with st.spinner('tunggu sebentar ...'):
        time.sleep(1)
        st.caption('Data test')
    st.dataframe(X_test.join(y_test))
    st.caption('Data train')
    st.dataframe(X_train.join(y_train))
