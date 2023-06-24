from cgitb import text
import streamlit as st
import pandas as pd
import time
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def app() :
    test_size = pd.read_csv('data/meta/test_size.csv')
    
    #penambahan upload_TF_IDF
    data_master = pd.read_csv('data/data master.csv',lineterminator='\r')
    df_data_num = pd.read_csv('data/meta/df_data_numeric.csv')
    df_data_num = df_data_num.to_numpy() 
    feature_column = pd.read_csv('data/meta/feature_names.csv')
    selected_column = pd.read_csv('data/meta/selected_feature.csv')
    X =df_data_num[:, :-1]
    y =df_data_num[:, -1]
    st.subheader('Klasifikasi Decision Tree (C4.5)')

    # data_suffle = st.checkbox('Acak Data',value=True)
    df_accuracy = pd.DataFrame(columns=['proses','akurasi'])
    # feature_column = feature_column.drop(feature_column[feature_column['label'] == ''].index)
    # st.write(feature_column['fitur'].values)
    
    with st.spinner('Wait for it...'):
        text_train, text_test, y_train, y_test = train_test_split(X, y, test_size = float(test_size['test size'][0]),random_state=1221)

        text_train = pd.DataFrame(text_train, columns=feature_column['fitur'].tolist())
        text_test = pd.DataFrame(text_test, columns=feature_column['fitur'].tolist())
        # st.write(feature_column['fitur'].to_numpy)
        # Create Decision Tree classifer object
        clf = DecisionTreeClassifier()
        # Train Decision Tree Classifer
        clf = clf.fit(text_train[selected_column['selected'].tolist()],y_train)
        
        #Predict the response for test dataset
        y_pred = clf.predict(text_test[selected_column['selected'].tolist()])
        fig = plt.figure(figsize=(25,20))
        _ = tree.plot_tree(clf, feature_names= selected_column['selected'].tolist(),class_names=list(clf.classes_),impurity = False,
        filled=True)
        fig.savefig(f"data/pictures/classification/train-PSO-C45.png")
        accuracy = classification_report(y_test,y_pred,output_dict=True)['accuracy']
        df_accuracy.at[1, 'akurasi'] = accuracy
        df_accuracy.at[1, 'proses'] = 'PSO C45'
        
        data_pred = pd.DataFrame({'Predsi Kelas':y_pred,'Kelas Sesunggunya':y_test})
        data_pred = (data_pred.join(data_master[['Nama','NISN','JK']]))
        data_pred.to_csv('data/hasil-akurasi-PSO-C45.csv',index=False)

            # text_representation = tree.export_text(clf)
            # st.write(text_representation)
            # st.write(tree.plot_tree(clf))

    with st.expander("Lihat Hasil"):
        # st.line_chart(df_accuracy['akurasi'])
        st.caption('data akurasi tiap percobaan')
        st.write(df_accuracy)
        st.caption(f'Tebel Prediksi klasifikasi C45-PSO')
        st.write(pd.read_csv('data/hasil-akurasi-PSO-C45.csv'))
        st.caption(f'GAMBAR POHON KEPUTUSAN PSO C45')
        st.image(f"data/pictures/classification/train-PSO-C45.png")
    
    data = st.file_uploader("upload data untuk memprediksi label data tersebut", type=['csv'])
    if data is not None:
        
        with st.spinner('Tunggu sebentar untuk memprediksinya...'):
            time.sleep(5)
        data = pd.read_csv(data,lineterminator='\n')
        # text_train, text_test, y_train, y_test = train_test_split(X, y, test_size = float(test_size['test size'][0]),random_state=1221)
        X = pd.DataFrame(X, columns=[feature_column['fitur'].tolist()])
        
        # text_test = pd.DataFrame(text_test, columns=[feature_column['fitur'].tolist()])
        clf = DecisionTreeClassifier()
        # Train Decision Tree Classifer
        clf = clf.fit(X[selected_column['selected'].tolist()],y)
        #Predict the response for test dataset
        mapping = {'ya': 1, 'tidak': 0}

# Mengubah data pada semua kolom kecuali kolom kelas
        data_selected = data[selected_column['selected'].tolist()].applymap(lambda x: mapping.get(x, x))
        y_pred = clf.predict(data_selected)
        y_pred = pd.DataFrame(y_pred, columns=["Label Prediksi"])
        st.caption('Data Terprediksi')
        st.write(y_pred.join(data))

    
    # st.write(text_train)
    # st.write(text_test)
