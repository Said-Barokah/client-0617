from cgitb import text
import streamlit as st
import pandas as pd
import time

def app() :
    
    from sklearn.model_selection import train_test_split
    #penambahan upload_TF_IDF
    data_master = pd.read_csv('data/data master.csv',lineterminator='\r')
    df_data_num = pd.read_csv('data/meta/df_data_numeric.csv')
    df_data_num = df_data_num.to_numpy() 
    feature_column = pd.read_csv('data/meta/feature_names.csv')
    selected_column = pd.read_csv('data/meta/selected_feature.csv')
    X =df_data_num[:, :-1]
    y =df_data_num[:, -1]
    st.subheader('Klasifikasi Decision Tree (C4.5)')

    test_size = (st.number_input('Data test Sebanyak',min_value=0.0,max_value=1.0,value=0.2,step=0.1,key='test_size'))
    # data_suffle = st.checkbox('Acak Data',value=True)
    df_accuracy = pd.DataFrame(columns=['proses','akurasi'])
    # feature_column = feature_column.drop(feature_column[feature_column['label'] == ''].index)
    # st.write(feature_column['fitur'].values)
    
    with st.spinner('Wait for it...'):
        for j in range(2):
            if (j==1):
                text_test = X[0:int((data_master.shape[0]-1)*test_size)]
                text_train = X[int(((data_master.shape[0]-1)*test_size)+1):int((data_master.shape[0]-1))]
                y_test = y[0:int((data_master.shape[0]-1)*test_size)]
                y_train = y[int(((data_master.shape[0]-1)*test_size)+1):int(data_master.shape[0]-1)]
                # text_train, text_test, y_train, y_test = train_test_split(X, y, test_size = test_size,train_size= train_size,stratify=y,random_state=10)
                text_train = pd.DataFrame(text_train, columns=[feature_column['fitur'].tolist()])
                
                text_test = pd.DataFrame(text_test, columns=[feature_column['fitur'].tolist()])
                from sklearn import tree
                from sklearn.metrics import classification_report
                from sklearn.tree import DecisionTreeClassifier
                from sklearn import tree 
                import matplotlib.pyplot as plt
                # st.write(feature_column['fitur'].to_numpy)
                # Create Decision Tree classifer object
                clf = DecisionTreeClassifier()
                # Train Decision Tree Classifer
                clf = clf.fit(text_train.loc[:, [selected_column['selected'].tolist()]],y_train)
                #Predict the response for test dataset
                y_pred = clf.predict(text_test.loc[:, [selected_column['selected'].tolist()]])
                fig = plt.figure(figsize=(25,20))
                _ = tree.plot_tree(clf, feature_names= selected_column['selected'].tolist(),class_names=list(clf.classes_),impurity = False,
                filled=True)
                fig.savefig(f"data/pictures/classification/train-PSO-C45.png")
                accuracy = classification_report(y_test,y_pred,output_dict=True)['accuracy']
                df_accuracy.at[1, 'akurasi'] = accuracy
                df_accuracy.at[1, 'proses'] = 'PSO C45'
                
                data_pred = pd.DataFrame({'kelas Pred':y_pred,'kelas':y_test})
                data_pred = pd.concat([data_master[['Nama','NISN','JK']][:int((data_master.shape[0]-1)*test_size)],text_test, data_pred], axis=1)
                data_pred.to_csv('data/hasil-akurasi-PSO-C45.csv',index=False)
            if(j == 0):
                st.write((data_master.shape[0]-1)*test_size)
                # text_train, text_test, y_train, y_test = train_test_split(X, y, test_size = test_size,train_size= train_size,stratify=y, random_state=1234)
                text_test = X[0:int((data_master.shape[0]-1)*test_size)]
                text_train = X[int(((data_master.shape[0]-1)*test_size)+1):int((data_master.shape[0]-1))]
                y_test = y[0:int((data_master.shape[0]-1)*test_size)]
                y_train = y[int(((data_master.shape[0]-1)*test_size)+1):int(data_master.shape[0]-1)]
                text_train = pd.DataFrame(text_train, columns=[feature_column['fitur'].values])
                text_test = pd.DataFrame(text_test, columns=[feature_column['fitur'].values])
                from sklearn import tree
                from sklearn.metrics import classification_report
                from sklearn.tree import DecisionTreeClassifier
                from sklearn import tree 
                import matplotlib.pyplot as plt
                # Create Decision Tree classifer object
                clf = DecisionTreeClassifier()
                # Train Decision Tree Classifer
                clf = clf.fit(text_train,y_train)
                #Predict the response for test dataset
                y_pred = clf.predict(text_test)
                fig = plt.figure(figsize=(25,20))
                _ = tree.plot_tree(clf, feature_names= feature_column['fitur'].tolist(), class_names=list(clf.classes_),impurity = False,
                filled=True)
                fig.savefig(f"data/pictures/classification/train-C45.png")
                accuracy = classification_report(y_test,y_pred,output_dict=True)['accuracy']
                # df_accuracy = df_accuracy.append({'akurasi C45' : accuracy},ignore_index=True)
                df_accuracy.at[2, 'akurasi'] = accuracy
                df_accuracy.at[2, 'proses'] = 'C45'
                data_pred = pd.DataFrame({'kelas Pred':y_pred,'kelas':y_test})
                data_pred = pd.concat([data_master[['Nama','NISN','JK']][:int((data_master.shape[0]-1)*test_size)],text_test, data_pred], axis=1)
                data_pred.to_csv('data/hasil-akurasi-C45.csv',index=False)
            # text_representation = tree.export_text(clf)
            # st.write(text_representation)
            # st.write(tree.plot_tree(clf))

    with st.expander("Lihat Hasil"):
        # st.line_chart(df_accuracy['akurasi'])
        st.caption('data akurasi tiap percobaan')
        st.write(df_accuracy)
        for i in range(2):
            if(i==1):
                st.caption(f'Tebel Prediksi klasifikasi C45-PSO')
                st.write(pd.read_csv('data/hasil-akurasi-PSO-C45.csv'))
                st.caption(f'GAMBAR POHON KEPUTUSAN PSO C45')
                st.image(f"data/pictures/classification/train-PSO-C45.png")
                continue
            st.caption(f'Tebel Prediksi klasifikasi C45')
            st.write(pd.read_csv('data/hasil-akurasi-C45.csv'))
            st.caption(f'GAMBAR POHON KEPUTUSAN C45')
            st.image(f"data/pictures/classification/train-C45.png")
    # st.write(text_train)
    # st.write(text_test)