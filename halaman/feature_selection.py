
import streamlit as st
import time

#sesuaikan dengan nama file
import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score
if os.path.exists("data/main_data.csv") and os.path.exists("data/meta/label_data.csv"):
    data = pd.read_csv('data/main_data.csv')
    mapping = {'ya': 1, 'tidak': 0}

# Mengubah data pada semua kolom kecuali kolom kelas
    data.iloc[:, :-1] = data.iloc[:, :-1].applymap(lambda x: mapping.get(x, x))

    
    label_data = pd.read_csv('data/meta/label_data.csv')
    # st.dataframe(data[label_data['label'][0]])
    feature_names = data.drop(label_data['label'][0], axis=1).columns
    
    feature_column = pd.DataFrame(feature_names,columns=['fitur'])
    feature_column.to_csv('data/meta/feature_names.csv',index=False)
    
    feature_names = feature_names.to_numpy() 
    data_np = data.to_numpy() 
    df_data_num = pd.DataFrame(data_np)
    df_data_num.to_csv('data/meta/df_data_numeric.csv',index=False)
    test_size = pd.read_csv('data/meta/test_size.csv')
    X =data_np[:, :-1]
    y =data_np[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size['test size'][0]), random_state=1221)
# !pip install niapy
from niapy.problems import Problem
# data penyimpanan dari beberapa atribut yang akan digunakan natinya
data_selesction_pso = pd.DataFrame(columns=['bobot fitur (k01-k20)','fitur yang terpilih','akurasi','nilai fitness'])

class SVMFeatureSelection(Problem):
    def __init__(self, X_train, y_train, alpha=0.99):
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha

        # st.write('said',self.X_train)

    def _evaluate(self, x):
        from sklearn.svm import SVC
        global data_selesction_pso
        selected = x > 0.5
        num_selected = selected.sum()
        if num_selected == 0:
            return 1.0
        accuracy = cross_val_score(SVC(), self.X_train[:, selected], self.y_train, cv=2, n_jobs=-1).mean()
        score = 1 - accuracy
        num_features = self.X_train.shape[1]

        # Improvisasi dai nilai akurasi itu sendiri
        fitness_value = self.alpha * score + (1 - self.alpha) * (num_selected / num_features)

        # untuk menyimpan suatu data nilai dari polpulasi tersebut
        data_baru ={'bobot fitur (k01-k20)': [x],'fitur yang terpilih':[feature_names[selected]],'akurasi':accuracy, 'nilai fitness': fitness_value}
        data_baru = pd.DataFrame(data_baru)
        data_selesction_pso = pd.concat([data_selesction_pso, data_baru], ignore_index=True)

        return fitness_value
def app() :
    st.subheader('Seleksi fitur PSO')
    from sklearn.datasets import load_breast_cancer
    from niapy.task import Task
    from niapy.algorithms.basic import ParticleSwarmOptimization

    population_size = st.number_input('population size', value=10)
    c1 = st.number_input('c1 (bobot kognitif)',value=2.0)
    c2 = st.number_input('c2 (bobot sosial)',value=2.0)
    w = st.number_input('w (bobot inertia)',value=0.7)
    min_velocity = st.number_input('minimal kecepatan',value=0)
    max_velocity = st.number_input('maksimal kecepatan',value=1)
    iterasi = st.number_input('jumlah generasi',value=100)

    df_variabel_pso = pd.DataFrame(data={
        'parameter' : ['test size','population size','bobot kognitif','bobot sosial','bobot inertia', 'minimal kecepatan','maximal kecepatan','jumlah generasi'],
        'nilai bobot' : [str(float(test_size['test size'][0])),population_size,c1,c2,w,min_velocity,max_velocity,iterasi] 
                                       })
    df_variabel_pso.to_csv('data/meta/variabel_pso.csv',index=False)
    problem = SVMFeatureSelection(X_train, y_train)
    task = Task(problem, max_iters=iterasi)
    algorithm = ParticleSwarmOptimization(population_size=population_size, seed=1234,  c1=c1, c2=c2, w=w, min_velocity=min_velocity, max_velocity=max_velocity)
    with st.spinner('tunggu sebentar ...'):
        time.sleep(1)
        best_features, best_fitness = algorithm.run(task)

    selected_features = best_features > 0.5
    df_selected_features = pd.DataFrame(feature_names[selected_features].tolist(), columns=['selected'])
    df_selected_features.to_csv('data/meta/selected_feature.csv',index=False)

    
    st.write('Number of selected features:', selected_features.sum())
    st.write('Selected features:', ', '.join(feature_names[selected_features].tolist()))
    df_atr = pd.read_csv('data/definisi atribut.csv')
    df__atr_selected = df_atr.loc[df_atr['Atribut'].isin(feature_names[selected_features].tolist())]
    st.write(df__atr_selected)
    # mengurutkan data menurut best features terbaik
    df_sort = data_selesction_pso.sort_values('nilai fitness', ascending=True)

    # Memilih 10 data teratas
    st.write(df_sort[:10])


    # Menyimpan dalam bentuk csv hasil dari ppulasi di PSO
    # data_selesction_pso.to_csv('/content/drive/MyDrive/Colab Notebooks/Project/Data Scientist/Client - 0528/populasi.csv')  
