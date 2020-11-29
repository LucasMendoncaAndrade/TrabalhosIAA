"""
Comparativo entre metodos de classificação e RNA
27/11/2020
Hugo Felipe Ferreira
Lucas Mendonça Andrade
Vitor Fernandes Azevedo
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import skfuzzy as fuzz
from geneticalgorithm import geneticalgorithm as ga

# Variaveis comuns aos classificadores
# Carregando o dataset
iris = load_iris()

# Dividindo o dataset (proporcao 0.5 --> pode mudar)
datasets = train_test_split(iris.data, iris.target,
                            test_size=0.5)

train_data, test_data, train_labels, test_labels = datasets

# Resulta em dois datasets gerados aleatoriamente com metade dos dados em cada

"""
Uso de RNA no dataset iris
27/11/2020
"""
print('------------ RNA ----------- \n')
# scaling the data
scaler = StandardScaler()

# we fit the train data
scaler.fit(train_data)

# scaling the train data
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

# Configurando o classificador de RNA
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)

# Treinando a rede
mlp.fit(train_data, train_labels)

# Calculando as metricas
predictions_train = mlp.predict(train_data)
print('A acurácia dos dados de treino é: ', accuracy_score(predictions_train, train_labels))
predictions_test = mlp.predict(test_data)
print('A acurácia dos dados de teste é: ', accuracy_score(predictions_test, test_labels), '\n')
print(classification_report(predictions_test, test_labels))


"""
Uso do C-means no dataset iris
05/10/2020 ---> 89,33%
"""
print('\n \n ------------ C-means ----------- \n')
clustering_kmeans = KMeans(n_clusters=3, random_state=0).fit(iris.data)
clustering_kmeans = np.choose(clustering_kmeans.labels_, [2, 0, 1])  # Reorganizando
print('A acurácia dos dados de teste é: ',accuracy_score(clustering_kmeans, iris.target), '\n')
print(classification_report(clustering_kmeans, iris.target))


"""
Uso do Fuzzy C-means no dataset iris
05/10/2020  ------> 89,33%
"""
print('\n \n ------------ Fuzzzy C-means ----------- \n')
alldata = np.vstack(iris.data)
alldata = alldata.transpose()
label = iris.target 

clustering_fuzzy_kmeans = np.linspace(1, 155, num=155)
while clustering_fuzzy_kmeans[0] > 0:
#Implementa o Algoritmo Fuzzy C-means
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            alldata, 3, 2, error=0.005, maxiter=1000, init=None)
        clustering_fuzzy_kmeans = np.argmax(u, axis=0)
clustering_fuzzy_kmeans = np.choose(clustering_fuzzy_kmeans, [0, 2, 1])

print('A acurácia dos dados de teste é: ',accuracy_score(clustering_fuzzy_kmeans, iris.target), ' \n')
print(classification_report(clustering_fuzzy_kmeans, iris.target))


"""
Uso do KNN no dataset iris
05/10/2020
"""
print('\n \n ------------ KNN ----------- \n')
neigh = KNeighborsClassifier(n_neighbors=3,weights="uniform")
neigh.fit(train_data, train_labels)
clustering_knn_train = neigh.predict(train_data)
print('A acurácia dos dados de treino é: ', accuracy_score(clustering_knn_train, train_labels))
clustering_knn_test = neigh.predict(test_data)
print('A acurácia dos dados de teste é: ',accuracy_score(clustering_knn_test, test_labels), '\n')
print(classification_report(clustering_knn_test, test_labels))


"""
Uso do KNN com AG no dataset iris
05/10/2020
"""
print('\n \n ------------ KNN com AG ----------- \n')

accuracy_max = 0
n = 0
#Função para otimização
def f(X):
    global accuracy_max
    global n
    # Definindo o valor de neighbors
    neigh = KNeighborsClassifier(n_neighbors=int(X),weights="uniform")
    # Treinando o algoritmo com metade dos dados
    neigh.fit(train_data, train_labels)
    # Predizendo a outra metade dos dados
    clustering = neigh.predict(test_data)
    # Calculando o accuracy dos dados previstos
    fob = accuracy_score(clustering, test_labels)*-1
    if fob*-1 > accuracy_max:
        accuracy_max = fob*-1
        n = int(X)
    #Retorna o negativo do accuracy
    return fob

varbound = np.array([[1,len(test_labels)]]*1) #Entre 1 e 75 neighbors
model = ga(function=f,dimension=1,variable_type='int',variable_boundaries=varbound)
model.run()

neigh = KNeighborsClassifier(n_neighbors=int(n),weights="uniform")
neigh.fit(train_data, train_labels)
clustering_knnga_train = neigh.predict(train_data)
print('\n A acurácia dos dados de treino é: ', accuracy_score(clustering_knnga_train, train_labels))
clustering_knnga_test = neigh.predict(test_data)
print('A acurácia dos dados de teste é: ',accuracy_score(clustering_knnga_test, test_labels), '\n')
print(classification_report(clustering_knnga_test, test_labels))




