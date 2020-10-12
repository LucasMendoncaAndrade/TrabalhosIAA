"""
Kmeans para dataset iris

05/10/2020

Autores:
    Hugo Felipe Ferreira
    Lucas Mendonça Andrade
    Eber José de Morais Junior
    Vitor Fernandes Azevedo

"""

import numpy as np #biblioteca comumente utilizada para aplicações matemáticas
from sklearn.cluster import KMeans #biblioteca para a realização do código K-means
from sklearn.datasets import load_iris #dados do Iris plants dataset
from sklearn.metrics import accuracy_score #para a realização do cálculo da eficiência
 
#Carregamento do iris dataset
iris = load_iris() 

#Implementação do Algoritmo K-means
kmeans = KMeans(n_clusters=3, random_state=0).fit(iris.data)

#Reordenação dos resultados
clust = np.choose(kmeans.labels_, [2, 0, 1])

#Comparação em porcentagem da clusterização obtida e da desejada
print(accuracy_score(iris.target, clust))

