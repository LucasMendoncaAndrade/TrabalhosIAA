"""
Kmeans para dataset wine

05/10/2020

Autores:
    Hugo Felipe Ferreira
    Lucas Mendonça Andrade
    Eber José de Morais Junior
    Vitor Fernandes Azevedo

"""

import numpy as np #biblioteca comumente utilizada para aplicações matemáticas
from sklearn.cluster import KMeans #biblioteca para a realização do código K-means
from sklearn.datasets import load_wine #dados do Wine recognition dataset
from sklearn.metrics import accuracy_score #para a realização do cálculo da eficiência
 
#Carregamento do wine dataset
wine = load_wine() 

#Implementa o Algoritmo K-means
kmeans = KMeans(n_clusters=3, random_state=0).fit(wine.data)

#Reordenando os resultados
clustering_2 = np.choose(kmeans.labels_, [1, 0, 2])

#Comparação em porcentagem da clusterização obtida e da desejada
print(accuracy_score(wine.target, clustering_2))

