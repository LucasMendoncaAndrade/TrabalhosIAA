"""
MiniBatchKMeans para dataset wine

08/10/2020

Autores:
    Hugo Felipe Ferreira
    Lucas Mendonça Andrade
    Eber José de Morais Junior
    Vitor Fernandes Azevedo

"""

import numpy as np #biblioteca comumente utilizada para aplicações matemáticas
from sklearn.cluster import MiniBatchKMeans #para a implementação do código Mini Batch K Means
from sklearn.datasets import load_wine #dados do Wine Recognition dataset
from sklearn.metrics import accuracy_score #para a realização do cálculo da eficiência
 
#Carrega o wine dataset
wine = load_wine() 

#Implementação do Algoritmo Mini Batch K-means
clustering = MiniBatchKMeans(n_clusters=3, random_state=0, batch_size=20).fit(wine.data)

#Reordenando os resultados
clust = np.choose(clustering.labels_, [0, 2, 1])

#Comparação em porcentagem da clusterização obtida e da desejada
print(accuracy_score(wine.target, clustering.labels_))

