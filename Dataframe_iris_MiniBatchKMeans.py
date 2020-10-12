"""
MiniBatchKMeans para dataset iris

05/10/2020

Autores:
    Hugo Felipe Ferreira
    Lucas Mendonça Andrade
    Eber José de Morais Junior
    Vitor Fernandes Azevedo

"""

import numpy as np #biblioteca comumente utilizada para aplicações matemáticas
from sklearn.cluster import MiniBatchKMeans #para a implementação do código Mini Batch K Means
from sklearn.datasets import load_iris #dados do Iris plants dataset
from sklearn.metrics import accuracy_score #para a realização do cálculo da eficiência
 
#Carregamwnto do iris dataset
iris = load_iris() 

#Implementação do Algoritmo Mini Batch K-means
clustering = MiniBatchKMeans(n_clusters=3, random_state=0, batch_size=35).fit(iris.data)

#Reordenação dos resultados
clust = np.choose(clustering.labels_, [0, 2, 1])

#Comparação em porcentagem da clusterização obtida e da desejada
print(accuracy_score(iris.target, clust))

