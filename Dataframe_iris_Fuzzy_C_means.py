"""
Fuzzy Cmeans para dataset iris

05/10/2020

Autores:
    Hugo Felipe Ferreira
    Lucas Mendonça Andrade
    Eber José de Morais Junior
    Vitor Fernandes Azevedo

"""

import skfuzzy as fuzz #biblioteca para a implementação do Fuzzy C-means
import numpy as np #bibliotexa comumente utilizada para aplicações matemáticas
from sklearn.metrics import accuracy_score #para a realização do cálculo da eficiência
from sklearn.datasets import load_iris #dados Iris plants dataset


#Carregamento do iris dataset em iris 
iris = load_iris() #variável com a matriz do iris
alldata = np.vstack(iris.data) #recebe toda a matriz do iris
alldata2 = alldata.transpose() #matriz transposta dos dados da iris
label = iris.target #clusterização desejada
ncenters = 3 #número de clusters (3 plantas)

cluster_membership = np.linspace(1, 155, num=155)

while cluster_membership[0] > 0:
#Implementação do Algoritmo Fuzzy C-means
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            alldata2, ncenters, 6, error=0.005, maxiter=1000, init=None)
        cluster_membership = np.argmax(u, axis=0)

#Comparação em porcentagem da clusterização obtida e da desejada
print(accuracy_score(iris.target, cluster_membership))

