"""
KNN para dataset wine

05/10/2020

Autores:
    Hugo Felipe Ferreira
    Lucas Mendonça Andrade
    Eber José de Morais Junior
    Vitor Fernandes Azevedo
"""
import itertools
from sklearn.neighbors import KNeighborsClassifier #para implementação do algoritmo KNN
from sklearn.datasets import load_iris #dados do Iris plants dataset
from sklearn.metrics import accuracy_score #para a realização do cálculo da eficiência
 
#Carregamento do iris dataset em iris 
iris = load_iris()
X = iris.data
y = iris.target 

#Dividindo o dataset em dois
X1 = X[0:25]
X2 = X[50:75]
X3 = X[100:125]
resultX1 = list(itertools.chain(X1, X2, X3))
y1 = y[0:25]
y2 = y[50:75]
y3 = y[100:125]
resulty1 = list(itertools.chain(y1, y2, y3))

X1 = X[25:50]
X2 = X[75:100]
X3 = X[125:151]
resultX2 = list(itertools.chain(X1, X2, X3))
y1 = y[25:50]
y2 = y[75:100]
y3 = y[125:151]
resulty2 = list(itertools.chain(y1, y2, y3))

#Implementação do Algoritmo KNN
neigh = KNeighborsClassifier(n_neighbors=3,weights="uniform")
neigh.fit(resultX1, resulty1)

#Prevendo novos valores
clustering = neigh.predict(resultX2)

#Comparação em porcentagem da clusterização obtida e da desejada
print(accuracy_score(resulty2, clustering))
