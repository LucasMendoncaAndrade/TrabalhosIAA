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
from sklearn.datasets import load_wine #dados do Wine Recognition dataset
from sklearn.metrics import accuracy_score #para a realização do cálculo da eficiência
 
#Carrega o iris dataset em iris 
wine = load_wine()
X = wine.data
y = wine.target 

#Dividindo o dataset em dois
X1 = X[0:29]
X2 = X[58:87]
X3 = X[116:147]
resultX1 = list(itertools.chain(X1, X2, X3))
y1 = y[0:29]
y2 = y[58:87]
y3 = y[116:147]
resulty1 = list(itertools.chain(y1, y2, y3))

X1 = X[29:58]
X2 = X[87:116]
X3 = X[147:179]
resultX2 = list(itertools.chain(X1, X2, X3))
y1 = y[29:58]
y2 = y[87:116]
y3 = y[147:179]
resulty2 = list(itertools.chain(y1, y2, y3))

#Implementação do Algoritmo KNN
neigh = KNeighborsClassifier(n_neighbors=3,weights="uniform")
neigh.fit(resultX1, resulty1)

#Prevendo novos valores
clustering = neigh.predict(resultX2)

#Comparação em porcentagem da clusterização obtida e da desejada
print(accuracy_score(resulty2, clustering))
