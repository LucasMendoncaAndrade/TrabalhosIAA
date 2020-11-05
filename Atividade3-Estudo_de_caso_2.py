"""
Utilização de AG para otimização do número de neighbors do KNN

26/10/2020

Autores:
    Hugo Felipe Ferreira
    Lucas Mendonça Andrade

"""
import itertools
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from geneticalgorithm import geneticalgorithm as ga

#Importando o dataset iris
iris = load_iris()
X = iris.data
y = iris.target 

#Dividindo o dataset em dois (metade para treinamento e metade para classificação)
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

#Função para otimização
def f(X):
    neigh = KNeighborsClassifier(n_neighbors=int(X),weights="uniform")
    neigh.fit(resultX1, resulty1)
    clustering = neigh.predict(resultX2)
    result = accuracy_score(resulty2, clustering)*-1
    #Retorna o negativo do accuracy
    return result

#Executando o AG
varbound = np.array([[1,len(resultX2)]]*1) #Entre 1 e 75 neighbors
model = ga(function=f,dimension=1,variable_type='int',variable_boundaries=varbound)
model.run()