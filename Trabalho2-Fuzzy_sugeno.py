"""
Sistema de controle Fuzzy para a classificação do Iris plants dataset
22/10/2020

Autores:
    Hugo Felipe Ferreira
    Lucas Mendonça Andrade

---------------------------

EXECUÇÃO DO CÓDIGO:
"""

#Bibliotecas necessárias:
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

#Carregamento do dataset
iris = load_iris()
alldata = np.vstack(iris.data)


#Definição do intervalo crisp das variáveis:
#Antecedents (Entradas): 
F1 = ctrl.Antecedent(np.arange(4.2, 8, 0.01), 'F1') # sepal_length
F2 = ctrl.Antecedent(np.arange(1.9, 4.5, 0.01), 'F2') # sepal_width
F3 = ctrl.Antecedent(np.arange(0.9, 7, 0.01), 'F3') # petal_length
F4 = ctrl.Antecedent(np.arange(0, 2.6, 0.01), 'F4') # petal_width

#Consequent (Saídas):
C = ctrl.Consequent(np.arange(0, 3, 1), 'C') # [0,1,2][Setosa,Versicolor,Virginica]


#Definição dos formatos dos subconjuntos fuzzy:
#Entradas:
F1['F1_min'] = fuzz.trapmf(F1.universe, [4.2, 4.2, 4.8, 5.9])
F1['F1_med'] = fuzz.trimf(F1.universe, [4.8, 5.95, 7.1])
F1['F1_max'] = fuzz.trapmf(F1.universe, [4.8, 7.1, 8, 8])

F2['F2_min'] = fuzz.trapmf(F2.universe, [1.9, 1.9, 2.1, 3.5])
F2['F2_med'] = fuzz.trimf(F2.universe, [2.1, 3.0, 3.9])
F2['F2_max'] = fuzz.trapmf(F2.universe, [2.2, 3.9, 4.5, 4.5])

F3['F3_min'] = fuzz.trapmf(F3.universe, [0.9, 0.9, 2.8, 2.9])
F3['F3_med'] = fuzz.trapmf(F3.universe, [2.9, 2.9, 4.4, 5.2])
F3['F3_max'] = fuzz.trapmf(F3.universe, [4.4, 5.2, 7, 7])

F4['F4_min'] = fuzz.trapmf(F4.universe, [0, 0, 0.9, 0.9])
F4['F4_med'] = fuzz.trapmf(F4.universe, [0.9, 0.9, 1.3, 1.9])
F4['F4_max'] = fuzz.trapmf(F4.universe, [1.3, 1.9, 2.6, 2.6]) 

#Saída:
C['C1'] = fuzz.trimf(C.universe, [0, 0, 1])
C['C2'] = fuzz.trimf(C.universe, [0, 1, 2])
C['C3'] = fuzz.trimf(C.universe, [1, 2, 2])


# Demonstração visual dos subconjuntos
F1.view() 
F2.view()
F3.view()
F4.view()
C.view()


#Definição das fuzzy rules
rule1 = ctrl.Rule(F1['F1_min'] & F2['F2_max'] & F3['F3_min'] & F4['F4_min'], C['C1'])
rule2 = ctrl.Rule(F1['F1_min'] & F2['F2_max'] & F3['F3_med'] & F4['F4_med'], C['C2'])
rule3 = ctrl.Rule(F1['F1_min'] & F2['F2_min'] & F3['F3_min'] & F4['F4_min'], C['C1'])
rule4 = ctrl.Rule(F1['F1_min'] & F2['F2_min'] & F3['F3_med'] & F4['F4_med'], C['C2'])
rule5 = ctrl.Rule(F1['F1_min'] & F2['F2_med'] & F3['F3_min'] & F4['F4_min'], C['C1'])
rule6 = ctrl.Rule(F1['F1_min'] & F2['F2_max'] & F3['F3_max'] & F4['F4_max'], C['C2'])
rule7 = ctrl.Rule(F1['F1_min'] & F2['F2_med'] & F3['F3_max'] & F4['F4_max'], C['C3'])
rule8 = ctrl.Rule(F1['F1_min'] & F2['F2_min'] & F3['F3_max'] & F4['F4_max'], C['C3'])
rule9 = ctrl.Rule(F1['F1_min'] & F2['F2_med'] & F3['F3_med'] & F4['F4_max'], C['C3'])
rule10 = ctrl.Rule(F1['F1_min'] & F2['F2_med'] & F3['F3_max'] & F4['F4_med'], C['C3'])
rule11 = ctrl.Rule(F1['F1_min'] & F2['F2_min'] & F3['F3_med'] & F4['F4_max'], C['C2'])
rule12 = ctrl.Rule(F1['F1_min'] & F2['F2_min'] & F3['F3_max'] & F4['F4_med'], C['C2'])
rule13 = ctrl.Rule(F1['F1_min'] & F2['F2_med'] & F3['F3_med'] & F4['F4_med'], C['C2'])
rule14 = ctrl.Rule(F1['F1_min'] & F2['F2_max'] & F3['F3_med'] & F4['F4_max'], C['C2'])
rule15 = ctrl.Rule(F1['F1_min'] & F2['F2_max'] & F3['F3_max'] & F4['F4_med'], C['C2'])
rule16 = ctrl.Rule(F1['F1_med'] & F2['F2_max'] & F3['F3_min'] & F4['F4_min'], C['C1'])
rule17 = ctrl.Rule(F1['F1_med'] & F2['F2_max'] & F3['F3_med'] & F4['F4_med'], C['C2'])
rule18 = ctrl.Rule(F1['F1_med'] & F2['F2_min'] & F3['F3_min'] & F4['F4_min'], C['C1'])
rule19 = ctrl.Rule(F1['F1_max'] & F2['F2_max'] & F3['F3_min'] & F4['F4_min'], C['C1'])
rule20 = ctrl.Rule(F1['F1_max'] & F2['F2_med'] & F3['F3_min'] & F4['F4_min'], C['C1'])
rule21 = ctrl.Rule(F1['F1_max'] & F2['F2_max'] & F3['F3_max'] & F4['F4_max'], C['C3'])
rule22 = ctrl.Rule(F1['F1_max'] & F2['F2_med'] & F3['F3_med'] & F4['F4_med'], C['C2'])
rule23 = ctrl.Rule(F1['F1_max'] & F2['F2_med'] & F3['F3_max'] & F4['F4_med'], C['C3'])
rule24 = ctrl.Rule(F1['F1_med'] & F2['F2_min'] & F3['F3_med'] & F4['F4_med'], C['C2'])
rule25 = ctrl.Rule(F1['F1_max'] & F2['F2_med'] & F3['F3_max'] & F4['F4_max'], C['C3'])
rule26 = ctrl.Rule(F1['F1_med'] & F2['F2_min'] & F3['F3_med'] & F4['F4_max'], C['C2'])
rule27 = ctrl.Rule(F1['F1_med'] & F2['F2_min'] & F3['F3_max'] & F4['F4_med'], C['C2'])
rule28 = ctrl.Rule(F1['F1_med'] & F2['F2_med'] & F3['F3_med'] & F4['F4_med'], C['C2'])
rule29 = ctrl.Rule(F1['F1_max'] & F2['F2_min'] & F3['F3_med'] & F4['F4_med'], C['C2'])
rule30 = ctrl.Rule(F1['F1_med'] & F2['F2_min'] & F3['F3_max'] & F4['F4_max'], C['C3'])
rule31 = ctrl.Rule(F1['F1_med'] & F2['F2_med'] & F3['F3_max'] & F4['F4_med'], C['C3'])
rule32 = ctrl.Rule(F1['F1_med'] & F2['F2_med'] & F3['F3_med'] & F4['F4_max'], C['C2'])
rule33 = ctrl.Rule(F1['F1_max'] & F2['F2_min'] & F3['F3_max'] & F4['F4_med'], C['C3'])
rule34 = ctrl.Rule(F1['F1_max'] & F2['F2_min'] & F3['F3_med'] & F4['F4_max'], C['C3'])
rule35 = ctrl.Rule(F1['F1_med'] & F2['F2_med'] & F3['F3_max'] & F4['F4_max'], C['C3'])
rule36 = ctrl.Rule(F1['F1_max'] & F2['F2_min'] & F3['F3_max'] & F4['F4_max'], C['C3'])
rule37 = ctrl.Rule(F1['F1_max'] & F2['F2_med'] & F3['F3_med'] & F4['F4_max'], C['C2'])
rule38 = ctrl.Rule(F1['F1_med'] & F2['F2_max'] & F3['F3_max'] & F4['F4_max'], C['C3'])
rule39 = ctrl.Rule(F1['F1_max'] & F2['F2_max'] & F3['F3_med'] & F4['F4_max'], C['C2'])
rule40 = ctrl.Rule(F1['F1_max'] & F2['F2_max'] & F3['F3_max'] & F4['F4_med'], C['C3'])
rule41 = ctrl.Rule(F1['F1_max'] & F2['F2_max'] & F3['F3_med'] & F4['F4_med'], C['C2'])
rule42 = ctrl.Rule(F1['F1_med'] & F2['F2_max'] & F3['F3_max'] & F4['F4_med'], C['C2'])
rule43 = ctrl.Rule(F1['F1_med'] & F2['F2_med'] & F3['F3_min'] & F4['F4_min'], C['C1'])
rule44 = ctrl.Rule(F1['F1_max'] & F2['F2_min'] & F3['F3_min'] & F4['F4_min'], C['C1'])


#Criação do sistema de controle e de sua simulação:
plantselect_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27, rule28, rule29, rule30, rule31, rule32, rule33, rule34, rule35, rule36, rule37, rule38, rule39, rule40, rule41, rule42, rule43, rule44]) #criação do controle fuzzy

plantselect = ctrl.ControlSystemSimulation(plantselect_ctrl) #simulação do controlador


#Simulação do controle:
    
result_final = []
    
for i in range(150):  
    plantselect.input['F1'] = alldata[i][0]
    plantselect.input['F2'] = alldata[i][1]
    plantselect.input['F3'] = alldata[i][2]
    plantselect.input['F4'] = alldata[i][3]
    plantselect.compute()
    resultado=plantselect.output['C']
    if resultado < 0.6666666666:
        resultado = 0
    if resultado > 0.6666666666:
        if resultado < 1.3333333333:
            resultado = 1
        else:
            resultado = 2
    result_final.append(resultado)
    print(resultado)

#Verificando resultados
print('')
print('Accuracy score:',accuracy_score(iris.target, result_final))

