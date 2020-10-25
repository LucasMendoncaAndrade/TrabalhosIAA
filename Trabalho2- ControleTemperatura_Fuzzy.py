"""
Sistema de controle Fuzzy para a temperatura do chuveiro

19/10/2020

Autores:
    Hugo Felipe Ferreira
    Lucas Mendonça Andrade

---------------------------

O objetivo do código é encontrar a potência de operação do chuveiro a partir
dos dados de vazão e temperatura do banho. Ou seja, o intuito é encontrar
o nível de potência de operação para um banho à temperatura desejada e com
a vazão encontrada no chuveiro.

Consequente, as entradas são VAZÃO e TEMPERATURA DESEJADA e a saída é a POTÊNCIA.

Entradas:
1)Vazão:
    Crisp set: 0 a 100%.
    Fuzzy set: Vazão Baixa (VB), Vazão Média (VM) e Vazão Alta (VA).

2)Temperatura:
    Crisp set: 20 a 50 °C
    Fuzzy set: Temperatura Baixa (TB), Temperatura Média (TM) e Temperatura Alta (TA)

Saída:
1) Potência:
    Crisp set: 0 a 100%
    Fuzzy set: Potência Baixa (PB), Potência Média (PM), Potência Alta (PA).

---------------------------

As Fuzzy Rules definidas foram:
1) IF vazão = 'VB' *and* temperatura = 'TB' *or* 'TM',
   THEN potência = 'PB'

2) IF vazão = 'VB' *and* temperatura = 'TA',
   THEN potência = 'PM'
   
3) IF vazão = 'VM' *and* temperatura = 'TB',
   THEN potência = 'PB'

4) IF vazão = 'VM' *and* temperatura = 'TM',
   THEN potência = 'PM'
   
5) IF vazão = 'VM' *and* temperatura = 'TA',
   THEN potência = 'PA'

6) IF vazão = 'VA' *and* temperatura = 'TB',
   THEN potência = 'PM'
   
7) IF vazão = 'VA' *and* temperatura = 'TM' *or* 'TA',
   THEN potência = 'PA'
   

EXECUÇÃO DO CÓDIGO:
"""

#Bibliotecas necessárias:
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

#Definição do intervalo crisp das variáveis:
#Antecedents (Entradas):
vazao = ctrl.Antecedent(np.arange(0, 101, 1), 'vazao') #vazão definida como o intervalo entre 0 e 100 % com passo de 1
temperatura = ctrl.Antecedent(np.arange(20, 51, 1), 'temperatura') #temperatura definida como o intervalo entre 20 e 50 °C com passo de 1

#Consequent (Saída):
potencia = ctrl.Consequent(np.arange(0, 101, 1), 'potencia') #potência definida como o intervalo entre 0 e 100 com passo de 1

#Definição dos formatos dos subconjuntos fuzzy (escolheu-se utilizar apenas triangulos)
#Entradas:
    
#Vazão:
vazao['VB'] = fuzz.trimf(vazao.universe, [0, 0, 50])
vazao['VM'] = fuzz.trimf(vazao.universe, [0, 50, 100])
vazao['VA'] = fuzz.trimf(vazao.universe, [50, 100, 100])

#Temperatura:
temperatura['TB'] = fuzz.trimf(temperatura.universe, [20, 20, 35])
temperatura['TM'] = fuzz.trimf(temperatura.universe, [20, 35, 50])
temperatura['TA'] = fuzz.trimf(temperatura.universe, [35, 50, 50])

#Saída:

#Potência:
potencia['PB'] = fuzz.trimf(potencia.universe, [0, 0, 50])
potencia['PM'] = fuzz.trimf(potencia.universe, [0, 50, 100])
potencia['PA'] = fuzz.trimf(potencia.universe, [50, 100, 100])

# Demonstraçã visual dos subconjuntos
vazao.view() #visualização dos subconjuntos da vazão
temperatura.view() #visualização dos subconjuntos da temperatura
potencia.view() #visualização dos subconjuntos da potência


#Definição das fuzzy rules
rule1 = ctrl.Rule(vazao['VB'] & temperatura['TB'] | temperatura['TM'], potencia['PB'])
rule2 = ctrl.Rule(vazao['VB'] & temperatura['TA'], potencia['PM'])
rule3 = ctrl.Rule(vazao['VM'] & temperatura['TB'], potencia['PB'])
rule4 = ctrl.Rule(vazao['VM'] & temperatura['TM'], potencia['PM'])
rule5 = ctrl.Rule(vazao['VM'] & temperatura['TA'], potencia['PA'])
rule6 = ctrl.Rule(vazao['VA'] & temperatura['TB'], potencia['PM'])
rule7 = ctrl.Rule(vazao['VA'] & temperatura['TM'], potencia['PA'])
rule8 = ctrl.Rule(vazao['VA'] & temperatura['TA'], potencia['PA'])

#Criação do sistema de controle e de sua simulação:
potenciaregulada_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7]) #criação do controle fuzzy

potenciaregulada = ctrl.ControlSystemSimulation(potenciaregulada_ctrl) #simulação do controlador


#Simulação do controle:
potenciaregulada.input['vazao'] = 80 #Qual o valor de vazão medido?
potenciaregulada.input['temperatura'] = 20 #Qual a temperatura desejada?

potenciaregulada.compute()

#Demonstração dos resultados
print(potenciaregulada.output['potencia'])
potencia.view(sim=potenciaregulada)