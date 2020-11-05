"""
GA aplicado para encontrar os ganhos de um controlador PID

Alunos: Hugo Felipe Ferreira
         Lucas Mendon�a Andrade
"""

import control as clt
from scipy.signal import lti, step
import numpy as np
from geneticalgorithm import geneticalgorithm as ga


#Parâmetros do controlador
#Kp = x[0]
#Ki = x[1]
#Kd = x[2]

#Pontos de resolução
dots = 5000

	
#Processo
k = 2
tal = 4

num = [k]
den = [tal,1]

#FT de H(s)
H = clt.TransferFunction(num,den)


def f(x):
    controler = clt.TransferFunction([x[2], x[0], x[1]],[1,0])
    sys = clt.feedback(controler*H) #fechando a malha
    #Aplica degrau
    sys2 = sys.returnScipySignalLTI()[0][0]
    t2,y2 = step(sys2,N = dots)
    return abs(1-y2[-1]) #retorna o erro

varbound=np.array([[0,10],[0,10],[0,10]])
vartype=np.array([['real'],['real'],['real']])

algorithm_param = {'max_num_iteration': 3000,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}
    
    
model=ga(function=f,dimension=3,variable_type='real',variable_boundaries=varbound)

model.run()


