import numpy as np
import pickle
import os
import math
import sys
import matplotlib 
import matplotlib.pyplot as plt
import weightfunction
import Bspline
import scipy 
from numpy import linalg as LA
import time


####### Abspeichern von f(x) an allen 5000 Punkten ########

def function_1(x):
    f = 0
    f = np.sin(8* x[0]) + np.sin(7 * x[1])
    f = f * weightfunction.circle(radius, x)
    return f
def function_2(x):
    f = 0
    f = np.sin(np.pi*x[0]*x[1])
    f = f * weightfunction.circle(radius,x)
    return f


degree = 5
radius = 0.4

levelymax = 8
for levelx in range(4,8):
    for levely  in range(4,levelymax):
        start = time.time()
        print(levelx,levely)
        level_x = levelx
        level_y = levely       
     
        filename = 'data_{}_{}_circle_degree_{}.pkl'.format(level_x, level_y, degree)
        path = '/home/hanausmc/pickle/circle'
        filepath = os.path.join(path, filename)
              
        with open(filepath, 'rb') as fp:
            data = pickle.load(fp)
        
        eval_function_2 = np.zeros((len(data['punkte']),1))
        for i in range(len(data['punkte'])):
            eval_function_2[i,0] = function_2(data['punkte'][i])
        
        
        data['eval_function_2'] = eval_function_2
               
        with open(filepath, 'wb') as fp:
            pickle.dump(data, fp)
            #print('saved data to {}'.format(filepath))       
        print(data.keys())

    levelymax=levelymax-1
