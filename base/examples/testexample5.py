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

def function_1_ellipse(x):
    f = 0
    f = np.sin(8* x[0]) + np.sin(7 * x[1])
    f = f * weightfunction.ellipse(radius1, radius2, x)
    return f
def function_2_ellipse(x):
    f = 0
    f = np.sin(np.pi*x[0]*x[1])
    f = f * weightfunction.ellipse(radius1,radius2,x)
    return f


degree = 3
radius = 0.4
radius1 = 0.45
radius2 = 0.1
# 
# level_x = 3
# level_y = 4
# 
# filename = 'data_{}_{}_circle_degree_{}.pkl'.format(level_x, level_y, degree)
# path = '/home/hanausmc/pickle/circle'
# filepath = os.path.join(path, filename)
# with open(filepath, 'rb') as fp:
#     data = pickle.load(fp)
# 
# print(data['alpha_1'].shape) 
# print(data['alpha_1'])
# print(data['I_all'].shape)      
# 
# 
# level_x = 3
# level_y = 5
# 
# filename = 'data_{}_{}_circle_degree_{}.pkl'.format(level_x, level_y, degree)
# path = '/home/hanausmc/pickle/circle'
# filepath = os.path.join(path, filename)
# with open(filepath, 'rb') as fp:
#     data = pickle.load(fp)    
# 
# print(data['alpha_1'].shape) 
# print(data['alpha_1'])



# levelymax = 9
# for levelx in range(4,9):
#     for levely  in range(4,levelymax):
#         start = time.time()
#         print(levelx,levely)
#         level_x = levelx
#         level_y = levely       
#           
# #         filename = 'punkte.pkl'
# #         path = '/home/hanausmc/pickle/circle'
# #         filepath = os.path.join(path, filename)
# #                 
# #         with open(filepath, 'rb') as fp:
# #             data = pickle.load(fp)
# #         punkte = data['punkte']
#          
#         filename = 'data_{}_{}_circle_degree_{}.pkl'.format(level_x, level_y, degree)
#         path = '/home/hanausmc/pickle/circle'
#         filepath = os.path.join(path, filename)
#         with open(filepath, 'rb') as fp:
#             data = pickle.load(fp)
#         #print(data['eval_Interpolation_1'].shape)
#         #print(data['punkte'].shape)
#              
#         #print(data['eval_function_1'].shape)
#         #print(data['eval_function_1'])
#         #print(data['eval_function_1'])
#         print(data['eval_Interpolation'])
#          
#  
#              
#          
# #         eval_function_2 = np.zeros((len(data['punkte']),1))
# #         for i in range(len(data['punkte'])):
# #             eval_function_2[i,0] = function_2(data['punkte'][i])
# #           
# #           
# #         data['eval_function_2'] = eval_function_2
# #                  
# #         with open(filepath, 'wb') as fp:
# #             pickle.dump(data, fp)
# #             print('saved data to {}'.format(filepath))       
#         #print(data.keys())
#         #print(data['eval_Interpolation'].shape)
#  
#     levelymax=levelymax-1
# 
levelymax = 9
for levelx in range(4,9):
    for levely  in range(4,levelymax):
        start = time.time()
        print(levelx,levely)
        level_x = levelx
        level_y = levely       
           
#         filename = 'punkte.pkl'
#         path = '/home/hanausmc/pickle/circle'
#         filepath = os.path.join(path, filename)
#                
#         with open(filepath, 'rb') as fp:
#             data = pickle.load(fp)
#         punkte = data['punkte']
   
           
        filename = 'data_{}_{}_ellipse_degree_{}.pkl'.format(level_x, level_y, degree)
        path = '/home/hanausmc/pickle/ellipse'
        filepath = os.path.join(path, filename)
        with open(filepath, 'rb') as fp:
            data = pickle.load(fp)
        #print(data.keys()) 
 
 
        #print(data['eval_Interpolation_2'].shape)
        #print(data['eval_function_1'].shape)
        #print(data['eval_function_1'])
                
            
        eval_function_1 = np.zeros((len(data['punkte']),1))
        for i in range(len(data['punkte'])):
            eval_function_1[i,0] = function_1_ellipse(data['punkte'][i])
           
           
        data['eval_function_1'] = eval_function_1
                  
        with open(filepath, 'wb') as fp:
            pickle.dump(data, fp)
#             print('saved data to {}'.format(filepath))       
        #print(data.keys())
        #print(data['eval_Interpolation'].shape)
   
    levelymax=levelymax-1
