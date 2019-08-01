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

# degree =  3
# 
# subfunc = {}
# 
# levelymax = 9
# for levelx in range(3,9):
#     for levely  in range(3,levelymax):
#         start = time.time()
#         print(levelx,levely)
#         level_x = levelx
#         level_y = levely  
#         filename = 'data_{}_{}_circle_degree_{}.pkl'.format(level_x, level_y, degree)
#         path = '/home/hanausmc/pickle/circle'
#         filepath = os.path.join(path, filename)
#              
#         with open(filepath, 'rb') as fp:
#             data = pickle.load(fp)
#             
#         subfunc[level_x,level_y,degree]  = data['punkte']
#     levelymax=levelymax-1
# 
# for i in range(3,9):
#     print(subfunc[i,3,3])



k = 2
l = 4
for i in range(3):
    print(k,l)
    k=k+1
    l=l-1