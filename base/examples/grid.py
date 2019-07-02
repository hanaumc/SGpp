import pysgpp
import math
import sys
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import weightfunction
import scipy 
from scipy import special
from mpl_toolkits import mplot3d
from numpy import linalg as LA, transpose, full
from matplotlib.pyplot import axis


# level_x = 1
# level_y = 2
# dim = 2

def createGrid(level_x, level_y, dim):
    h_x = 2**(-level_x)
    h_y = 2**(-level_y)
    points_x = int(1/h_x+1)
    points_y = int(1/h_y+1)
    x = np.zeros((points_x*points_y, dim))
    ind = np.zeros((len(x),dim))
    lvl = np.zeros((len(x),dim))
    
    x0=0
    for j in range(points_y):
        for i in range(points_x):
            x[x0] = [i*h_x, j*h_y]
            ind[x0] = [i,j]
            lvl[x0] = [level_x, level_y]
            x0=x0+1
    return(x)

          
    
    