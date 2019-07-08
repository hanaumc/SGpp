import pysgpp
import math
import sys
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import weightfunction
import grid
import scipy 
from scipy import special
from mpl_toolkits import mplot3d
from numpy import linalg as LA, transpose, full
from matplotlib.pyplot import axis


def printLine():
    print("--------------------------------------------------------------------------------------")

def evalBspline(n, k, xi, x):
    if n == 0:
        return 1 if xi[k] <= x < xi[k+1] else 0
    if xi[k+n] == xi[k]:
        c1 = 0
    else:
        c1 = (x-xi[k])/(xi[k+n]-xi[k]) * evalBspline(n-1, k, xi, x)
    if xi[k+n+1] == xi[k+1]:
        c2 = 0
    else:
        c2 = (1-(x-xi[k+1])/(xi[k+n+1]-xi[k+1])) * evalBspline(n-1, k+1, xi, x)
    return c1 + c2






def B(x, k, i, t):
    if k == 0:
        return 1.0 if t[i] <= x < t[i+1] else 0.0
    if t[i+k] == t[i]:
        c1 = 0.0
    else:
       c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
    if t[i+k+1] == t[i+1]:
       c2 = 0.0
    else:
       c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
    return c1 + c2