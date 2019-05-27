import pysgpp
import math
import sys
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import weightfunction
from mpl_toolkits import mplot3d
from numpy import linalg as LA


def printLine():
    print("--------------------------------------------------------------------------------------")


x = np.arange(4,8,1)
y = np.zeros(4)
y[0]=3
y[1]=4
y[2]=3
y[3]=5
plt.plot(x,y)
plt.show()
