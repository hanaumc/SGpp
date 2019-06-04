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

x1 = np.random.rand((6))
x2 = np.random.rand((5))
x3 = np.random.rand((4))


y1 = [3,4,5,6,7,8]
y2 = [4,5,6,7,8]
y3 = [5,6,7,8]


plt.scatter(y1,x1, c='mediumblue', s=50, lw=0)
plt.plot(y1,x1, c='mediumblue')
plt.scatter(y2,x2, c='crimson', s=50, lw=0)
plt.plot(y2,x2,c='crimson')
plt.scatter(y3,x3, c='green', s=50, lw=0)
plt.plot(y3,x3,c='green')

plt.show()