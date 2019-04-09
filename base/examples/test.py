import pysgpp
import math
import sys
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import weightfunction


def printLine():
    print("--------------------------------------------------------------------------------------")
    

dim = 2
radius = 0.1 
degree = 2
grid = pysgpp.Grid.createWEBsplineGrid(dim, degree)

gridStorage = grid.getStorage()
print("dimensionality:         {}".format(gridStorage.getDimension()))

level = 3
grid.getGenerator().regular(level)
print("number of grid points:  {}".format(gridStorage.getSize()))

alpha = pysgpp.DataVector(gridStorage.getSize(),0.0)
beta = pysgpp.DataVector(gridStorage.getSize(),0.0)
print("length of alpha vector: {}".format(len(alpha)))
print("length of beta vector: {}".format(len(beta)))

printLine()
for i in range(gridStorage.getSize()):
  gp = gridStorage.getPoint(i)
  alpha[i] = gp.getStandardCoordinate(0)
  beta[i] = gp.getStandardCoordinate(1)

x = np.zeros((len(alpha),dim))
eval_circle= np.zeros(len(alpha))
for i in range(len(alpha)):
    x[i] = [alpha[i],beta[i]]    
    eval_circle[i]=weightfunction.circle(radius, x[i])


printLine()
p0=0
n0=0
for i in range(len(eval_circle)):
    if eval_circle[i] > 0:
        p0=p0+1
    else:
        n0=n0+1
I_j = np.zeros((p0,dim))
J_i = np.zeros((n0,dim))
#print(pos)
#print(neg)
p1=0
n1=0 
for i in range(len(eval_circle)):
    if eval_circle[i] > 0:
        I_j[p1]=x[i]
        p1=p1+1
    else:
        J_i[n1]=x[i]
        n1=n1+1

plt.scatter(I_j[:,0], I_j[:,1], c='red')
plt.scatter(J_i[:,0], J_i[:,1], c='blue')
plt.show()

