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

#print("alpha: {}".format(alpha))
#print("beta: {}".format(beta))

x = np.zeros((len(alpha),dim))
eval_circle= np.zeros(len(alpha))
for i in range(len(alpha)):
    x[i] = [alpha[i],beta[i]]    
    eval_circle[i]=weightfunction.circle(radius, x[i])
print(x)
print(x[:,0])
print(eval_circle)
#colors = np.random.rand(len(alpha))
#area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
#plt.scatter(x, y, c=colors, alpha=0.5)
#plt.show()

printLine()
p=0
n=0
for i in range(len(eval_circle)):
    if eval_circle[i] > 0:
        p=p+1
        print("pos")
    else:
        n=n+1
        print("neg")
print(p)    
print(n)
    





