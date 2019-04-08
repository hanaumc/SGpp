import pysgpp
import math
import sys
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import weightfunction

    
def printLine():
    print("--------------------------------------------------------------------------------------")
    
radius = 0.4
dim = 2

x1 = np.linspace(0, 1, 50)
X = np.meshgrid(x1, x1)
       
Z = weightfunction.circle(0.4,X)
print("Z={}".format(Z))

# Plot von Kreis
plt.contour(X[0], X[1], Z, colors='black');
plt.axis('equal')

    

dim = 2
radius = 0.4 
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

x = np.random.rand(len(alpha))
y = np.random.rand(len(alpha))
for i in range(len(alpha)):
    x[i] = alpha[i]
    y[i] = beta[i]
print(x)
print(y)
colors = np.random.rand(len(alpha))
#area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
plt.scatter(x, y, c=colors, alpha=0.5)
#plt.show()