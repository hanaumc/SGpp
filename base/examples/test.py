import pysgpp
import math
import sys
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import weightfunction
from mpl_toolkits import mplot3d

    
def printLine():
    print("--------------------------------------------------------------------------------------")
    
dim = 2         # Dimension
radius = 0.1    # Radius von Kreis
degree = 3      # Grad von B-Splines
level = 5       # Level von Sparse Grid

# Gitter für Kreis erzeugen und auswerten
x0 = np.linspace(0, 1, 50)
if dim == 2:
    X = np.meshgrid(x0, x0)
elif dim == 3:
    X = np.meshgrid(x0, x0, x0)
      
Z = weightfunction.circle(0.4,X)

# Plot von Kreis
#plt.contour(X[0], X[1], Z, colors='black');
#plt.axis('equal')

# Erzeugen von Gitter
grid = pysgpp.Grid.createWEBsplineGrid(dim, degree)
gridStorage = grid.getStorage()
print("dimensionality:         {}".format(gridStorage.getDimension()))
grid.getGenerator().regular(level)
print("number of grid points:  {}".format(gridStorage.getSize()))

# Vektor 'x' enthält Koordinaten von Gitterpunkten
# anschl. auswerten von Gewichtsfunktion des Kreises an Gitterpunkten
x = np.zeros((gridStorage.getSize(),dim))
eval_circle= np.zeros(gridStorage.getSize())

if dim == 2:
    for i in range(gridStorage.getSize()):
        gp = gridStorage.getPoint(i)
        x[i] = [gp.getStandardCoordinate(0), gp.getStandardCoordinate(1)]   
        eval_circle[i]=weightfunction.circle(radius, x[i])
elif dim == 3:
    for i in range(gridStorage.getSize()):
        gp = gridStorage.getPoint(i)
        x[i] = [gp.getStandardCoordinate(0), gp.getStandardCoordinate(1), gp.getStandardCoordinate(2)]
        eval_circle[i]=weightfunction.circle(radius, x[i])            

# Überprüfung auf innere und äußere Punkte 
p0=0
n0=0
for i in range(len(eval_circle)):
    if eval_circle[i] > 0:
        p0=p0+1
    else:
        n0=n0+1
I_all = np.zeros((p0,dim))
J_all = np.zeros((n0,dim))
p1=0
n1=0 
for i in range(len(eval_circle)):
    if eval_circle[i] > 0:
        I_all[p1]=x[i]
        p1=p1+1
    else:
        J_all[n1]=x[i]
        n1=n1+1

# Plot von inneren und äußeren Punkten 
#ax = plt.axes(projection='3d')
#ax.contour3D(X[0], X[1], Z, 50, cmap='binary')
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z');

if dim == 2:
    #ax = plt.axes(projection='3d')
    #ax.contour3D(X[0], X[1], Z, 50, cmap='binary')
    plt.scatter(I_all[:,0], I_all[:,1], c='b')
    plt.scatter(J_all[:,0], J_all[:,1], c='r')
elif dim == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(I_all[:,0], I_all[:,1], I_all[:,2], c='red')
    ax.scatter(J_all[:,0], J_all[:,1], J_all[:,2], c='blue')
#plt.show()

# Bestimme Gitterweite h
h = 2**(-level)
print(J_all)
print((degree-1)*h)

# Bestimme Eckpunkte von Träger von Punkt (x,y)
J_relevant = np.zeros(dim)
print(J_relevant)
if dim == 2:
    for i in range(len(J_all)):
        if weightfunction.circle(radius, J_all[i]-(degree-1)*h ) > 0 or weightfunction.circle(radius, [J_all[i,0]-(degree-1)*h, J_all[i,1]+(degree-1)*h]) > 0 or weightfunction.circle(radius, [J_all[i,0]+(degree-1)*h, J_all[i,1]-(degree-1)*h]) > 0 or weightfunction.circle(radius, J_all[i]+(degree-1)*h) > 0: 
            J_relevant = np.vstack((J_relevant, J_all[i]))            
J_relevant = np.delete(J_relevant, 0, 0)
        
        
    

print(J_relevant)
print(len(J_relevant))
if dim == 2:
    #ax = plt.axes(projection='3d')
    #ax.contour3D(X[0], X[1], Z, 50, cmap='binary')
    plt.scatter(I_all[:,0], I_all[:,1], c='b')
    plt.scatter(J_relevant[:,0], J_relevant[:,1], c='y')
    plt.scatter(J_relevant[5,0], J_relevant[5,1], c='c')
#plt.show()
# Punkt 5 ist der links oben für den nearest neighbors getestet wird.

pts = np.array([[0, 0], [2.1, 2.9], [5,3]])


