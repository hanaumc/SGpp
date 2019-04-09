#import pysgpp
import math
import sys
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
<<<<<<< HEAD
import weightfunction
from mpl_toolkits import mplot3d
=======
>>>>>>> 2f10010dccbfdf9f992c1abf5a713d9d49f5fbcb

    
def printLine():
    print("--------------------------------------------------------------------------------------")
    
dim = 2         # Dimension
radius = 0.1    # Radius von Kreis
degree = 2      # Grad von B-Splines
level = 4       # Level von Sparse Grid

<<<<<<< HEAD
# Gitter für Kreis erzeugen und auswerten
x1 = np.linspace(0, 1, 50)
X = np.meshgrid(x1, x1)       
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
for i in range(gridStorage.getSize()):
    gp = gridStorage.getPoint(i)
    x[i] = [gp.getStandardCoordinate(0), gp.getStandardCoordinate(1)]    
    eval_circle[i]=weightfunction.circle(radius, x[i])

# Überprüfung auf innere und äußere Punkte 
p0=0
n0=0
for i in range(len(eval_circle)):
    if eval_circle[i] > 0:
        p0=p0+1
    else:
        n0=n0+1
I_j = np.zeros((p0,dim))
J_i = np.zeros((n0,dim))
p1=0
n1=0 
for i in range(len(eval_circle)):
    if eval_circle[i] > 0:
        I_j[p1]=x[i]
        p1=p1+1
    else:
        J_i[n1]=x[i]
        n1=n1+1

# Plot von inneren und äußeren Punkten 
ax = plt.axes(projection='3d')
ax.contour3D(X[0], X[1], Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
plt.scatter(I_j[:,0], I_j[:,1], c='b')
plt.scatter(J_i[:,0], J_i[:,1], c='r')
plt.show()

=======
# x wird Spaltenvektor mit einträgen von Punkt 1 in Zeile 1, Punkt 2 in Zeile 2, usw...

def circle(radius, x):
    w = radius
    for i in range(len(x)):
        w = w - (x[i]-0.5)**2
    return w

# Ab hier testplot    
radius = 0.4
dim = 2

def testcircle(radius, x1,x2):
    w=radius-(x1-0.5)**2-(x2-0.5)**2
    return w

x1 = np.linspace(0, 1, 3)
#x2 = np.linspace(0, 1, 3)

X = np.meshgrid(x1, x1)
print("X={}".format(X))
Z_testcircle = testcircle(0.4, X[0], X[1])
print("Z_testcircle={}".format(Z_testcircle))


#Zweightcircle = np.zeros((len(X),len(Y)))
x = np.zeros((X[1].size, dim))
#print(X[1][1][1])

m=0
for k in range(len(x1)):
  for n in range(len(x1)):
    for j in range(dim):
      x[m,j] = X[j][k][n]
    m=m+1

#print(x)
       
Z_circle = circle(0.4,X)
print("Z_circle={}".format(Z_circle))


#for i in range(len(X)):
#    for j in range(len(Y)):
#        Z_weightcircle[i,j] = 1
        #print(Zweightcircle)
        
    
#Z_weightcircle = weightfunction.circle(0.4,x)
#print("Z_weightcircle={}".format(Z_weightcircle))
#plt.contour(X, Y, Z_testcircle, colors='black');
#plt.show()
>>>>>>> 2f10010dccbfdf9f992c1abf5a713d9d49f5fbcb
