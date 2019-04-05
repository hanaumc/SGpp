import pysgpp
import math
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import weightfunction
from array import array


radius = 0.4
dim = 2

def circle(radius, x1,x2):
    w=radius-(x1-0.5)**2-(x2-0.5)**2
    return w

x1 = np.linspace(0, 1, 3)
x2 = np.linspace(0, 1, 3)

X, Y = np.meshgrid(x1, x2)
print("X={}".format(X))
print("Y={}".format(Y))
#print("Xausg=".format(X[[1,1]]))
Z_testcircle = circle(0.4, X, Y)
#Z = test(X,Y)
print("Z_testcircle={}".format(Z_testcircle))

Zweightcircle = np.zeros((len(X),len(Y)))
x = np.zeros((X.size, dim))

for i in range(len(x)):
    for j in range(dim):# hier hängt es 
        x[i][j]= j
print(x)

#for i in range(X.size):
 #   for j in range(len(x)):
  #      print(x[i][j])

        

#for i in range(len(X)):
#    for j in range(len(Y)):
#        Z_weightcircle[i,j] = 1
        #print(Zweightcircle)
        
    
#Z_weightcircle = weightfunction.circle(0.4,x)
#print("Z_weightcircle={}".format(Z_weightcircle))
#plt.contour(X, Y, Z, colors='black');
#plt.show()

