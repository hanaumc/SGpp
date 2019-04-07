#import pysgpp
import math
import sys
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt


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
