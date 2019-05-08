import pysgpp
import math
import sys
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt

degree = 3

x = np.arange(0,3)
y = np.arange(3,6)
size = np.int((degree+1)*(degree+2)/2)

l = np.zeros((size, 3))

#l.setColumn(1,t)

k=0
for i in range(degree+1):
    for j in range (degree+1):
        if i+j<=3:
            #print(pow(x,i)*pow(y,j))
            l[k]=(pow(x,i)*pow(y,j))
            k=k+1
        
l=np.transpose(l)
print(l)        
print(l[:,0])
eval = pysgpp.DataMatrix(3, size)
for j in range(size):
    for i in range(degree):
        eval.set(i,j,l[i,j])

print(eval)
