import pysgpp
import math
import sys
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import weightfunction
from mpl_toolkits import mplot3d
from numpy import linalg as LA



def nearneighbors(dim, n_neighbors, I_all, J_relevant):
    diff = np.zeros((len(I_all), dim))
    distance = np.zeros((len(I_all),dim))
    j=0 #(NN z.B. für J_relevant[0])
    for i in range(len(I_all)):
        diff[i] = I_all[i]-J_relevant[j]
        distance[i,0] = LA.norm(diff[i])
        distance[i,1] = i
        sort=distance[np.argsort(distance[:,0])]
# Lösche Punkte die Anzahl Nearest Neighbor überschreitet
    i = len(I_all)-1
    while i >= n_neighbors:
        sort = np.delete(sort, i , 0)
        i = i-1
# Bestimme die Nearest Neighbor inneren Punkte 
    NN = np.zeros((len(sort), dim))
    for i in range(len(sort)):
        NN[i] = I_all[int(sort[i,1])]
    return NN
#    plt.scatter(I_all[:,0], I_all[:,1], c='mediumblue',s=50,lw=0)
#    plt.scatter(J_relevant[:,0], J_relevant[:,1], c='goldenrod',s=50,lw=0)
#    plt.scatter(J_relevant[j,0], J_relevant[j,1], c='cyan',s=50,lw=0)
#    plt.scatter(NN[:,0], NN[:,1], c='limegreen',s=50,lw=0)
# Index NN Punkte in Gesamtgitterpunkte x
#    index_NN = np.zeros(n_neighbors)
#    for i in range(len(NN)):
#        for j in range(len(x)):
#            if NN[i,0]==x[j,0] and NN[i,1]==x[j,1]:
#                index_NN[i] = j
#    print('Index_NN = {}'.format(index_NN))
    



