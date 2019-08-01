import numpy as np
import pickle
import os
import math
import sys
import matplotlib 
import matplotlib.pyplot as plt
import weightfunction
import Bspline
import scipy 
from numpy import linalg as LA
import time


# with open('/home/hanausmc/pickle/circle/punkte.pkl', 'rb') as fp:
#     punkte_loaded = pickle.load(fp)
    
#print(punkte_loaded['punkte'].shape)


dim = 2         # Dimension
radius = 0.4    # Radius von Kreis
degree = 5     # Grad von B-Splines (nur ungerade)
#level_x = 2   # Level in x Richtung    
#level_y = 2     # Level in y Richtung




 
 
levelymax = 7
for levelx in range(6,9):
    for levely  in range(4,levelymax):
        start = time.time()
        print(levelx,levely)
        level_x = levelx
        level_y = levely
          
  
      
   
        # Pruefe ob Level hoch genug
        if level_x and level_y < np.log2(degree+1):
            print('Error: Level zu niedrig. Es muss Level >= log2(degree+1) sein ')
            quit()
            
            
            
        # Gitterweite
        h_x = 2**(-level_x)
        h_y = 2**(-level_y)
            
        # Definiere Knotenfolge
             
        # Not a Knot
        xi = np.zeros(2**level_x+degree+1+1)
        for k in range(2**level_x+degree+1+1):
            if k in range(degree+1):
                xi[k] = (k-degree)*h_x
            elif k in range(degree+1, 2**level_x+1):
                xi[k] = ((k+(degree-1)/2)-degree)*h_x
            elif k in range(2**level_x+1, 2**level_x+degree+1+1):
                xi[k] = ((k+degree-1)-degree)*h_x
                       
        yi = np.zeros(2**level_y+degree+1+1)
        for k in range(2**level_y+degree+1+1):
            if k in range(degree+1):
                yi[k] = (k-degree)*h_y
            elif k in range(degree+1, 2**level_y+1):
                yi[k] = ((k+(degree-1)/2)-degree)*h_y
            elif k in range(2**level_y+1, 2**level_y+degree+1+1):
                yi[k] = ((k+degree-1)-degree)*h_y   
                    
            
        # Index von Bspline auf Knotenfolge
        index_Bspline_x = np.arange(-(degree-1)/2, len(xi)-3*(degree+1)/2+1, 1)
        index_Bspline_y = np.arange(-(degree-1)/2, len(yi)-3*(degree+1)/2+1, 1)
        # print(index_Bspline_x)
        # print(xi)
            
        k=0
        index_all_Bsplines = np.zeros((len(index_Bspline_x)*len(index_Bspline_y),dim))
        for i in index_Bspline_x:
            for j in index_Bspline_y:
                index_all_Bsplines[k] = [i,j]
                k=k+1
        #print(index_all_Bsplines)
            
        # Index (x,y) der Bsplines mit Knotenmittelpunkt im inneren des Gebiets
        index_inner_Bsplines = np.zeros(dim)
        index_outer_Bsplines = np.zeros(dim)
            
        for i in index_Bspline_x:
            for j in index_Bspline_y:
                if weightfunction.circle(radius,[xi[int(i+degree)], yi[int(j+degree)]]) > 0:
                    index_inner_Bsplines = np.vstack((index_inner_Bsplines, [i,j]))
                else:
                    index_outer_Bsplines = np.vstack((index_outer_Bsplines, [i,j]))
        index_inner_Bsplines = np.delete(index_inner_Bsplines, 0, 0)
        index_outer_Bsplines = np.delete(index_outer_Bsplines, 0, 0)
        # print(index_inner_Bsplines)
        # printLine()
        #print(index_outer_Bsplines)
            
        # Pruefe ob genug innere Bsplines vorhanden sind 
        if len(index_inner_Bsplines) < (degree+1)**2:
            print('Nicht genug innere Punkte. Erhoehe Level oder Gebiet.')   
            #quit() 
            
        # Definiere Bsplinemittelpunkte als Vektor
        k=0
        midpoints = np.zeros((len(index_Bspline_x)*len(index_Bspline_y), dim))
        for i in index_Bspline_x:
            for j in index_Bspline_y:
                midpoints[k] = [xi[int(i+degree)], yi[int(j+degree)]]
                k=k+1
        #print(midpoints)
            
        # Unterteilung in innere und aeussere Bsplines durch Mittelpunkte der Bsplines
        I_all = np.zeros((len(index_inner_Bsplines), dim))
        k=0
        for i in index_inner_Bsplines:
            I_all[k] = [xi[int(i[0]+degree)], yi[int(i[1]+degree)]]
            k=k+1
        J_all = np.zeros((len(index_outer_Bsplines), dim))
        k=0
        for j in index_outer_Bsplines:
            J_all[k] = [xi[int(j[0]+degree)], yi[int(j[1]+degree)]]
            k=k+1
        #print(I_all)
        #print(J_all)
            
        #print(index_inner_Bsplines)
            
            
            
            
            
        print("dimensionality:           {}".format(dim))
        print("level:                    {}".format((level_x, level_y)))
        print("number of Bsplines:       {}".format(len(index_Bspline_x)*len(index_Bspline_y)))
            
        supp_x = np.zeros((degree+2))
        supp_y = np.zeros((degree+2))
        index_outer_relevant_Bsplines = np.zeros((dim))
        for j in range(len(index_outer_Bsplines)):
            k=0
            for i in range(-int((degree+1)/2), int((degree+1)/2)+1, 1):
                supp_x[k] = xi[int(index_outer_Bsplines[j,0]+i+degree)]
                supp_y[k] = yi[int(index_outer_Bsplines[j,1]+i+degree)]
                k=k+1 
                grid_supp = np.meshgrid(supp_x, supp_y)
                eval_supp = np.zeros((len(supp_x), len(supp_y)))
            for h in range(len(supp_x)):
                for g in range(len(supp_y)):
                      
                    eval_supp[h,g] = weightfunction.circle(radius, [grid_supp[0][g,h], grid_supp[1][g,h]])
            if (eval_supp > 0).any():
                index_outer_relevant_Bsplines = np.vstack((index_outer_relevant_Bsplines, [index_outer_Bsplines[j]]))
        index_outer_relevant_Bsplines = np.delete(index_outer_relevant_Bsplines,0,0)
        #print(index_outer_relevant_Bsplines)
            
        J_relevant = np.zeros((len(index_outer_relevant_Bsplines), dim))
        k=0
        for j in index_outer_relevant_Bsplines:
            J_relevant[k] = [xi[int(j[0]+degree)], yi[int(j[1]+degree)]]
            k=k+1
        #print(J_relevant)
            
            
        # Index der inneren Bsplines unter Gesamtanzahl (n+1)**2
        index_I_all = np.zeros(len(I_all))
        for i in range(len(I_all)):
            for j in range(len(midpoints)):
                if I_all[i, 0] == midpoints[j, 0] and I_all[i, 1] == midpoints[j, 1]:
                    index_I_all[i] = j
        #print(index_I_all)
            
        # Index der aeusseren Bsplines unter Gesamtanzahl (n+1)**2
        index_J_all = np.zeros(len(J_all))
        for i in range(len(J_all)):
            for j in range(len(midpoints)):
                if J_all[i, 0] == midpoints[j, 0] and J_all[i, 1] == midpoints[j, 1]:
                    index_J_all[i] = j
        #print(index_J_all)
            
        # Index der relevanten aeusseren Bsplines unter Gesamtanzahl (n+1)**2
        index_J_relevant = np.zeros(len(J_relevant))
        for i in range(len(J_relevant)):
            for j in range(len(midpoints)):
                if J_relevant[i, 0] == midpoints[j, 0] and J_relevant[i, 1] == midpoints[j, 1]:
                    index_J_relevant[i] = j
        #print(index_J_relevant)
            
            
        # Definiere Gitter 
        x = np.arange(0, 1+h_x, h_x)
        y = np.arange(0, 1+h_y, h_y)
        grid = np.meshgrid(x,y)
            
        # Definiere Gitterpunkte als Vektor
        k=0
        gp = np.zeros((len(x)*len(y), dim))
        for i in range(len(x)):
            for j in range(len(y)):
                 gp[k] = [grid[0][j,i], grid[1][j,i]]
                 k=k+1
        #print(gp)
            
        # Monome definieren und an allen Knotenmittelpunkten auswerten
        size_monomials = (degree+1)**2
        n_neighbors = size_monomials
        eval_monomials = np.zeros((size_monomials, len(gp)))
        k = 0
        for j in range(degree + 1):
            for i in range (degree + 1):
                eval_monomials[k] = (pow(gp[:, 0], i) * pow(gp[:, 1], j))
                k = k + 1   
        eval_monomials = np.transpose(eval_monomials)
        #print(eval_monomials)
             
        # Aufstellen der Interpolationsmatrix A_ij = b_j(x_i)
        A = np.zeros((len(index_Bspline_x)*len(index_Bspline_y), len(gp)))
        for l in range(len(gp)):
            k=0
            for i in index_Bspline_x:
                for j in index_Bspline_y:
                    A[l,k] = Bspline.evalBspline(degree, i, xi, gp[l,0]) * Bspline.evalBspline(degree, j, yi, gp[l,1])
                    k=k+1        
        #print(A)
        print(A.shape)
            
        data = {'A':A}
        filename = 'data_{}_{}_circle_degree_{}.pkl'.format(level_x, level_y, degree)
        path = '/home/hanausmc/pickle/circle'
        filepath = os.path.join(path, filename)
        with open(filepath, 'wb') as fp:
            pickle.dump(data, fp)
        stop = time.time()
        duration = stop-start
        print('duration: {}'.format(duration))
    levelymax=levelymax-1
    
    
    
    











# 
# 
# # for lx in range(3,8):
# #    for ly in range(3,8):
# 
# # exemplary grid : (2,3)
# degree = 3
# 
# level = [2, 3]
# # levelx = 2
# # levely = 3
# 
# # inner
# I = [1, 2, 3]
# 
# # outer
# J = [0, 0, 9]
# 
# # interpolation matrix
# A = np.zeros((3, 3))
# A[0, 2] = 17
# 
# # extension coefficients
# E = [1, 2, 3, 4]
# 
# data = {'level': level,
#         'I':I,
#         'J':J,
#         'A':A,
#         }
# data['E'] = E
# 
# filename = 'data_{}_{}_circle_degree{}.pkl'.format(level[0], level[1], degree)
# path = '/home/hanausmc/pickle/circle'
# filepath = os.path.join(path, filename)
# 
# with open(filepath, 'wb') as fp:
#     pickle.dump(data, fp)
#     print('saved data to {}'.format(filepath))
# 
# with open(filepath, 'rb') as fp:
#     data_loaded = pickle.load(fp)
#     
# print(data_loaded.keys())
# A_loaded = data_loaded['A']
# print(A_loaded)
