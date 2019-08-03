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

# Definieren der 1. Zielfunktion
def function_1(x):
    f = 0
    f = np.sin(8* x[0]) + np.sin(7 * x[1])
    f = f * weightfunction.circle(radius, x)
    return f
def function_2(x):
    f = 0
    f = np.sin(np.pi*x[0]*x[1])
    f = f * weightfunction.circle(radius,x)
    return f

def NNsearch(sort, j, q, I_all, h_x, h_y,NN):
    xblock = np.arange(I_all[int(sort[q,1]), 0], I_all[int(sort[q,1]), 0]+(degree+1)*h_x, h_x)
    yblock = np.arange(I_all[int(sort[q,1]), 1], I_all[int(sort[q,1]), 1]+(degree+1)*h_y, h_y)
    block = np.meshgrid(xblock, yblock)
    eval_block = np.zeros(((degree+1)**2, 1))
    s=0
    for i in range(degree+1):
        for t in range(degree+1):
            eval_block[s] = weightfunction.circle(radius,[block[0][t,i], block[1][t,i]])
            s=s+1
    if np.all(eval_block>0) == True:
        s=0
        for i in range(degree+1):
            for t in range(degree+1):
                NN[j,s] = [block[0][t,i], block[1][t,i]]
                s=s+1
    else:
        xblock = np.arange(I_all[int(sort[q,1]), 0], I_all[int(sort[q,1]), 0]+(degree+1)*h_x, h_x)
        yblock = np.arange(I_all[int(sort[q,1]), 1], I_all[int(sort[q,1]), 1]-(degree+1)*h_y, -h_y)
        block = np.meshgrid(xblock, yblock)
        s=0
        for i in range(degree+1):
            for t in range(degree+1):
                eval_block[s] = weightfunction.circle(radius,[block[0][t,i],block[1][t,i]])
                s=s+1
        if np.all(eval_block>0) == True:
            s=0
            for i in range(degree+1):
                for t in range(degree+1):
                    NN[j,s] = [block[0][t,i], block[1][t,i]]
                    s=s+1
        else:
            xblock = np.arange(I_all[int(sort[q,1]), 0], I_all[int(sort[q,1]), 0]-(degree+1)*h_x, -h_x)
            yblock = np.arange(I_all[int(sort[q,1]), 1], I_all[int(sort[q,1]), 1]-(degree+1)*h_y, -h_y)
            block = np.meshgrid(xblock, yblock)
            s=0
            for i in range(degree+1):
                for t in range(degree+1):
                    eval_block[s] = weightfunction.circle(radius,[block[0][t,i],block[1][t,i]])
                    s=s+1
            if np.all(eval_block>0) == True:
                s=0
                for i in range(degree+1):
                    for t in range(degree+1):
                        NN[j,s] = [block[0][t,i], block[1][t,i]]
                        s=s+1
            else:
                xblock = np.arange(I_all[int(sort[q,1]), 0], I_all[int(sort[q,1]), 0]-(degree+1)*h_x, -h_x)
                yblock = np.arange(I_all[int(sort[q,1]), 1], I_all[int(sort[q,1]), 1]+(degree+1)*h_y, h_y)
                block = np.meshgrid(xblock, yblock)
                s=0
                for i in range(degree+1):
                    for t in range(degree+1):
                        eval_block[s] = weightfunction.circle(radius,[block[0][t,i],block[1][t,i]])
                        s=s+1
                if np.all(eval_block>0) == True:
                    s=0
                    for i in range(degree+1):
                        for t in range(degree+1):
                            NN[j,s] = [block[0][t,i], block[1][t,i]]
                            s=s+1
                elif q == len(I_all)-1:
                    print('Fehler: kein (n+1)x(n+1) Block im Gebiet gefunden. Erhoehe Level.')
                    quit()
                else:
                    NNsearch(sort, j, q+1, I_all, h_x, h_y,NN)
    return NN

dim = 2         # Dimension
radius = 0.4    # Radius von Kreis
degree = 5     # Grad von B-Splines (nur ungerade)
#level_x = 2   # Level in x Richtung    
#level_y = 2     # Level in y Richtung
 
levelymax = 8
for levelx in range(4,8):
    for levely  in range(4,levelymax):
        start = time.time()
        print(levelx,levely)
        level_x = levelx
        level_y = levely   
 
        filename = 'data_{}_{}_circle_degree_{}.pkl'.format(level_x, level_y, degree)
        path = '/home/hanausmc/pickle/circle'
        filepath = os.path.join(path, filename)
            
        with open(filepath, 'rb') as fp:
            data = pickle.load(fp)
               
        dim = data['dim']
        degree = data['degree']
        level_x = data['level_x'] 
        level_y = data['level_y'] 
               
           
        # Pruefe ob Level hoch genug
        if level_x and level_y < np.log2(degree+1):
            print('Error: Level zu niedrig. Es muss Level >= log2(degree+1) sein ')
            quit()
           
        # Gitter fuer Kreis erzeugen und auswerten
        x0 = np.linspace(0, 1, 50)
        X = np.meshgrid(x0, x0) 
        Z = weightfunction.circle(radius, X)
        h_x = data['h_x'] 
        h_y = data['h_y'] 
           
        # Definiere Knotenfolge
        # Not a Knot
        xi = data['xi'] 
        yi = data['yi'] 
           
           
        # Index von Bspline auf Knotenfolge        
        index_Bspline_x = data['index_Bspline_x'] 
        index_Bspline_y = data['index_Bspline_y']
       
           
        index_all_Bsplines = data['index_all_Bsplines'] 
           
        # Index (x,y) der Bsplines mit Knotenmittelpunkt im inneren des Gebiets        
        index_inner_Bsplines = data['index_inner_Bsplines']  
        index_outer_Bsplines = data['index_outer_Bsplines'] 
           
        # Pruefe ob genug innere Bsplines vorhanden sind 
        if len(index_inner_Bsplines) < (degree+1)**2:
            print('Nicht genug innere Punkte. Erhoehe Level oder Gebiet.')   
            #quit() 
           
        # Definiere Bsplinemittelpunkte als Vektor        
        midpoints = data['midpoints']
           
        # Unterteilung in innere und aeussere Bsplines durch Mittelpunkte der Bsplines       
        I_all = data['I_all'] 
        J_all = data['J_all'] 
                  
        print("dimensionality:           {}".format(dim))
        print("level:                    {}".format((level_x, level_y)))
        print("number of Bsplines:       {}".format(len(index_Bspline_x)*len(index_Bspline_y)))
   
        index_outer_relevant_Bsplines = data['index_outer_relevant_Bsplines'] 
                
        # Relevante aussere Punkte
        J_relevant = data['J_relevant'] 
           
        # Index der inneren Bsplines unter Gesamtanzahl (n+1)**2        
        index_I_all = data['index_I_all']
           
        # Index der aeusseren Bsplines unter Gesamtanzahl (n+1)**2        
        index_J_all = data['index_J_all'] 
           
        # Index der relevanten aeusseren Bsplines unter Gesamtanzahl (n+1)**2        
        index_J_relevant = data['index_J_relevant']     
           
           
        # Definiere Gitter 
        x = np.arange(0, 1+h_x, h_x)
        y = np.arange(0, 1+h_y, h_y)
        grid = np.meshgrid(x,y)
           
        # Definiere Gitterpunkte als Vektor        
        gp = data['gp'] 
           
        # Monome definieren und an allen Knotenmittelpunkten auswerten
        size_monomials = (degree+1)**2
        n_neighbors = size_monomials
        eval_monomials = data['eval_monomials'] 
           
        # Interpolationsmatrix A
        A = data['A']
           
        # Koeffizienten der Interpolation, erhalten aus A*coeffs = eval_monomials
        coeffs = data['coeffs']
           
        # Nearest Neighbors fuer NN Block
        NN = data['NN']
           
        # Index der NN unter allen Punkten
        index_NN = data['index_NN'] 
            
        # Definiere Koeffizientenmatrix der aeusseren relevanten Punkte        
        coeffs_J_relevant = data['coeffs_J_relevant']
             
        # Definiere Koeffizientenmatrix der nearest neighbors
        coeffs_NN = data['coeffs_NN']
            
        # Ueberpruefe ob Determinante der Koeffizientenmatrix der NN ungleich 0 
        if (np.linalg.det(coeffs_NN) == 0).any():
            print('Waehle Nearest Neighbors anders, so dass Koeffizientenmatrix der NN nicht singulaer')
            quit()
            
            
   
        # Extension coeffs erhalten aus coeffs_NN*extension_coeffs = coeffs_J_relevant 
        extension_coeffs = data['extension_coeffs']
            
        error_LGS_extension = LA.norm(np.matmul(coeffs_NN, extension_coeffs)- coeffs_J_relevant)
        if error_LGS_extension > pow(10, -14):
            print('LGS extension failed. error > 10e-14')
        print('error_LGS_extension: {}'.format(error_LGS_extension))
           
           
#         A_WEB = np.zeros((len(I_all),len(I_all)))
#         for p in range(len(I_all)):  
#             # Definiere J(i)
#             extended_Bspline = 0
#             c=0        
#             for i in index_I_all: 
#                 J_i = np.zeros(1)
#                 index_NN_relevant = np.zeros(1)
#                 bi = Bspline.evalBspline(degree, index_all_Bsplines[int(i),0], xi, I_all[p,0])*Bspline.evalBspline(degree, index_all_Bsplines[int(i),1], yi, I_all[p,1])
#                 for k in range(len(index_NN)): # Definiere 
#                     if (i == index_NN[k]).any():
#                         J_i = np.hstack((J_i, index_J_relevant[k]))
#                                 
#                         for l in range(index_NN.shape[1]):
#                             if i == index_NN[k,l]:
#                                 index_NN_relevant = np.hstack((index_NN_relevant, l))
#                         #print(index_J_relevant[j])
#                 J_i = np.delete(J_i, 0)
#                 index_NN_relevant = np.delete(index_NN_relevant, 0)
#                 #print(J_i)
#                 #print(index_NN_relevant)
#                 g=0
#                 inner_sum = 0
#                 for j in J_i:
#                     for t in range(len(index_J_relevant)):
#                         if j == index_J_relevant[t]:
#                             inner_sum = inner_sum + extension_coeffs[t, int(index_NN_relevant[g])]*Bspline.evalBspline(degree, index_all_Bsplines[int(j),0], xi, I_all[p,0])*Bspline.evalBspline(degree, index_all_Bsplines[int(j),1], yi, I_all[p,1]) 
#                             #print(extension_coeffs[t])#, int(index_NN_relevant[g])])
#                             #print(extension_coeffs[t, int(index_NN_relevant[g])])
#                             #print(index_NN_relevant[g])
#                             g=g+1
#                         
#                 extended_Bspline = extended_Bspline + ( bi + inner_sum)
#                 A_WEB[p,c] = weightfunction.circle(radius, I_all[p])*extended_Bspline
#                 c=c+1          
#                 
#         #print(A_WEB.shape)
#             
#         data['A_WEB'] = A_WEB

        A_WEB = data['A_WEB']
           
        b_2 = np.zeros((len(I_all),1))
        for i in range(len(I_all)):
            b_2[i] = function_2(gp[int(index_I_all[i])])
        #print(b)
           
        data['b_2'] = b_2
           
        alpha_2 = np.linalg.solve(A_WEB, b_2)
        #print(alpha)
        data['alpha_2'] = alpha_2
           
        error_LGS_WEBspline = LA.norm(np.matmul(A_WEB, alpha_2)- b_2)
        if error_LGS_WEBspline > pow(10, -14):
            print('LGS WEBspline failed. error > 10e-14')
        print('error_LGS_WEBspline: {}'.format(error_LGS_WEBspline))
           
           
        with open(filepath, 'wb') as fp:
            pickle.dump(data, fp)
            #print('saved data to {}'.format(filepath))
        stop = time.time()
        duration = stop-start
        print('duration: {}'.format(duration))

 
    levelymax=levelymax-1








#            
#         data = {'A':A}
#         filename = 'data_{}_{}_circle_degree_{}.pkl'.format(level_x, level_y, degree)
#         path = '/home/hanausmc/pickle/circle'
#         filepath = os.path.join(path, filename)
#         with open(filepath, 'wb') as fp:
#             pickle.dump(data, fp)
#         stop = time.time()
#         duration = stop-start
#         print('duration: {}'.format(duration))

    
    






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
