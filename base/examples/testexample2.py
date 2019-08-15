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





def NNsearch(sort, j, q, I_all, h_x, h_y,NN):
    xblock = np.arange(I_all[int(sort[q,1]), 0], I_all[int(sort[q,1]), 0]+(degree+1)*h_x, h_x)
    yblock = np.arange(I_all[int(sort[q,1]), 1], I_all[int(sort[q,1]), 1]+(degree+1)*h_y, h_y)
    block = np.meshgrid(xblock, yblock)
    eval_block = np.zeros(((degree+1)**2, 1))
    s=0
    for i in range(degree+1):
        for t in range(degree+1):
            eval_block[s] = weightfunction.ellipse(radius1, radius2,[block[0][t,i], block[1][t,i]])
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
                eval_block[s] = weightfunction.ellipse(radius1, radius2,[block[0][t,i],block[1][t,i]])
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
                    eval_block[s] = weightfunction.ellipse(radius1, radius2,[block[0][t,i],block[1][t,i]])
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
                        eval_block[s] = weightfunction.ellipse(radius1, radius2,[block[0][t,i],block[1][t,i]])
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
radius = 0.4
radius1 = 0.45    # Radius von Kreis
radius2 = 0.1
degree = 1     # Grad von B-Splines (nur ungerade)
#level_x = 2   # Level in x Richtung    
#level_y = 2     # Level in y Richtung

filename = 'punkte.pkl'
path = '/home/hanausmc/pickle/ellipse'
filepath = os.path.join(path, filename)

with open(filepath, 'rb') as fp:
    data = pickle.load(fp)
    
punkte = data['punkte']



levelymax = 8
for levelx in range(4,8):
    for levely  in range(4,levelymax):
        start = time.time()
        print(levelx,levely)
        level_x = levelx
        level_y = levely  
        
        #data = {}  
        
        filename = 'data_{}_{}_circle_degree_{}.pkl'.format(level_x, level_y, degree)
        path = '/home/hanausmc/pickle/circle'
        filepath = os.path.join(path, filename)
        
        with open(filepath, 'rb') as fp:
            data=pickle.load(fp)
        
        print(data.keys())
         
# #         with open(filepath, 'rb') as fp:
# #             data = pickle.load(fp)
# #             
# #         print(data.keys())
#              
#         data['punkte'] = punkte    
#         data['level_x'] = level_x
#         data['level_y'] = level_y
#         data['degree'] = degree   
#            
#         # Pruefe ob Level hoch genug
#         if level_x and level_y < np.log2(degree+1):
#             print('Error: Level zu niedrig. Es muss Level >= log2(degree+1) sein ')
#             quit()
#            
#         # Gitter fuer Kreis erzeugen und auswerten
#         x0 = np.linspace(0, 1, 50)
#         X = np.meshgrid(x0, x0) 
#         Z = weightfunction.ellipse(radius1, radius2, X)
#            
#         # Gitterweite
#         h_x = 2**(-level_x)
#         h_y = 2**(-level_y)
#            
#         data['h_x'] = h_x
#         data['h_y'] = h_y
#            
#         # Definiere Knotenfolge
#             
#         # Uniform
#         # xi = np.arange(-(degree+1)/2, 1/h_x+(degree+1)/2+1, 1)*h_x
#         # yi = np.arange(-(degree+1)/2, 1/h_y+(degree+1)/2+1, 1)*h_y
#            
#         # Not a Knot
#         xi = np.zeros(2**level_x+degree+1+1)
#         for k in range(2**level_x+degree+1+1):
#             if k in range(degree+1):
#                 xi[k] = (k-degree)*h_x
#             elif k in range(degree+1, 2**level_x+1):
#                 xi[k] = ((k+(degree-1)/2)-degree)*h_x
#             elif k in range(2**level_x+1, 2**level_x+degree+1+1):
#                 xi[k] = ((k+degree-1)-degree)*h_x
#                       
#         yi = np.zeros(2**level_y+degree+1+1)
#         for k in range(2**level_y+degree+1+1):
#             if k in range(degree+1):
#                 yi[k] = (k-degree)*h_y
#             elif k in range(degree+1, 2**level_y+1):
#                 yi[k] = ((k+(degree-1)/2)-degree)*h_y
#             elif k in range(2**level_y+1, 2**level_y+degree+1+1):
#                 yi[k] = ((k+degree-1)-degree)*h_y   
#                    
#         data['xi'] = xi
#         data['yi'] = yi
#            
#            
#         # Index von Bspline auf Knotenfolge
#         index_Bspline_x = np.arange(-(degree-1)/2, len(xi)-3*(degree+1)/2+1, 1)
#         index_Bspline_y = np.arange(-(degree-1)/2, len(yi)-3*(degree+1)/2+1, 1)
#         # print(index_Bspline_x)
#         # print(xi)
#            
#         data['index_Bspline_x'] = index_Bspline_x
#         data['index_Bspline_y'] = index_Bspline_y
#            
#         k=0
#         index_all_Bsplines = np.zeros((len(index_Bspline_x)*len(index_Bspline_y),dim))
#         for i in index_Bspline_x:
#             for j in index_Bspline_y:
#                 index_all_Bsplines[k] = [i,j]
#                 k=k+1
#         #print(index_all_Bsplines)
#            
#         data['index_all_Bsplines'] = index_all_Bsplines
#            
#         # Index (x,y) der Bsplines mit Knotenmittelpunkt im inneren des Gebiets
#         index_inner_Bsplines = np.zeros(dim)
#         index_outer_Bsplines = np.zeros(dim)
#         for i in index_Bspline_x:
#             for j in index_Bspline_y:
#                 if weightfunction.ellipse(radius1, radius2,[xi[int(i+degree)], yi[int(j+degree)]]) > 0:
#                     index_inner_Bsplines = np.vstack((index_inner_Bsplines, [i,j]))
#                 else:
#                     index_outer_Bsplines = np.vstack((index_outer_Bsplines, [i,j]))
#         index_inner_Bsplines = np.delete(index_inner_Bsplines, 0, 0)
#         index_outer_Bsplines = np.delete(index_outer_Bsplines, 0, 0)
#         # print(index_inner_Bsplines)
#         # printLine()
#         #print(index_outer_Bsplines)
#            
#         data['index_inner_Bsplines'] = index_inner_Bsplines
#         data['index_outer_Bsplines'] = index_outer_Bsplines
#            
#         # Pruefe ob genug innere Bsplines vorhanden sind 
#         if len(index_inner_Bsplines) < (degree+1)**2:
#             print('Nicht genug innere Punkte. Erhoehe Level oder Gebiet.')   
#             #quit() 
#            
#         # Definiere Bsplinemittelpunkte als Vektor
#         k=0
#         midpoints = np.zeros((len(index_Bspline_x)*len(index_Bspline_y), dim))
#         for i in index_Bspline_x:
#             for j in index_Bspline_y:
#                 midpoints[k] = [xi[int(i+degree)], yi[int(j+degree)]]
#                 k=k+1
#         #print(midpoints)
#            
#         data['midpoints'] = midpoints
#            
#         # Unterteilung in innere und aeussere Bsplines durch Mittelpunkte der Bsplines
#         I_all = np.zeros((len(index_inner_Bsplines), dim))
#         k=0
#         for i in index_inner_Bsplines:
#             I_all[k] = [xi[int(i[0]+degree)], yi[int(i[1]+degree)]]
#             k=k+1
#         J_all = np.zeros((len(index_outer_Bsplines), dim))
#         k=0
#         for j in index_outer_Bsplines:
#             J_all[k] = [xi[int(j[0]+degree)], yi[int(j[1]+degree)]]
#             k=k+1
#         #print(I_all)
#         #print(J_all)
#            
#         data['I_all'] = I_all
#         data['J_all'] = J_all
#            
#            
#            
#            
#         print("dimensionality:           {}".format(dim))
#         print("level:                    {}".format((level_x, level_y)))
#         print("number of Bsplines:       {}".format(len(index_Bspline_x)*len(index_Bspline_y)))
#            
#         data['dim'] = dim
#         data['Anzahl_Gitterpunkte'] = len(index_Bspline_x)*len(index_Bspline_y)
#            
#         supp_x = np.zeros((degree+2))
#         supp_y = np.zeros((degree+2))
#         index_outer_relevant_Bsplines = np.zeros((dim))
#         for j in range(len(index_outer_Bsplines)):
#             k=0
#             for i in range(-int((degree+1)/2), int((degree+1)/2)+1, 1):
#                 supp_x[k] = xi[int(index_outer_Bsplines[j,0]+i+degree)]
#                 supp_y[k] = yi[int(index_outer_Bsplines[j,1]+i+degree)]
#                 k=k+1 
#                 grid_supp = np.meshgrid(supp_x, supp_y)
#                 eval_supp = np.zeros((len(supp_x), len(supp_y)))
#             for h in range(len(supp_x)):
#                 for g in range(len(supp_y)):
#                     eval_supp[h,g] = weightfunction.ellipse(radius1, radius2, [grid_supp[0][g,h], grid_supp[1][g,h]])
#             if (eval_supp > 0).any():
#                 index_outer_relevant_Bsplines = np.vstack((index_outer_relevant_Bsplines, [index_outer_Bsplines[j]]))
#         index_outer_relevant_Bsplines = np.delete(index_outer_relevant_Bsplines,0,0)
#         #print(index_outer_relevant_Bsplines)
#            
#         data['index_outer_relevant_Bsplines'] = index_outer_relevant_Bsplines
#            
#         J_relevant = np.zeros((len(index_outer_relevant_Bsplines), dim))
#         k=0
#         for j in index_outer_relevant_Bsplines:
#             J_relevant[k] = [xi[int(j[0]+degree)], yi[int(j[1]+degree)]]
#             k=k+1
#         #print(J_relevant)
#            
#         data['J_relevant'] = J_relevant
#            
#         # Index der inneren Bsplines unter Gesamtanzahl (n+1)**2
#         index_I_all = np.zeros(len(I_all))
#         for i in range(len(I_all)):
#             for j in range(len(midpoints)):
#                 if I_all[i, 0] == midpoints[j, 0] and I_all[i, 1] == midpoints[j, 1]:
#                     index_I_all[i] = j
#         #print(index_I_all)
#            
#         data['index_I_all'] = index_I_all
#            
#         # Index der aeusseren Bsplines unter Gesamtanzahl (n+1)**2
#         index_J_all = np.zeros(len(J_all))
#         for i in range(len(J_all)):
#             for j in range(len(midpoints)):
#                 if J_all[i, 0] == midpoints[j, 0] and J_all[i, 1] == midpoints[j, 1]:
#                     index_J_all[i] = j
#         #print(index_J_all)
#            
#         data['index_J_all'] = index_J_all
#            
#         # Index der relevanten aeusseren Bsplines unter Gesamtanzahl (n+1)**2
#         index_J_relevant = np.zeros(len(J_relevant))
#         for i in range(len(J_relevant)):
#             for j in range(len(midpoints)):
#                 if J_relevant[i, 0] == midpoints[j, 0] and J_relevant[i, 1] == midpoints[j, 1]:
#                     index_J_relevant[i] = j
#         #print(index_J_relevant)
#            
#         data['index_J_relevant'] = index_J_relevant
#            
#            
#         #     a = np.meshgrid(xi,yi)
#         #     plt.scatter(a[0],a[1])
#         #     plt.scatter(J_all[:,0], J_all[:,1], c='crimson', s=50, lw=0)
#         #     plt.scatter(I_all[:,0], I_all[:,1], c='mediumblue', s=50, lw=0)
#         #     plt.scatter(J_relevant[:, 0], J_relevant[:, 1], c='goldenrod', s=50, lw=0)
#         #     #plt.show()
#            
#            
#            
#         # Definiere Gitter 
#         x = np.arange(0, 1+h_x, h_x)
#         y = np.arange(0, 1+h_y, h_y)
#         grid = np.meshgrid(x,y)
#            
#         #print(xi)
#         #print(x)
#            
#            
#            
#            
#            
#            
#            
#         ####################### Test 2D Coeffs uniform auf [0,1] mit Identifizierung auf Bspline Mittelpunkte #######################
#            
#         # Definiere Gitterpunkte als Vektor
#         k=0
#         gp = np.zeros((len(x)*len(y), dim))
#         for i in range(len(x)):
#             for j in range(len(y)):
#                  gp[k] = [grid[0][j,i], grid[1][j,i]]
#                  k=k+1
#         #print(gp)
#            
#         data['gp'] = gp
#            
#         # Monome definieren und an allen Knotenmittelpunkten auswerten
#         size_monomials = (degree+1)**2
#         n_neighbors = size_monomials
#         eval_monomials = np.zeros((size_monomials, len(gp)))
#         k = 0
#         for j in range(degree + 1):
#             for i in range (degree + 1):
#                 eval_monomials[k] = (pow(gp[:, 0], i) * pow(gp[:, 1], j))
#                 k = k + 1   
#         eval_monomials = np.transpose(eval_monomials)
#         #print(eval_monomials)
#            
#         data['eval_monomials'] = eval_monomials
#             
#         # Aufstellen der Interpolationsmatrix A_ij = b_j(x_i)
#         A = np.zeros((len(index_Bspline_x)*len(index_Bspline_y), len(gp)))
#         for l in range(len(gp)):
#             k=0
#             for i in index_Bspline_x:
#                 for j in index_Bspline_y:
#                     A[l,k] = Bspline.evalBspline(degree, i, xi, gp[l,0]) * Bspline.evalBspline(degree, j, yi, gp[l,1])
#                     k=k+1        
#         #print(A)
#         #print(A.shape)
#                 
#         data['A'] = A
#         #coeffs = data['coeffs']
#         coeffs = np.linalg.solve(A, eval_monomials)
#         data['coeffs'] = coeffs
#         # Nearest Neighbors bestimmen
#         k=1
#         if k == 0:
#             # Nearest Neighbors nach Abstand
#             distance = np.zeros((len(I_all), dim))
#             NN = np.zeros((len(J_relevant), n_neighbors, dim))
#             for j in range(len(J_relevant)):
#                 for i in range(len(I_all)):
#                     diff = I_all[i] - J_relevant[j]
#                     distance[i, 0] = LA.norm(diff)
#                     distance[i, 1] = i
#                     sort = distance[np.argsort(distance[:, 0])]
#             # Loesche Punkte die Anzahl Nearest Neighbor ueberschreitet
#                 i = len(I_all) - 1
#                 while i >= n_neighbors:
#                     sort = np.delete(sort, i , 0)
#                     i = i - 1
#             # Bestimme die Nearest Neighbor inneren Punkte
#                 for i in range(len(sort)):
#                     NN[j,i] = I_all[int(sort[i,1])]
#                             
#             # Index der nearest neighbor Punkte unter allen Punkten x
#             index_NN = np.zeros((len(J_relevant), n_neighbors))
#             for j in range(NN.shape[0]):
#                 for i in range(NN.shape[1]):
#                     for k in range(len(gp)):
#                         if NN[j, i, 0] == gp[k,0] and NN[j, i, 1] == gp[k,1]:
#                             index_NN[j, i] = k
#                  
#             # Nearest Neighbors sortieren nach Index im Gesamtgitter
#             index_NN=np.sort(index_NN,axis=1)
#                  
#                              
#         elif k == 1:
#             # Nearest Neighbors mit naehestem (n+1)x(n+1) Block
#             distance = np.zeros((len(I_all), dim))
#             NN = np.zeros((len(J_relevant), n_neighbors, dim))
#             for j in range(len(J_relevant)):
#                 for i in range(len(I_all)):
#                     diff = I_all[i] - J_relevant[j]
#                     distance[i, 0] = LA.norm(diff)
#                     distance[i, 1] = i
#                     sort = distance[np.argsort(distance[:, 0])]
#                 NNsearch(sort, j, 0, I_all, h_x, h_y,NN)
#             NN=NN
#                      
#             index_NN = np.zeros((len(J_relevant), n_neighbors))
#             for j in range(NN.shape[0]):
#                 for i in range(NN.shape[1]):
#                     for k in range(len(gp)):
#                         if NN[j, i, 0] == gp[k,0] and NN[j, i, 1] == gp[k,1]:
#                             index_NN[j, i] = k
#                   
#             # Nearest Neighbors sortieren nach Index im Gesamtgitter
#             #index_NN=np.sort(index_NN,axis=1)
#         #print(index_NN)
#            
#         data['NN'] = NN
#         data['index_NN'] = index_NN
#             
#         # Definiere Koeffizientenmatrix der aeusseren relevanten Punkte
#         coeffs_J_relevant = np.zeros((len(J_relevant),1, size_monomials))
#         #print(index_outer_relevant_Bsplines)
#         k=0
#         #print(coeffs_J_relevant)
#         for i in index_J_relevant: #len(x)*(index_outer_relevant_Bsplines[:,0]+(degree-1)/2)+(index_outer_relevant_Bsplines[:,1]+(degree-1)/2):
#             coeffs_J_relevant[k] = coeffs[int(i)]
#             k=k+1
#         coeffs_J_relevant = np.transpose(coeffs_J_relevant, [0,2,1])
#         #print(coeffs_J_relevant)
#            
#         data['coeffs_J_relevant'] = coeffs_J_relevant           
#              
#         # # Definiere Koeffizientenmatrix der nearest neighbors
#         coeffs_NN = np.zeros((len(J_relevant),n_neighbors, size_monomials))
#         for i in range(len(index_NN)):
#             k=0
#             for j in index_NN[i]:
#                 coeffs_NN[i,k] = coeffs[int(j)]
#                 k=k+1
#         coeffs_NN = np.transpose(coeffs_NN, [0,2,1])
#         #print(coeffs_NN)     
#        
#         data['coeffs_NN'] = coeffs_NN
#             
#         # Ueberpruefe ob Determinante der Koeffizientenmatrix der NN ungleich 0
#         #print(np.linalg.det(coeffs_NN) ) 
#         if (np.linalg.det(coeffs_NN) == 0).any():
#             print('Waehle Nearest Neighbors anders, so dass Koeffizientenmatrix der NN nicht singulaer')
#             quit()
#             
#             
#         extension_coeffs = np.zeros((len(J_relevant), size_monomials, 1))
#         for i in range(coeffs_NN.shape[0]):
#             extension_coeffs[i] = np.linalg.solve(coeffs_NN[i], coeffs_J_relevant[i])
#         #print(extension_coeffs.shape)
#             
#         data['extension_coeffs'] = extension_coeffs
#             
#         error_LGS_extension = LA.norm(np.matmul(coeffs_NN, extension_coeffs)- coeffs_J_relevant)
#         if error_LGS_extension > pow(10, -14):
#             print('LGS extension failed. error > 10e-14')
#         print('error_LGS_extension: {}'.format(error_LGS_extension))
#          
#          
#  
#          
#          
#         with open(filepath, 'wb') as fp:
#             pickle.dump(data, fp)
#             #print('saved data to {}'.format(filepath))
#         stop = time.time()
#         duration = stop-start
#         print(duration)

    levelymax=levelymax-1






