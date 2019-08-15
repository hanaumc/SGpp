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

radius1 = 0.45
radius2 = 0.1
degree = 3



# levelymax = 9
# for levelx in range(3,9):
#     for levely  in range(3,levelymax):
#         start = time.time()
#         print(levelx,levely)
#         level_x = levelx
#         level_y = levely       
#       
#         filename = 'data_{}_{}_circle_degree_{}.pkl'.format(level_x, level_y, degree)
#         path = '/home/hanausmc/pickle/circle'
#         filepath = os.path.join(path, filename)
#                 
#         with open(filepath, 'rb') as fp:
#             data = pickle.load(fp)
#         #print(data.keys())
#         print(data['eval_Interpolation_2'].shape)
#         print(data['webspline'])
#     levelymax = levelymax-1





# 
# filename = 'punkte.pkl'
# path = '/home/hanausmc/pickle/ellipse'
# filepath = os.path.join(path, filename)
# 
# with open(filepath, 'rb') as fp:
#     data = pickle.load(fp)
#     
# punkte = data['punkte']



def NNsearch(sort, j, q):
    xblock = np.arange(I_all[int(sort[q,1]), 0], I_all[int(sort[q,1]), 0]+(degree+1)*h_x, h_x)
    yblock = np.arange(I_all[int(sort[q,1]), 1], I_all[int(sort[q,1]), 1]+(degree+1)*h_y, h_y)
    block = np.meshgrid(xblock, yblock)
    eval_block = np.zeros(((degree+1)**2, 1))
    s=0
    for i in range(degree+1):
        for t in range(degree+1):
            eval_block[s] = weightfunction.ellipse(radius1,radius2,[block[0][t,i], block[1][t,i]])
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
                    NNsearch(sort, j, q+1)
    return NN
 
 
dim = 2         # Dimension
radius1 = 0.45    # Radius von Kreis
radius2 = 0.1
degree = 5      # Grad von B-Splines (nur ungerade)
level_x = 4   # Level in x Richtung    
level_y = 4     # Level in y Richtung
 
# Pruefe ob Level hoch genug
if level_x and level_y < np.log2(degree+1):
    print('Error: Level zu niedrig. Es muss Level >= log2(degree+1) sein ')
    quit()
 
# Gitter fuer Kreis erzeugen und auswerten
x0 = np.linspace(0, 1, 100)
X = np.meshgrid(x0, x0) 
Z = weightfunction.ellipse(radius1, radius2, X)
 
# Plot von Ellipse
plt.contour(X[0], X[1], Z, 0)
plt.axis('equal')
#plt.show()
 
# Gitterweite
h_x = 2**(-level_x)
h_y = 2**(-level_y)
 
# Definiere Knotenfolge
  
# Uniform
# xi = np.arange(-(degree+1)/2, 1/h_x+(degree+1)/2+1, 1)*h_x
# yi = np.arange(-(degree+1)/2, 1/h_y+(degree+1)/2+1, 1)*h_y
 
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
print(len(index_Bspline_x))
index_Bspline_y = np.arange(-(degree-1)/2, len(yi)-3*(degree+1)/2+1, 1)
print(len(index_Bspline_y))
# print(index_Bspline_x)
# print(xi)
print("dimensionality:           {}".format(dim))
print("level:                    {}".format((level_x, level_y)))
print("number of Bsplines:       {}".format(len(index_Bspline_x)*len(index_Bspline_y)))
 
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
        if weightfunction.ellipse(radius1,radius2,[xi[int(i+degree)], yi[int(j+degree)]]) > 0:
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
            eval_supp[h,g] = weightfunction.ellipse(radius1, radius2, [grid_supp[0][g,h], grid_supp[1][g,h]])
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
  
  
a = np.meshgrid(xi,yi)
plt.scatter(a[0],a[1])
plt.scatter(J_all[:,0], J_all[:,1], c='crimson', s=50, lw=0)
plt.scatter(I_all[:,0], I_all[:,1], c='mediumblue', s=50, lw=0)
plt.scatter(J_relevant[:, 0], J_relevant[:, 1], c='goldenrod', s=50, lw=0)
plt.show()
  
  
  
# Definiere Gitter 
x = np.arange(0, 1+h_x, h_x)
y = np.arange(0, 1+h_y, h_y)
grid = np.meshgrid(x,y)
  
#print(xi)
#print(x)
 
 
####################### Test 2D Coeffs uniform auf [0,1] mit Identifizierung auf Bspline Mittelpunkte #######################
  
# Definiere Gitterpunkte als Vektor
k=0
gp = np.zeros((len(x)*len(y), dim))
for i in range(len(x)):
    for j in range(len(y)):
         gp[k] = [grid[0][j,i], grid[1][j,i]]
         k=k+1
#print(gp)
  
# # Monome definieren und an allen Knotenmittelpunkten auswerten
size_monomials = (degree+1)**2
n_neighbors = size_monomials
# eval_monomials = np.zeros((size_monomials, len(gp)))
# k = 0
# for j in range(degree + 1):
#     for i in range (degree + 1):
#         eval_monomials[k] = (pow(gp[:, 0], i) * pow(gp[:, 1], j))
#         k = k + 1   
# eval_monomials = np.transpose(eval_monomials)
# #print(eval_monomials)
#   
# # Aufstellen der Interpolationsmatrix A_ij = b_j(x_i)
# A = np.zeros((len(index_Bspline_x)*len(index_Bspline_y), len(gp)))
# for l in range(len(gp)):
#     k=0
#     for i in index_Bspline_x:
#         for j in index_Bspline_y:
#             A[l,k] = Bspline.evalBspline(degree, i, xi, gp[l,0]) * Bspline.evalBspline(degree, j, yi, gp[l,1])
#             k=k+1        
# print(A)
# #print(A.shape)
#  
#   
#   
# # Loese LGS und erhalte coeffs
# coeffs = np.linalg.solve(A, eval_monomials)
# #print(coeffs)
# #print(coeffs.shape)
#   
# # Test ob Loesen des LGS erfolgreich war
# error_LGS_coeffs = LA.norm(eval_monomials - np.matmul(A, coeffs))
# if error_LGS_coeffs > pow(10, -14):
#     print('LGS coeffs failed. error > 10e-14')
  
  
# Test ob Monome richtig interpoliert werden   
# # Beliebige Punkte im Gebiet
# anzahl = 20
# punkte = np.zeros((anzahl, 2))
# counter = 0 
# while counter < anzahl:
#     z = np.random.rand(1, 2)
#     if weightfunction.circle(radius, z[0]) > 0:
#         punkte[counter] = z[0]
#         counter = counter + 1
#      
# # Fehler zu Monomen bestimmen
# L2fehler = 0
# for k in range(len(punkte)):
#     summe = 0 
#     c = 0
#     for i in index_Bspline_x:
#         for j in index_Bspline_y:
#             summe = summe + coeffs[c,1] * Bspline.evalBspline(degree, i, xi, punkte[k,0]) * Bspline.evalBspline(degree, j, yi, punkte[k,1])
#             c=c+1
#     #fehler = summe - 1                         #coeffs[c,0], Monom 1                    
#     fehler = summe - punkte[k,0]               #coeffs[c,1], Monom x
#     #fehler = summe - punkte[k,0]**2
#     #fehler = summe - punkte[k,0]**3
#     #fehler = summe - punkte[k,0]*punkte[k,1]
#     #fehler = summe - punkte[k,1]               #coeffs[c,2], Monom y
#     #fehler = summe-(punkte[k,0]*punkte[k,1])    #coeffs[c,3], Monom x*y
#     L2fehler = L2fehler + fehler**2
# L2fehler = np.sqrt(L2fehler) 
# print(L2fehler)
# printLine()
  
  
# Nearest Neighbors bestimmen
k=1
if k == 0:
    # Nearest Neighbors nach Abstand
    distance = np.zeros((len(I_all), dim))
    NN = np.zeros((len(J_relevant), n_neighbors, dim))
    for j in range(len(J_relevant)):
        for i in range(len(I_all)):
            diff = I_all[i] - J_relevant[j]
            distance[i, 0] = LA.norm(diff)
            distance[i, 1] = i
            sort = distance[np.argsort(distance[:, 0])]
    # Loesche Punkte die Anzahl Nearest Neighbor ueberschreitet
        i = len(I_all) - 1
        while i >= n_neighbors:
            sort = np.delete(sort, i , 0)
            i = i - 1
    # Bestimme die Nearest Neighbor inneren Punkte
        for i in range(len(sort)):
            NN[j,i] = I_all[int(sort[i,1])]
                  
    # Index der nearest neighbor Punkte unter allen Punkten x
    index_NN = np.zeros((len(J_relevant), n_neighbors))
    for j in range(NN.shape[0]):
        for i in range(NN.shape[1]):
            for k in range(len(gp)):
                if NN[j, i, 0] == gp[k,0] and NN[j, i, 1] == gp[k,1]:
                    index_NN[j, i] = k
       
    # Nearest Neighbors sortieren nach Index im Gesamtgitter
    index_NN=np.sort(index_NN,axis=1)
       
                   
elif k == 1:
    # Nearest Neighbors mit naehestem (n+1)x(n+1) Block
    distance = np.zeros((len(I_all), dim))
    NN = np.zeros((len(J_relevant), n_neighbors, dim))
    for j in range(len(J_relevant)):
        for i in range(len(I_all)):
            diff = I_all[i] - J_relevant[j]
            distance[i, 0] = LA.norm(diff)
            distance[i, 1] = i
            sort = distance[np.argsort(distance[:, 0])]
        NNsearch(sort, j, 0)
    NN=NN
           
    index_NN = np.zeros((len(J_relevant), n_neighbors))
    for j in range(NN.shape[0]):
        for i in range(NN.shape[1]):
            for k in range(len(gp)):
                if NN[j, i, 0] == gp[k,0] and NN[j, i, 1] == gp[k,1]:
                    index_NN[j, i] = k
        
    # Nearest Neighbors sortieren nach Index im Gesamtgitter
    #index_NN=np.sort(index_NN,axis=1)
#print(index_NN)
   
# Plot der nearest neighbors 
for i in range(len(J_relevant)):
    plt.scatter(I_all[:, 0], I_all[:, 1], c='mediumblue', s=50, lw=0)
    plt.scatter(J_relevant[:, 0], J_relevant[:, 1], c='goldenrod', s=50, lw=0)
    plt.scatter(J_relevant[i, 0], J_relevant[i, 1], c='cyan', s=50, lw=0) 
    plt.scatter(NN[i,:,0], NN[i, :, 1], c='limegreen', s=50, lw=0)
    plt.contour(X[0], X[1], Z, 0)
    plt.axis('equal')
    plt.show()
 





# # with open('/home/hanausmc/pickle/circle/punkte.pkl', 'rb') as fp:
# #     punkte_loaded = pickle.load(fp)
#     
# #print(punkte_loaded['punkte'].shape)
# 
# 
# dim = 2         # Dimension
# radius = 0.4    # Radius von Kreis
# degree = 5     # Grad von B-Splines (nur ungerade)
# #level_x = 2   # Level in x Richtung    
# #level_y = 2     # Level in y Richtung
# 
# 
# 
# 
#  
#  
# levelymax = 7
# for levelx in range(6,9):
#     for levely  in range(4,levelymax):
#         start = time.time()
#         print(levelx,levely)
#         level_x = levelx
#         level_y = levely
#           
#   
#       
#    
#         # Pruefe ob Level hoch genug
#         if level_x and level_y < np.log2(degree+1):
#             print('Error: Level zu niedrig. Es muss Level >= log2(degree+1) sein ')
#             quit()
#             
#             
#             
#         # Gitterweite
#         h_x = 2**(-level_x)
#         h_y = 2**(-level_y)
#             
#         # Definiere Knotenfolge
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
#             
#         # Index von Bspline auf Knotenfolge
#         index_Bspline_x = np.arange(-(degree-1)/2, len(xi)-3*(degree+1)/2+1, 1)
#         index_Bspline_y = np.arange(-(degree-1)/2, len(yi)-3*(degree+1)/2+1, 1)
#         # print(index_Bspline_x)
#         # print(xi)
#             
#         k=0
#         index_all_Bsplines = np.zeros((len(index_Bspline_x)*len(index_Bspline_y),dim))
#         for i in index_Bspline_x:
#             for j in index_Bspline_y:
#                 index_all_Bsplines[k] = [i,j]
#                 k=k+1
#         #print(index_all_Bsplines)
#             
#         # Index (x,y) der Bsplines mit Knotenmittelpunkt im inneren des Gebiets
#         index_inner_Bsplines = np.zeros(dim)
#         index_outer_Bsplines = np.zeros(dim)
#             
#         for i in index_Bspline_x:
#             for j in index_Bspline_y:
#                 if weightfunction.circle(radius,[xi[int(i+degree)], yi[int(j+degree)]]) > 0:
#                     index_inner_Bsplines = np.vstack((index_inner_Bsplines, [i,j]))
#                 else:
#                     index_outer_Bsplines = np.vstack((index_outer_Bsplines, [i,j]))
#         index_inner_Bsplines = np.delete(index_inner_Bsplines, 0, 0)
#         index_outer_Bsplines = np.delete(index_outer_Bsplines, 0, 0)
#         # print(index_inner_Bsplines)
#         # printLine()
#         #print(index_outer_Bsplines)
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
#         #print(index_inner_Bsplines)
#             
#             
#             
#             
#             
#         print("dimensionality:           {}".format(dim))
#         print("level:                    {}".format((level_x, level_y)))
#         print("number of Bsplines:       {}".format(len(index_Bspline_x)*len(index_Bspline_y)))
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
#                       
#                     eval_supp[h,g] = weightfunction.circle(radius, [grid_supp[0][g,h], grid_supp[1][g,h]])
#             if (eval_supp > 0).any():
#                 index_outer_relevant_Bsplines = np.vstack((index_outer_relevant_Bsplines, [index_outer_Bsplines[j]]))
#         index_outer_relevant_Bsplines = np.delete(index_outer_relevant_Bsplines,0,0)
#         #print(index_outer_relevant_Bsplines)
#             
#         J_relevant = np.zeros((len(index_outer_relevant_Bsplines), dim))
#         k=0
#         for j in index_outer_relevant_Bsplines:
#             J_relevant[k] = [xi[int(j[0]+degree)], yi[int(j[1]+degree)]]
#             k=k+1
#         #print(J_relevant)
#             
#             
#         # Index der inneren Bsplines unter Gesamtanzahl (n+1)**2
#         index_I_all = np.zeros(len(I_all))
#         for i in range(len(I_all)):
#             for j in range(len(midpoints)):
#                 if I_all[i, 0] == midpoints[j, 0] and I_all[i, 1] == midpoints[j, 1]:
#                     index_I_all[i] = j
#         #print(index_I_all)
#             
#         # Index der aeusseren Bsplines unter Gesamtanzahl (n+1)**2
#         index_J_all = np.zeros(len(J_all))
#         for i in range(len(J_all)):
#             for j in range(len(midpoints)):
#                 if J_all[i, 0] == midpoints[j, 0] and J_all[i, 1] == midpoints[j, 1]:
#                     index_J_all[i] = j
#         #print(index_J_all)
#             
#         # Index der relevanten aeusseren Bsplines unter Gesamtanzahl (n+1)**2
#         index_J_relevant = np.zeros(len(J_relevant))
#         for i in range(len(J_relevant)):
#             for j in range(len(midpoints)):
#                 if J_relevant[i, 0] == midpoints[j, 0] and J_relevant[i, 1] == midpoints[j, 1]:
#                     index_J_relevant[i] = j
#         #print(index_J_relevant)
#             
#             
#         # Definiere Gitter 
#         x = np.arange(0, 1+h_x, h_x)
#         y = np.arange(0, 1+h_y, h_y)
#         grid = np.meshgrid(x,y)
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
#         # Aufstellen der Interpolationsmatrix A_ij = b_j(x_i)
#         A = np.zeros((len(index_Bspline_x)*len(index_Bspline_y), len(gp)))
#         for l in range(len(gp)):
#             k=0
#             for i in index_Bspline_x:
#                 for j in index_Bspline_y:
#                     A[l,k] = Bspline.evalBspline(degree, i, xi, gp[l,0]) * Bspline.evalBspline(degree, j, yi, gp[l,1])
#                     k=k+1        
#         #print(A)
#         print(A.shape)
#             
#         data['A'] = A
#         filename = 'data_{}_{}_circle_degree_{}.pkl'.format(level_x, level_y, degree)
#         path = '/home/hanausmc/pickle/circle'
#         filepath = os.path.join(path, filename)
#         with open(filepath, 'wb') as fp:
#             pickle.dump(data, fp)
#         stop = time.time()
#         duration = stop-start
#         print('duration: {}'.format(duration))
#     levelymax=levelymax-1
    
    
    
    











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
