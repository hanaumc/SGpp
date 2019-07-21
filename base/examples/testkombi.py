import pysgpp
import math
import sys
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import weightfunction
import Bspline
import scipy 
from scipy import special
from mpl_toolkits import mplot3d
from numpy import linalg as LA, transpose, full, vstack
from matplotlib.pyplot import axis
from _socket import AI_ALL


def printLine():
    print("--------------------------------------------------------------------------------------")

def NNsearch(sort, j, q):
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
                    print('Fehler: Gitter nicht fein genug. Erhoehe Level.')
                    quit()
                else:
                    NNsearch(sort, j, q+1)
    return NN

dim = 2         # Dimension
radius = 0.4    # Radius von Kreis
degree = 3      # Grad von B-Splines (nur ungerade)
level_x = 3    # Level in x Richtung    
level_y = 3     # Level in y Richtung

# Pruefe ob Level hoch genug
if level_x and level_y < np.log2(degree+1):
    print('Error: Level zu niedrig. Es muss Level >= log2(degree+1) sein ')
    quit()

# Gitter fuer Kreis erzeugen und auswerten
x0 = np.linspace(0, 1, 50)
X = np.meshgrid(x0, x0) 
Z = weightfunction.circle(radius, X)

# Plot von Kreis
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
index_Bspline_y = np.arange(-(degree-1)/2, len(yi)-3*(degree+1)/2+1, 1)
# print(index_Bspline_x)
# print(xi)

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
print(index_outer_Bsplines)

# Pruefe ob genug innere Bsplines vorhanden sind 
if len(index_inner_Bsplines) < (degree+1)**2:
    print('Nicht genug innere Punkte. Erhoehe Level oder Gebiet.')   
    quit() 

# Definiere Gitterpunkte als Vektor
k=0
gp = np.zeros((len(index_Bspline_x)*len(index_Bspline_y), dim))
for i in index_Bspline_x:
    for j in index_Bspline_y:
        gp[k] = [xi[int(i+degree)], yi[int(j+degree)]]
        k=k+1
#print(gp)

# Unterteilung in innere und aeussere Bsplines durch Mittelpunkte der Bsplines
k=0
I_all = np.zeros((len(index_inner_Bsplines), dim))
J_all = np.zeros((len(index_outer_Bsplines), dim))
for i in index_inner_Bsplines:
    I_all[k] = [xi[int(i[0]+degree)], yi[int(i[1]+degree)]]
    k=k+1
k=0
for j in index_outer_Bsplines:
    J_all[k] = [xi[int(j[0]+degree)], yi[int(j[1]+degree)]]
    k=k+1
#print(I_all)
#print(J_all)


a = np.meshgrid(xi,yi)
plt.scatter(a[0],a[1])
#plt.show()


 
# # Definiere Gitter 
# x = np.arange(0, 1+h_x, h_x)
# y = np.arange(0, 1+h_y, h_y)
# grid = np.meshgrid(x,y)


print("dimensionality:           {}".format(dim))
print("level:                    {}".format((level_x, level_y)))
print("number of Bsplines:       {}".format(len(index_Bspline_x)*len(index_Bspline_y)))



# Plot von inneren und aeusseren Punkten
plt.scatter(J_all[:,0], J_all[:,1], c='crimson', s=50, lw=0)
plt.scatter(I_all[:,0], I_all[:,1], c='mediumblue', s=50, lw=0)
#plt.show()


for i in range(0,5):
    print(xi[int(index_outer_Bsplines[2,1]+(i+(degree-1)/2))])


# for i in index_outer_Bsplines:
#     for j in index_outer_Bsplines:
#         supp_x = (xi[i[0]+(degree-1)/2])
#        # supp_y = 


# Unterteilung der aeusseren Bsplines in relevante und irrelevante Bsplines
# index_outer_relevant_Bsplines = np.zeros((dim))
# J_relevant = np.zeros((dim))
# for j in range(len(J_all)):    
#     
#     supp_x = np.arange(J_all[j,0]-int((degree+1)/2)*h_x, J_all[j,0]+int((degree+1)/2+1)*h_x, h_x)
#     supp_y = np.arange(J_all[j,1]-int((degree+1)/2)*h_y, J_all[j,1]+int((degree+1)/2+1)*h_y, h_y)
#     print(supp_x)
#     grid_supp_points = np.meshgrid(supp_x, supp_y)
#     eval_supp_points = np.zeros((len(supp_x), len(supp_y)))
#     for i in range(len(supp_x)):
#         for k in range(len(supp_y)):
#             eval_supp_points[i,k] = weightfunction.circle(radius, [grid_supp_points[0][k,i], grid_supp_points[1][k,i]])
#     if (eval_supp_points > 0).any():
#         J_relevant = np.vstack((J_relevant, J_all[j]))      
# J_relevant = np.delete(J_relevant, 0, 0)
# print(J_relevant)
# 
# for i in range(len(J_relevant)):
#     print(i)



# # Unterteilung der aeusseren Punkte in relevante und irrelevante Punkte
# J_relevant = np.zeros(dim)
# for j in range(len(J_all)):
#     supp_x = np.arange(J_all[j,0]-int((degree+1)/2)*h_x, J_all[j,0]+int((degree+1)/2+1)*h_x, h_x)
#     supp_y = np.arange(J_all[j,1]-int((degree+1)/2)*h_y, J_all[j,1]+int((degree+1)/2+1)*h_y, h_y)
#     grid_supp_points = np.meshgrid(supp_x, supp_y)
#     eval_supp_points = np.zeros((len(supp_x), len(supp_y)))
#     for i in range(len(supp_x)):
#         for k in range(len(supp_y)):
#             eval_supp_points[i,k] = weightfunction.circle(radius, [grid_supp_points[0][k,i], grid_supp_points[1][k,i]])
#     if (eval_supp_points > 0).any():
#         J_relevant = np.vstack((J_relevant, J_all[j]))      
# J_relevant = np.delete(J_relevant, 0, 0)
# #print(J_relevant)
# 
# # Index der auesseren relevanten Punkte unter allen Punkten
# index_J_relevant = np.zeros(len(J_relevant))
# for i in range(len(J_relevant)):
#     for j in range(len(gp)):
#         if J_relevant[i, 0] == gp[j, 0] and J_relevant[i, 1] == gp[j, 1]:
#             index_J_relevant[i] = j
# #print(index_J_relevant)
# 
# 
# # Plot von inneren, aeusseren und relevanten aeusseren Punkten
# plt.scatter(grid[0], grid[1], c='crimson', s=50, lw=0)
# plt.scatter(I_all[:,0], I_all[:,1], c='mediumblue', s=50, lw=0)
# plt.scatter(J_relevant[:, 0], J_relevant[:, 1], c='goldenrod', s=50, lw=0)
# #plt.show()
# 
# # Monome definieren und an allen Gitterpunkten auswerten
# size_monomials = (degree+1)**2
# n_neighbors = size_monomials
# eval_monomials = np.zeros((size_monomials, len(gp)))
# k = 0
# for j in range(degree + 1):
#     for i in range (degree + 1):
#         eval_monomials[k] = (pow(gp[:, 0], i) * pow(gp[:, 1], j))
#         k = k + 1   
# eval_monomials = np.transpose(eval_monomials)
   
# Aufstellen der Interpolationsmatrix A_ij = b_j(x_i)
# A = np.zeros((len(index_Bspline_x)*len(index_Bspline_y), len(gp)))
# for l in range(len(gp)):
#     k=0
#     for i in range(len(index_Bspline_x)):
#         for j in range(len(index_Bspline_y)):
#             A[l,k] = Bspline.evalBspline(degree, i, xi, gp[l,0]) * Bspline.evalBspline(degree, j, yi, gp[l,1])
#             k=k+1        
# print(A)
#      
# # Loese LGS und erhalte coeffs
# coeffs = np.linalg.solve(A, eval_monomials)
# #print(coeffs)
#  
# # Test ob Loesen des LGS erfolgreich war
# error = eval_monomials - np.matmul(A, coeffs)
# error = LA.norm(error)
# if error > pow(10, -14):
#     print('failed. error > 10e-14')
# 
# 
# k=1
# if k == 0:
#     # Nearest Neighbors nach Abstand
#     distance = np.zeros((len(I_all), dim))
#     NN = np.zeros((len(J_relevant), n_neighbors, dim))
#     for j in range(len(J_relevant)):
#         for i in range(len(I_all)):
#             diff = I_all[i] - J_relevant[j]
#             distance[i, 0] = LA.norm(diff)
#             distance[i, 1] = i
#             sort = distance[np.argsort(distance[:, 0])]
#     # Loesche Punkte die Anzahl Nearest Neighbor ueberschreitet
#         i = len(I_all) - 1
#         while i >= n_neighbors:
#             sort = np.delete(sort, i , 0)
#             i = i - 1
#     # Bestimme die Nearest Neighbor inneren Punkte
#         for i in range(len(sort)):
#             NN[j,i] = I_all[int(sort[i,1])]
#                
#     # Index der nearest neighbor Punkte unter allen Punkten x
#     index_NN = np.zeros((len(J_relevant), n_neighbors))
#     for j in range(NN.shape[0]):
#         for i in range(NN.shape[1]):
#             for k in range(len(gp)):
#                 if NN[j, i, 0] == gp[k,0] and NN[j, i, 1] == gp[k,1]:
#                     index_NN[j, i] = k
#     
#     # Nearest Neighbors sortieren nach Index im Gesamtgitter
#     index_NN=np.sort(index_NN,axis=1)
#     
#     
#     # Plot der nearest neighbors 
#     for i in range(len(J_relevant)):
#         plt.scatter(I_all[:, 0], I_all[:, 1], c='mediumblue', s=50, lw=0)
#         plt.scatter(J_relevant[:, 0], J_relevant[:, 1], c='goldenrod', s=50, lw=0)
#         plt.scatter(J_relevant[i, 0], J_relevant[i, 1], c='cyan', s=50, lw=0) 
#         plt.scatter(NN[i,:,0], NN[i, :, 1], c='limegreen', s=50, lw=0)
#         plt.contour(X[0], X[1], Z, 0)
#         plt.axis('equal')
#         #plt.show()
#                 
# elif k == 1:
#     # Nearest Neighbors mit naehestem (n+1)x(n+1) Block
#     distance = np.zeros((len(I_all), dim))
#     NN = np.zeros((len(J_relevant), n_neighbors, dim))
#     for j in range(len(J_relevant)):
#         for i in range(len(I_all)):
#             diff = I_all[i] - J_relevant[j]
#             distance[i, 0] = LA.norm(diff)
#             distance[i, 1] = i
#             sort = distance[np.argsort(distance[:, 0])]
#         NNsearch(sort, j, 0)
#     NN=NN
#         
#     index_NN = np.zeros((len(J_relevant), n_neighbors))
#     for j in range(NN.shape[0]):
#         for i in range(NN.shape[1]):
#             for k in range(len(gp)):
#                 if NN[j, i, 0] == gp[k,0] and NN[j, i, 1] == gp[k,1]:
#                     index_NN[j, i] = k
#      
#     # Nearest Neighbors sortieren nach Index im Gesamtgitter
#     #index_NN=np.sort(index_NN,axis=1)
#     #print(index_NN)
# 
# 
# # Plot der nearest neighbors 
# for i in range(len(J_relevant)):
#     plt.scatter(I_all[:, 0], I_all[:, 1], c='mediumblue', s=50, lw=0)
#     plt.scatter(J_relevant[:, 0], J_relevant[:, 1], c='goldenrod', s=50, lw=0)
#     plt.scatter(J_relevant[i, 0], J_relevant[i, 1], c='cyan', s=50, lw=0) 
#     plt.scatter(NN[i,:,0], NN[i, :, 1], c='limegreen', s=50, lw=0)
#     plt.contour(X[0], X[1], Z, 0)
#     plt.axis('equal')
#     #plt.show()        
# 
# 
#                        
# # Definiere Koeffizientenmatrix der aeusseren relevanten Punkte
# coeffs_J_relevant = np.zeros((len(J_relevant),1, size_monomials))
# for i in range(len(J_relevant)):
#     coeffs_J_relevant[i] = coeffs[int(index_J_relevant[i])]
# coeffs_J_relevant = np.transpose(coeffs_J_relevant, [0,2,1])
# #print(coeffs_J_relevant)
#          
# 
# # # Definiere Koeffizientenmatrix der nearest neighbors
# coeffs_NN = np.zeros((len(J_relevant),n_neighbors, size_monomials))
# for j in range(len(J_relevant)):
#     for i in range(n_neighbors):
#         coeffs_NN[j,i] = coeffs[int(index_NN[j, i])]
# coeffs_NN = np.transpose(coeffs_NN, [0,2,1])
# #print(coeffs_NN)     
# 
# # det = np.linalg.det(coeffs_NN)
# # print(det) 
# 
# extension_coeffs = np.zeros((len(J_relevant), size_monomials, 1))
# 
# for i in range(coeffs_NN.shape[0]):
#     extension_coeffs[i] = np.linalg.solve(coeffs_NN[i], coeffs_J_relevant[i])
# #print(extension_coeffs.shape)
# 
#        
# # Definiere J(i) 
# # J_i = np.zeros((index_NN.shape[1], len(I_all)))  # x
# # for i in range(len(I_all)):#index_x
# #     for j in range((index_NN.shape[1])):
# #         for k in range((index_NN.shape[0])):
# #             if index_gp[i] == index_NN[k, j]:
# #                 # print(i,index_J_relevant[j])
# #                 J_i[j, i] = index_J_relevant[j]
# # print(J_i)       
# 
# # Index der aeusseren Punkte unter den relevanten auesseren Punkten
# for k in range(len(I_all)):
#     J_i = np.zeros(1)
#     for i in range(NN.shape[0]):
#         for j in range(NN.shape[1]):
#             if NN[i,j,0] == I_all[k,0] and NN[i,j,1] == I_all[k,1]:
#                 J_i = np.append(J_i,i)
#     J_i = np.delete(J_i,0)
#     #print(J_i)
#     
# 
# 
# 
# 
#     
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
# 
# 
# 
# 
# 
# 
# # # Beliebige Punkte im Gebiet
# # anzahl = 20
# # punkte = np.zeros((anzahl, 2))
# # counter = 0 
# # while counter < anzahl:
# #     z = np.random.rand(1, 2)
# #     if weightfunction.circle(radius, z[0]) > 0:
# #         punkte[counter] = z[0]
# #         counter = counter + 1
# #   
# # # Fehler zu Monomen bestimmen
# # L2fehler = 0
# # for k in range(len(punkte)):
# #     summe = 0 
# #     c = 0
# #     for i in range(len(index_Bspline_x)):
# #         for j in range(len(index_Bspline_y)):
# #             summe = summe + coeffs[c,1] * Bspline.evalBspline(degree, i, xi, punkte[k,0]) * Bspline.evalBspline(degree, j, yi, punkte[k,1])
# #             c=c+1
# #     #fehler = summe - 1                         #coeffs[c,0], Monom 1                    
# #     fehler = summe - punkte[k,0]               #coeffs[c,1], Monom x
# #     #fehler = summe - punkte[k,0]**2
# #     #fehler = summe - punkte[k,0]**3
# #     #fehler = summe - punkte[k,0]*punkte[k,1]
# #     #fehler = summe - punkte[k,1]               #coeffs[c,2], Monom y
# #     #fehler = summe-(punkte[k,0]*punkte[k,1])    #coeffs[c,3], Monom x*y
# #     L2fehler = L2fehler + fehler**2
# # L2fehler = np.sqrt(L2fehler) 
# # print(L2fehler)
# # printLine()
#   
  