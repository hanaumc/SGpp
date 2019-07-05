import pysgpp
import math
import sys
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import weightfunction
import grid
import Bspline
import scipy 
from scipy import special
from mpl_toolkits import mplot3d
from numpy import linalg as LA, transpose, full
from matplotlib.pyplot import axis


def printLine():
    print("--------------------------------------------------------------------------------------")

# a = sum( i for i in range(4))
# print(a)

dim = 2         # Dimension
radius = 0.2    # Radius von Kreis
degree = 3      # Grad von B-Splines (nur ungerade)
level_x = 5     # Level in x Richtung    
level_y = 5     # Level in y Richtung

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
xi = np.arange(-(degree+1)/2, 1/h_x+(degree+1)/2+1, 1)*h_x
yi = np.arange(-(degree+1)/2, 1/h_y+(degree+1)/2+1, 1)*h_y

# Index von Bspline auf Knotenfolge
index_Bspline_x = np.arange(0, 1/h_x+1, 1)
index_Bspline_y = np.arange(0, 1/h_y+1, 1)

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

print("dimensionality:           {}".format(dim))
print("number of grid points:    {}".format(len(x)*len(y)))

# Auswerten von Gewichtsfunktion an Gitterpunkten und Unterteilung in innere und aeussere Punkte
I_all = np.zeros((dim))
J_all = np.zeros((dim))
eval_grid = np.zeros((len(x),len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        eval_grid[i,j] = weightfunction.circle(radius, [grid[0][j,i], grid[1][j,i]])
        if eval_grid[i,j] > 0:
            I_all = np.vstack((I_all, [grid[0][j,i], grid[1][j,i]]))
        else:
            J_all = np.vstack((J_all, [grid[0][j,i], grid[1][j,i]]))
I_all = np.delete(I_all, 0, 0)
J_all = np.delete(J_all, 0, 0)            

# Plot von inneren und aeusseren Punkten 
plt.scatter(grid[0], grid[1], c='crimson', s=50, lw=0)
plt.scatter(I_all[:,0], I_all[:,1], c='mediumblue', s=50, lw=0)
#plt.show()

# Unterteilung der aeusseren Punkte in relevante und irrelevante Punkte
J_relevant = np.zeros(dim)
for j in range(len(J_all)):
    supp_x = np.arange(J_all[j,0]-int((degree+1)/2)*h_x, J_all[j,0]+int((degree+1)/2+1)*h_x, h_x)
    supp_y = np.arange(J_all[j,1]-int((degree+1)/2)*h_y, J_all[j,1]+int((degree+1)/2+1)*h_y, h_y)
    grid_supp_points = np.meshgrid(supp_x, supp_y)
    eval_supp_points = np.zeros((len(supp_x), len(supp_y)))
    for i in range(len(supp_x)):
        for k in range(len(supp_y)):
            eval_supp_points[i,k] = weightfunction.circle(radius, [grid_supp_points[0][k,i], grid_supp_points[1][k,i]])
    if (eval_supp_points > 0).any():
        J_relevant = np.vstack((J_relevant, J_all[j]))      
J_relevant = np.delete(J_relevant, 0, 0)
#print(J_relevant)

# Plot von inneren, aeusseren und relevanten aeusseren Punkten
plt.scatter(grid[0], grid[1], c='crimson', s=50, lw=0)
plt.scatter(I_all[:,0], I_all[:,1], c='mediumblue', s=50, lw=0)
plt.scatter(J_relevant[:, 0], J_relevant[:, 1], c='goldenrod', s=50, lw=0)
#plt.show()

# Monome definieren und an allen Gitterpunkten auswerten
size_monomials = (degree+1)**2
eval_monomials = np.zeros((size_monomials, len(gp)))
k = 0
for j in range(degree + 1):
    for i in range (degree + 1):
        eval_monomials[k] = (pow(gp[:, 0], i) * pow(gp[:, 1], j))
        k = k + 1   
eval_monomials = np.transpose(eval_monomials)
#print(eval_monomials)

# eval_monomials = pow(gp[:,0], 1)*pow(gp[:,1],0)
# 
# eval_monomials = np.transpose(eval_monomials)
#print(eval_monomials)

# Aufstellen der Interpolationsmatrix A_ij = b_j(x_i)
A = np.zeros((len(index_Bspline_x)*len(index_Bspline_y), len(gp)))
#print(A.shape)
for l in range(len(gp)):
    k=0
    for i in range(len(index_Bspline_x)):
        for j in range(len(index_Bspline_y)):
            A[l,k] = Bspline.evalBspline(degree, i, xi, gp[l,0]) * Bspline.evalBspline(degree, j, yi, gp[l,1])
            k=k+1        
#print(A)
    
# Loese LGS und erhalte coeffs
coeffs = np.linalg.solve(A, eval_monomials)
#print(coeffs) 

# Test ob Loesen des LGS erfolgreich war
error = eval_monomials - np.matmul(A, coeffs)
error = LA.norm(error)
#print(error)
if error > pow(10, -14):
    print('failed. error > 10e-14')
printLine()


# Beliebige Punkte im Gebiet
anzahl = 20
punkte = np.zeros((anzahl, 2))
counter = 0 
while counter < anzahl:
    z = np.random.rand(1, 2)
    if weightfunction.circle(radius, z[0]) > 0:
        punkte[counter] = z[0]
        counter = counter + 1

punkt = [0.45,0.6]
# for i in range(len(index_Bspline_x)):
#     print(Bspline.evalBspline(degree, i, xi, punkt[0]))
# printLine()
# for i in range(len(index_Bspline_y)):
#     print(Bspline.evalBspline(degree, i, yi, punkt[1]))
# printLine()
# for i in range(len(index_Bspline_x)):
#     for j in range(len(index_Bspline_y)):
#         print(Bspline.evalBspline(degree, i, xi, punkt[0])*Bspline.evalBspline(degree, j, yi, punkt[1]))

L2fehler = 0
for k in range(len(punkte)):
    summe = 0 
    c = 0
    for i in range(len(index_Bspline_x)):
        for j in range(len(index_Bspline_y)):
            summe = summe + coeffs[c,0] * Bspline.evalBspline(degree, i, xi, punkte[k,0]) * Bspline.evalBspline(degree, j, yi, punkte[k,1])
            c=c+1
    fehler = summe - 1                          #coeffs[i,0], Monom 1                    
    #fehler = summe - punkte[j,0]               #coeffs[i,1], Monom x
    #fehler = summe - punkte[j,1]               #coeffs[i,2], Monom y
    #fehler = summe-(punkte[j,0]*punkte[j,1])   #coeffs[i,3], Monom x*y
#     print(fehler)
#     printLine()
    L2fehler = L2fehler + fehler**2
L2fehler = np.sqrt(L2fehler) 
print(L2fehler)

        
        
        
# for j in range(len(punkte)):
#     summe = 0
# 
#     for l in range(len(index_Bspline_x)):
#         for m in range(len(index_Bspline_y)):
#             summe = summe + coeffs[i] * Bspline.evalBspline(degree, l, xi, punkte[j,0]) * Bspline.evalBspline(degree, m, yi, punkte[j,1])
#     #summe = summe + coeffs[i,1]*basis.eval(int(level_x), int(ind[i,0]), punkte[j,0])*basis.eval(int(level_y), int(ind[i,1]), punkte[j,1])
# #print(punkte[j])
# print(summe)
# #fehler = summe - 1                         #coeffs[i,0], Monom 1
# fehler = summe - punkte[j,0]                #coeffs[i,1], Monom x
# #    fehler = summe - punkte[j,1]               #coeffs[i,2], Monom y
# #    fehler = summe-(punkte[j,0]*punkte[j,1])   #coeffs[i,3], Monom x*y
# print(fehler)
# printLine()
  
  
# # Beliebige Punkte im Einheitsquadrat
# punkte = np.random.rand(20,2)
# for j in range(punkte.shape[0]):
#     summe = 0
#     for i in range(len(gp)):
#         for l in range(len(index_Bspline_x)):
#             for m in range(len(index_Bspline_y)):
#                 summe = summe + coeffs[i,0] * Bspline.evalBspline(degree, l, xi, punkte[j,0]) * Bspline.evalBspline(degree, m, yi, punkte[j,1])
#     print(punkte[j])
#     print(summe)
#     fehler = summe - 1                         #coeffs[i,0], Monom 1
# #    fehler = summe - punkte[j,0]                #coeffs[i,1], Monom x
# #    fehler = summe - punkte[j,1]               #coeffs[i,2], Monom y
# #    fehler = summe-(punkte[j,0]*punkte[j,1])   #coeffs[i,3], Monom x*y
#     print(fehler)
#     printLine()
# 
#         
# # Test ob Loesen des LGS erfolgreich war
# error = eval_monomials - np.matmul(A, coeffs)
# error = LA.norm(error)
# print(error)
# if error > pow(10, -14):
#     print('failed. error > 10e-14')
  
  

# 
# 
# # Auswerten von Gewichtsfunktion des Kreises an Gitterpunkten x
# index_x = np.zeros(len(x))
# eval_circle = np.zeros(len(x))
# for i in range(len(x)):
#     index_x[i] = i   
#     eval_circle[i] = weightfunction.circle(radius, x[i])
# 
# # # Index der inneren Punkte unter Gesamtpunkten x
# # index_I_all = np.zeros(len(I_all))
# # for i in range(len(I_all)):
# #     for j in range(len(x)):
# #         if I_all[i, 0] == x[j, 0] and I_all[i, 1] == x[j, 1]:
# #             index_I_all[i] = j
# # 
# # # Index der aeusseren Punkte unter Gesamtpunkten x
# # index_J_all = np.zeros(len(J_all))
# # for i in range(len(J_all)):
# #     for j in range(len(x)):
# #         if J_all[i, 0] == x[j, 0] and J_all[i, 1] == x[j, 1]:
# #             index_J_all[i] = j
# # 
# # 
# # # Bestimme Gitterweite (h_x,h_y) in Abhaengigkeit vom Level
# # 
# # h = np.zeros((len(x),dim))
# # for i in range(len(h)):
# #     h[i] = 2**(-lvl[i])
# # 
# # 
# # 
# # # Index der relevanten aeusseren Punkte unter Gesamtpunkten x
# # index_J_relevant = np.zeros(len(J_relevant))
# # for i in range(len(J_relevant)):
# #     for j in range(len(x)):
# #         if J_relevant[i, 0] == x[j, 0] and J_relevant[i, 1] == x[j, 1]:
# #             index_J_relevant[i] = j
# #      
# #           
# 
# 
# # Anzahl nearest neighbors
# n_neighbors = (degree+1)**2
# 
# #print(x)
# # Monome definieren und an allen Gitterpunkten auswerten
# size_monomials = n_neighbors
# eval_monomials = np.zeros((size_monomials, len(x)))
# #print(eval_monomials.shape)
# k = 0
# for j in range(degree + 1):
#     for i in range (degree + 1):
#         eval_monomials[k] = (pow(x[:, 0], i) * pow(x[:, 1], j))
#         k = k + 1   
# eval_monomials = np.transpose(eval_monomials)
# #print(eval_monomials)
#  
# # Definiere Matrix A fuer Interpolation der coeffs
# A = np.zeros((len(x), len(x)))
# for i in range(len(x)):
#     for j in range(len(x)):
#         A[i, j] = basis.eval(int(level_x), int(ind[j, 0]), x[i, 0]) * basis.eval(int(level_y), int(ind[j, 1]), x[i, 1])   
# #print(A.shape)
# #  
# # # x_str = np.array_repr(A).replace('\n', '')
# # # print(x_str)
# #  
# # Loese LGS und erhalte coeffs
# coeffs = np.linalg.solve(A, eval_monomials)
# #print(coeffs)
# #  
# # Beliebige Punkte im Gebiet
# 
# anzahl = 100
# punkte = np.zeros((anzahl, 2))
# counter = 0 
# while counter < anzahl:
#     z = np.random.rand(1, 2)
#     if weightfunction.circle(radius, z[0]) > 0:
#         punkte[counter] = z[0]
#         counter = counter + 1
# for j in range(punkte.shape[0]):
#     summe = 0
#     for i in range(len(x)):
#         summe = summe + coeffs[i,1]*basis.eval(int(level_x), int(ind[i,0]), punkte[j,0])*basis.eval(int(level_y), int(ind[i,1]), punkte[j,1])
#     #print(punkte[j])
#     #print(summe)
#     #fehler = summe - 1                         #coeffs[i,0], Monom 1
#     fehler = summe - punkte[j,0]                #coeffs[i,1], Monom x
#     #    fehler = summe - punkte[j,1]               #coeffs[i,2], Monom y
#     #    fehler = summe-(punkte[j,0]*punkte[j,1])   #coeffs[i,3], Monom x*y
# #     print(fehler)
# #     printLine()
# #  
# #  
# # # # Beliebige Punkte im Einheitsquadrat
# # # punkte = np.random.rand(20,2)
# # # for j in range(punkte.shape[0]):
# # #     summe = 0
# # #     for i in range(len(x)):
# # #         summe = summe + coeffs[i,0]*basis.eval(int(lvl[i,0]), int(ind[i,0]), punkte[j,0])*basis.eval(int(lvl[i,1]), int(ind[i,1]), punkte[j,1])
# # #     print(punkte[j])
# # #     print(summe)
# # #     fehler = summe - 1                         #coeffs[i,0], Monom 1
# # # #    fehler = summe - punkte[j,0]                #coeffs[i,1], Monom x
# # # #    fehler = summe - punkte[j,1]               #coeffs[i,2], Monom y
# # # #    fehler = summe-(punkte[j,0]*punkte[j,1])   #coeffs[i,3], Monom x*y
# # #     print(fehler)
# # #     printLine()
# #      
# #          
# # # Test ob Loesen des LGS erfolgreich war
# # error = eval_monomials - np.matmul(A, coeffs)
# # error = LA.norm(error)
# # print(error)
# # if error > pow(10, -14):
# #     print('failed. error > 10e-14')
# #  


