import pysgpp
import math
import sys
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import weightfunction

import scipy 
from scipy import special
from mpl_toolkits import mplot3d
from numpy import linalg as LA, transpose, full
from matplotlib.pyplot import axis


def printLine():
    print("--------------------------------------------------------------------------------------")






dim =  2  # Dimension
radius = 0.3  # Radius von Kreis
degree = 1  # Grad von B-Splines (nur ungerade)
level = 2       # Level von Sparse Grid

# Gitter fuer Kreis erzeugen und auswerten
x0 = np.linspace(0, 1, 50)
if dim == 2:
    X = np.meshgrid(x0, x0)
elif dim == 3:
    X = np.meshgrid(x0, x0, x0)      
Z = weightfunction.circle(radius, X)

# Plot von Kreis
plt.contour(X[0], X[1], Z, 0)
plt.axis('equal')
#plt.show()

# Festlegen der Basis
basis = pysgpp.SBsplineBase(degree)

# Erzeugen von Gitter
grid = pysgpp.Grid.createLinearBoundaryGrid(dim, degree)
gridStorage = grid.getStorage()
print("dimensionality:           {}".format(gridStorage.getDimension()))
grid.getGenerator().full(level)
print("number of grid points:    {}".format(gridStorage.getSize()))

# Vektor 'x' enthaelt Koordinaten von Gitterpunkten
# anschl. auswerten von Gewichtsfunktion des Kreises an Gitterpunkten
x = np.zeros((gridStorage.getSize(), dim))
index_x = np.zeros((gridStorage.getSize()))
lvl = np.zeros((gridStorage.getSize(), dim))
ind = np.zeros((gridStorage.getSize(), dim))
eval_circle = np.zeros(gridStorage.getSize())


for i in range(gridStorage.getSize()):
    gp = gridStorage.getPoint(i)
    lvl[i] = [gp.getLevel(0), gp.getLevel(1)]
    ind[i] = [gp.getIndex(0), gp.getIndex(1)]
    x[i] = [gp.getStandardCoordinate(0), gp.getStandardCoordinate(1)]
    index_x[i] = i   
    eval_circle[i] = weightfunction.circle(radius, x[i])

print(x)

# Ueberpruefung auf innere und aeussere Punkte 
p0 = 0
n0 = 0
for i in range(len(eval_circle)):
    if eval_circle[i] > 0:
        p0 = p0 + 1
    else:
        n0 = n0 + 1
I_all = np.zeros((p0, dim))
J_all = np.zeros((n0, dim))
p1 = 0
n1 = 0 
for i in range(len(eval_circle)):
    if eval_circle[i] > 0:
        I_all[p1] = x[i]
        p1 = p1 + 1
    else:
        J_all[n1] = x[i]
        n1 = n1 + 1

# Plot von inneren und aeusseren Punkten 
plt.scatter(I_all[:, 0], I_all[:, 1], c='mediumblue', s=50, lw=0)
plt.scatter(J_all[:, 0], J_all[:, 1], c='crimson', s=50, lw=0)
plt.axis('equal')
plt.show()

# Index der inneren Punkte unter Gesamtpunkten x
index_I_all = np.zeros(len(I_all))
for i in range(len(I_all)):
    for j in range(len(x)):
        if I_all[i, 0] == x[j, 0] and I_all[i, 1] == x[j, 1]:
            index_I_all[i] = j

# Index der aeusseren Punkte unter Gesamtpunkten x
index_J_all = np.zeros(len(J_all))
for i in range(len(J_all)):
    for j in range(len(x)):
        if J_all[i, 0] == x[j, 0] and J_all[i, 1] == x[j, 1]:
            index_J_all[i] = j


# Bestimme Gitterweite (h_x,h_y) in Abhaengigkeit vom Level

h = np.zeros((len(x),dim))
for i in range(len(h)):
    h[i] = 2**(-lvl[i])


# 3D Matrix
supp_points = np.zeros((len(J_all),(degree+2)**dim, dim))
for l in range(len(J_all)):
    m = 0
    for i in range(int(-(degree+1)/2),int((degree+1)/2)+1):
        for j in range(int(-(degree+1)/2),int((degree+1)/2)+1):
            supp_points[l,m,0] = J_all[l,0]+i*h[int(index_J_all[l]),0]
            supp_points[l,m,1] = J_all[l,1]+j*h[int(index_J_all[l]),1]  
            m=m+1
#print(supp_points)


# Bestimme relevante aeussere Punkte durch Auswerten der Gewichtsfunktion an supp_points.
# Falls Gewichtsfunktion an einem supp_point positiv, dann ist es relevanter aeusserer Punkt
J_relevant = np.zeros(dim)
eval_supp_points_circle = np.zeros(supp_points.shape[1])
for i in range(supp_points.shape[0]):
    for j in range(supp_points.shape[1]):
        eval_supp_points_circle[j] = weightfunction.circle(radius, supp_points[i,j])
    if (eval_supp_points_circle > 0).any():
        J_relevant = np.vstack((J_relevant, J_all[i]))      
J_relevant = np.delete(J_relevant, 0, 0)
#print(J_relevant)

# Index der relevanten aeusseren Punkte unter Gesamtpunkten x
index_J_relevant = np.zeros(len(J_relevant))
for i in range(len(J_relevant)):
    for j in range(len(x)):
        if J_relevant[i, 0] == x[j, 0] and J_relevant[i, 1] == x[j, 1]:
            index_J_relevant[i] = j
     
          

# # Plotte relevante Punkte
# plt.scatter(I_all[:, 0], I_all[:, 1], c='mediumblue', s=50, lw=0)
# plt.scatter(J_relevant[:, 0], J_relevant[:, 1], c='goldenrod', s=50, lw=0)
# #plt.scatter(x[index_J_relevant[10], 0], x[index_J_relevant[10], 1], c='cyan', s=50, lw=0)
# #plt.scatter(x[index_I_all[25], 0], x[index_I_all[25], 1], c='red', s=50, lw=0)
# plt.axis('equal')
# plt.show()


# Anzahl nearest neighbors
n_neighbors = (degree+1)**2

# Monome definieren und an allen Gitterpunkten auswerten
size_monomials = n_neighbors
eval_monomials = np.zeros((size_monomials, gridStorage.getSize()))
k = 0
for j in range(degree + 1):
    for i in range (degree + 1):
        if i + j <= degree:
            eval_monomials[k] = (pow(x[:, 0], i) * pow(x[:, 1], j))
            k = k + 1    
eval_monomials = np.transpose(eval_monomials)
#print(eval_monomials)

# Definiere Matrix A fuer Interpolation der coeffs
A = np.zeros((gridStorage.getSize(), gridStorage.getSize()))
for i in range(gridStorage.getSize()):
    for j in range(gridStorage.getSize()):
        A[i, j] = basis.eval(int(lvl[j, 0]), int(ind[j, 0]), x[i, 0]) * basis.eval(int(lvl[j, 1]), int(ind[j, 1]), x[i, 1])   
#print(A)

# x_str = np.array_repr(A).replace('\n', '')
# print(x_str)

# Loese LGS und erhalte coeffs
coeffs = np.linalg.solve(A, eval_monomials)
#print(coeffs)

# # Beliebige Punkte im Gebiet
# punkte = np.zeros((20, 2))
# counter = 0 
# while counter < 20:
#     z = np.random.rand(1, 2)
#     if weightfunction.circle(radius, z[0]) > 0:
#         punkte[counter] = z[0]
#         counter = counter + 1
# for j in range(punkte.shape[0]):
#     summe = 0
#     for i in range(len(x)):
#         summe = summe + coeffs[i,0]*basis.eval(int(lvl[i,0]), int(ind[i,0]), punkte[j,0])*basis.eval(int(lvl[i,1]), int(ind[i,1]), punkte[j,1])
#     print(punkte[j])
#     print(summe)
#     fehler = summe - 1                         #coeffs[i,0], Monom 1
#     #    fehler = summe - punkte[j,0]                #coeffs[i,1], Monom x
#     #    fehler = summe - punkte[j,1]               #coeffs[i,2], Monom y
#     #    fehler = summe-(punkte[j,0]*punkte[j,1])   #coeffs[i,3], Monom x*y
#     print(fehler)
#     printLine()


# # Beliebige Punkte im Einheitsquadrat
# punkte = np.random.rand(20,2)
# for j in range(punkte.shape[0]):
#     summe = 0
#     for i in range(len(x)):
#         summe = summe + coeffs[i,0]*basis.eval(int(lvl[i,0]), int(ind[i,0]), punkte[j,0])*basis.eval(int(lvl[i,1]), int(ind[i,1]), punkte[j,1])
#     print(punkte[j])
#     print(summe)
#     fehler = summe - 1                         #coeffs[i,0], Monom 1
# #    fehler = summe - punkte[j,0]                #coeffs[i,1], Monom x
# #    fehler = summe - punkte[j,1]               #coeffs[i,2], Monom y
# #    fehler = summe-(punkte[j,0]*punkte[j,1])   #coeffs[i,3], Monom x*y
#     print(fehler)
#     printLine()
    
        
# Test ob Loesen des LGS erfolgreich war
error = eval_monomials - np.matmul(A, coeffs)
error = LA.norm(error)
if error > pow(10, -14):
    print('failed. error > 10e-14')



