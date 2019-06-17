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
from numpy import linalg as LA, transpose


def printLine():
    print("--------------------------------------------------------------------------------------")


# Zielfunktion
def function(x):
    f = 0
    f = np.sin(8 * x[0]) + np.sin(7 * x[1])
    f = f * weightfunction.circle(radius, x)
    return f

# Mit WEB-Splines interpolierte Funktion
def function_tilde(p, l):
    f_tilde = 0
    for i in range(len(I_all)):      
        f_tilde = f_tilde + alpha[i] * WEBspline(p, i, l)
    return f_tilde


dim = 2  # Dimension
radius = 0.3  # Radius von Kreis
degree = 1  # Grad von B-Splines (nur ungerade)
level = 3       # Level von Sparse Grid

p = np.zeros((10000, 2))
counter = 0 
while counter < 10000:
    z = np.random.rand(1, 2)
    if weightfunction.circle(radius, z[0]) > 0:
        p[counter] = z[0]
        counter = counter + 1

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
grid = pysgpp.Grid.createWEBsplineGrid(dim, degree)
gridStorage = grid.getStorage()
print("dimensionality:           {}".format(gridStorage.getDimension()))
grid.getGenerator().regular(level)
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
#plt.show()

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


# Bestimme relevante aeussere Punkte durch Auswerten der Gewichtsfunktion an den Eckpunkten.
# Falls Gewichtsfunktion an einem Eckpunkt positiv, dann ist es relevanter aeusserer Punkt
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
    
            

# Plotte relevante Punkte
plt.scatter(I_all[:, 0], I_all[:, 1], c='mediumblue', s=50, lw=0)
plt.scatter(J_relevant[:, 0], J_relevant[:, 1], c='goldenrod', s=50, lw=0)
#plt.scatter(x[30, 0], x[30, 1], c='cyan', s=50, lw=0)
plt.axis('equal')
#plt.show()

# Anzahl Nearest Neighbors
n_neighbors = (degree + 1) ** dim
print("Anzahl Nearest Neighbors: {}".format(n_neighbors))

# Berechne Abstand der relevanten aeusseren Punkte zu allen inneren Punkten und sortiere nach Abstand
diff = np.zeros((len(I_all), dim))
distance = np.zeros((len(I_all), dim))

# k=1: nearest neighbors fuer alle relevanten aeusseren Punkten
# k=0: nearest neighbors fuer einen bestimmten relevanten aeusseren Punkt
k = 1
if k == 1:
    NN = np.zeros((n_neighbors, dim * len(J_relevant)))
    for j in range(len(J_relevant)):
        for i in range(len(I_all)):
            diff[i] = I_all[i] - J_relevant[j]
            distance[i, 0] = LA.norm(diff[i])
            distance[i, 1] = i
            sort = distance[np.argsort(distance[:, 0])]

# Loesche Punkte die Anzahl Nearest Neighbor ueberschreitet
        i = len(I_all) - 1
        while i >= n_neighbors:
            sort = np.delete(sort, i , 0)
            i = i - 1 

# Bestimme die Nearest Neighbor inneren Punkte 
        for i in range(len(sort)):
            NN[i, dim * j] = I_all[int(sort[i, 1]), 0]
            NN[i, dim * j + 1] = I_all[int(sort[i, 1]), 1]
        
# Index der nearest neighbor Punkte unter allen Punkten x
    index_NN = np.zeros((n_neighbors, len(J_relevant)))
    for j in range(len(J_relevant)):
        for i in range(n_neighbors):
            for k in range(len(x)):
                if NN[i, dim * j] == x[k, 0] and NN[i, dim * j + 1] == x[k, 1]:
                    index_NN[i, j] = k

# Nearest Neighbors sortieren nach Index im Gesamtgitter
    index_NN=np.sort(index_NN,axis=0)
    for j in range(index_NN.shape[1]):
        for i in range(index_NN.shape[0]):
            NN[i,dim * j] = x[int(index_NN[i,j]),0]
            NN[i,dim * j +1] = x[int(index_NN[i,j]),1]

# Plot der nearest neighbors 
    for i in range(len(J_relevant)):
        plt.scatter(I_all[:, 0], I_all[:, 1], c='mediumblue', s=50, lw=0)
        plt.scatter(J_relevant[:, 0], J_relevant[:, 1], c='goldenrod', s=50, lw=0)
        plt.scatter(J_relevant[i, 0], J_relevant[i, 1], c='cyan', s=50, lw=0) 
        plt.scatter(NN[:, dim * i], NN[:, dim * i + 1], c='limegreen', s=50, lw=0)
        #plt.contour(X[0], X[1], Z, 0)
        plt.axis('equal')
        #plt.show()
else:
    j = 0  # Setze j auf den Index des zu betrachtenden aeusseren Punktes
    for i in range(len(I_all)):
        diff[i] = I_all[i] - J_relevant[j]
        distance[i, 0] = LA.norm(diff[i])
        distance[i, 1] = i
        sort = distance[np.argsort(distance[:, 0])]

# Loesche Punkte die Anzahl Nearest Neighbor ueberschreitet
    i = len(I_all) - 1
    while i >= n_neighbors:
        sort = np.delete(sort, i , 0)
        i = i - 1

# Bestimme die Nearest Neighbor inneren Punkte 
    NN = np.zeros((len(sort), dim))
    for i in range(len(sort)):
        NN[i] = I_all[int(sort[i, 1])]

# Index der nearest neighbor Punkte unter allen Punkten x
    index_NN = np.zeros(n_neighbors)
    for i in range(len(NN)):
        for k in range(len(x)):
            if NN[i, 0] == x[k, 0] and NN[i, 1] == x[k, 1]:
                index_NN[i] = k
                
# Plot der nearest neighbors fuer einen Punkt aeusseren Punkt j
    plt.scatter(I_all[:, 0], I_all[:, 1], c='mediumblue', s=50, lw=0)
    plt.scatter(J_relevant[:, 0], J_relevant[:, 1], c='goldenrod', s=50, lw=0)
    plt.scatter(J_relevant[j, 0], J_relevant[j, 1], c='cyan', s=50, lw=0)
    plt.scatter(NN[:, 0], NN[:, 1], c='limegreen', s=50, lw=0)
    plt.axis('equal')
    plt.show()



# Monome definieren und an allen Gitterpunkten auswerten
if degree == 1:
    size_monomials = 4
    eval_monomials = np.zeros((size_monomials, gridStorage.getSize()))
    k = 0
    for j in range(2):
        for i in range(2):
            if i + j <= 2 :
                eval_monomials[k] = (pow(x[:, 0], i) * pow(x[:, 1], j))
                k = k + 1    
    eval_monomials = np.transpose(eval_monomials)
else:
    size_monomials = int(scipy.special.binom(degree+dim, dim))   
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


x_str = np.array_repr(A).replace('\n', '')
print(x_str)

# Loese LGS und erhalte coeffs
coeffs = np.linalg.solve(A, eval_monomials)
print(coeffs)

# Beliebige Punkte im Einheitsquadrat
punkte = np.random.rand(10,2)
print(punkte)
for j in range(punkte.shape[0]):
    summe = 0
    for i in range(len(x)):
        summe = summe + coeffs[i,0]*basis.eval(int(lvl[i,0]), int(ind[i,0]), punkte[j,0])*basis.eval(int(lvl[i,1]), int(ind[i,1]), punkte[j,1])
    print(summe)
    fehler = summe - 1                         #coeffs[i,0], Monom 1
#    fehler = summe - punkte[j,0]                #coeffs[i,1], Monom x
#    fehler = summe - punkte[j,1]               #coeffs[i,2], Monom y
#    fehler = summe-(punkte[j,0]*punkte[j,1])   #coeffs[i,3], Monom x*y
    print(fehler)
    printLine()

        
a = np.linalg.solve(A, eval_monomials[:,1])
#print(a)



# Test ob Loesen des LGS erfolgreich war
error = eval_monomials - np.matmul(A, coeffs)
error = LA.norm(error)
if error > pow(10, -14):
    print('failed. error > 10e-14')



# Definiere Koeffizientenmatrix der aeusseren relevanten Punkte
coeffs_J_relevant = np.zeros((len(J_relevant), size_monomials))
for j in range(size_monomials):
    for i in range(len(J_relevant)):
        coeffs_J_relevant[i,j] = coeffs[int(index_J_relevant[i]),j]
coeffs_J_relevant = transpose(coeffs_J_relevant)
#print(coeffs_J_relevant)



coeffs_inner_NN = np.zeros((len(J_relevant),size_monomials, n_neighbors))
for k in range(len(J_relevant)):
    for j in range(n_neighbors):
        for i in range(size_monomials):
            coeffs_inner_NN[k,i,j] = coeffs[int(index_NN[j,k]),i]
#print(coeffs_inner_NN)

extension_coeffs = np.zeros((len(J_relevant),n_neighbors))
for i in range(coeffs_inner_NN.shape[0]):
    solution=np.linalg.lstsq(coeffs_inner_NN[i], coeffs_J_relevant[:,i])
    extension_coeffs[i] = solution[0]
extension_coeffs = transpose(extension_coeffs)
#print(extension_coeffs)


# Definiere J(i) 
J_i = np.zeros((index_NN.shape[1], len(I_all)))  # x
for i in range(len(I_all)):#index_x
    for j in range((index_NN.shape[1])):
        for k in range((index_NN.shape[0])):
            if index_x[i] == index_NN[k, j]:
                # print(i,index_J_relevant[j])
                J_i[j, i] = index_J_relevant[j]
#print(J_i)
 
# Matrix A_WEB mit WEB Splines ausgewertet an inneren Punkten fuellen: a_l,i = WEBspline_i(x_l) fuer alle i,l in innere Punkte I
A_WEB = np.zeros((len(I_all), len(I_all)))  
for i in range(len(I_all)):
    k = 0
    for l in range(len(x)):
        if eval_circle[l] > 0:
            bi = basis.eval(int(lvl[int(index_I_all[i]), 0]), int(ind[int(index_I_all[i]), 0]), x[l, 0]) * basis.eval(int(lvl[int(index_I_all[i]), 1]), int(ind[int(index_I_all[i]), 1]), x[l, 1])   
            sum = 0
            for j in range(index_NN.shape[1]):
                for m in range(index_NN.shape[0]):
                    if index_NN[m, j] == index_I_all[i] and J_i[j, i] != 0: 
                        sum = sum + extension_coeffs[m, j] * basis.eval(int(lvl[int(J_i[j, i]), 0]), int(ind[int(J_i[j, i]), 0]), x[l, 0]) * basis.eval(int(lvl[int(J_i[j, i]), 1]), int(ind[int(J_i[j, i]), 1]), x[l, 1])
            extended_Bspline = bi + sum 
            WEBspline = weightfunction.circle(radius, x[l]) * extended_Bspline
            A_WEB[k, i] = WEBspline
            k = k + 1
#print(A_WEB)


# Zielfunktion auswerten an inneren Punkten 
ev_f = np.zeros((len(I_all), 1))
for i in range(len(I_all)):
    ev_f[i] = function(x[int(index_I_all[i])])
    #ev_f[i] = x[i,0]+x[i,1]
#print(ev_f) 
 
# LGS loesen fuer Interpolationskoeffizient alpha
alpha = np.linalg.solve(A_WEB, ev_f)
#print(alpha)
 
# # Interpolation von f und Fehlerberechnung 
# err = 0
# for l in range(len(p)):
#     f = np.sin(8 * p[l,0]) + np.sin(7 * p[l,1])
#    # f = p[l,0]+p[l,1]
#     f = f * weightfunction.circle(radius, p[l])
#     f_tilde = 0
#     for i in range(len(I_all)):
#         bi = basis.eval(int(lvl[int(index_I_all[i]), 0]), int(ind[int(index_I_all[i]), 0]), p[l, 0]) * basis.eval(int(lvl[int(index_I_all[i]), 1]), int(ind[int(index_I_all[i]), 1]), p[l, 1])   
#         sum = 0
#         for j in range(index_NN.shape[1]):
#             for m in range(index_NN.shape[0]):
#                 if index_NN[m, j] == index_I_all[i] and J_i[j, i] != 0: 
#                     sum = sum + extension_coeffs[m, j] * basis.eval(int(lvl[int(J_i[j, i]), 0]), int(ind[int(J_i[j, i]), 0]), p[l, 0]) * basis.eval(int(lvl[int(J_i[j, i]), 1]), int(ind[int(J_i[j, i]), 1]), p[l, 1])
#         extended_Bspline = bi + sum 
#         WEBspline = weightfunction.circle(radius, p[l]) * extended_Bspline      
#         f_tilde = f_tilde + alpha[i] * WEBspline
#     err = err + (f - f_tilde) ** 2
# err = err ** (1 / 2)
# print('error : {}'.format(err[0]))  


