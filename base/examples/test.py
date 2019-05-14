import pysgpp
import math
import sys
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import weightfunction
from mpl_toolkits import mplot3d
from numpy import linalg as LA
import nearestneighbors


    
def printLine():
    print("--------------------------------------------------------------------------------------")
    
dim = 2         # Dimension
radius = 0.1    # Radius von Kreis
degree = 3      # Grad von B-Splines
level = 4       # Level von Sparse Grid

# Gitter für Kreis erzeugen und auswerten
x0 = np.linspace(0, 1, 50)
if dim == 2:
    X = np.meshgrid(x0, x0)
elif dim == 3:
    X = np.meshgrid(x0, x0, x0)      
Z = weightfunction.circle(radius,X)

# Plot von Kreis
plt.contour(X[0], X[1], Z, 0)
plt.axis('equal')
#plt.show()

# Erzeugen von Gitter
grid = pysgpp.Grid.createWEBsplineGrid(dim, degree)
gridStorage = grid.getStorage()
print("dimensionality:           {}".format(gridStorage.getDimension()))
grid.getGenerator().regular(level)
print("number of grid points:    {}".format(gridStorage.getSize()))

# Vektor 'x' enthält Koordinaten von Gitterpunkten
# anschl. auswerten von Gewichtsfunktion des Kreises an Gitterpunkten
x = np.zeros((gridStorage.getSize(),dim))
eval_circle= np.zeros(gridStorage.getSize())

if dim == 2:
    for i in range(gridStorage.getSize()):
        gp = gridStorage.getPoint(i)
        x[i] = [gp.getStandardCoordinate(0), gp.getStandardCoordinate(1)]   
        eval_circle[i]=weightfunction.circle(radius, x[i])
elif dim == 3:
    for i in range(gridStorage.getSize()):
        gp = gridStorage.getPoint(i)
        x[i] = [gp.getStandardCoordinate(0), gp.getStandardCoordinate(1), gp.getStandardCoordinate(2)]
        eval_circle[i]=weightfunction.circle(radius, x[i])            


# Überprüfung auf innere und äußere Punkte 
p0=0
n0=0
index_I = np.zeros(len(eval_circle))
#index_J = np.zeros(len(eval_circle))
for i in range(len(eval_circle)):
    if eval_circle[i] > 0:
        p0=p0+1
        index_I[i] = i
    else:
        n0=n0+1
        #index_J[i]= i
I_all = np.zeros((p0,dim))
J_all = np.zeros((n0,dim))
p1=0
n1=0 
for i in range(len(eval_circle)):
    if eval_circle[i] > 0:
        I_all[p1]=x[i]
        p1=p1+1
    else:
        J_all[n1]=x[i]
        n1=n1+1
print(index_I)
#print(index_J)

# Plot von inneren und äußeren Punkten 
if dim == 2:
    #ax = plt.axes(projection='3d')
    #ax.contour3D(X[0], X[1], Z, 50, cmap='binary')
    plt.scatter(I_all[:,0], I_all[:,1], c='mediumblue', s=50, lw=0)
    plt.scatter(J_all[:,0], J_all[:,1], c='crimson', s=50, lw=0)
elif dim == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(I_all[:,0], I_all[:,1], I_all[:,2], c='mediumblue', s=50, lw=0)
    ax.scatter(J_all[:,0], J_all[:,1], J_all[:,2], c='crimson', s=50, lw=0)
plt.axis('equal')
#plt.show()

# Bestimme Gitterweite h
h = 2**(-level)
print("Gitterweite:              {}".format((degree-1)*h))

# Bestimme Eckpunkte von Träger von Punkt (x,y)
J_relevant = np.zeros(dim)
if dim == 2:
    for i in range(len(J_all)):
        if weightfunction.circle(radius, J_all[i]-(degree-1)*h ) > 0 or weightfunction.circle(radius, [J_all[i,0]-(degree-1)*h, J_all[i,1]+(degree-1)*h]) > 0 or weightfunction.circle(radius, [J_all[i,0]+(degree-1)*h, J_all[i,1]-(degree-1)*h]) > 0 or weightfunction.circle(radius, J_all[i]+(degree-1)*h) > 0: 
            J_relevant = np.vstack((J_relevant, J_all[i]))            
J_relevant = np.delete(J_relevant, 0, 0)     


# Plotte relevante Punkte
if dim == 2:
    #ax = plt.axes(projection='3d')
    #ax.contour3D(X[0], X[1], Z, 50, cmap='binary')
    plt.scatter(I_all[:,0], I_all[:,1], c='mediumblue', s=50, lw=0)
    plt.scatter(J_relevant[:,0], J_relevant[:,1], c='goldenrod', s=50, lw=0)
plt.axis('equal')
#plt.show()

# Anzahl Nearest Neighbors
n_neighbors = (degree+1)**dim
print("Anzahl Nearest Neighbors: {}".format(n_neighbors))

# Punkt 5 ist der links oben für den nearest neighbors getestet wird.
# Berechne Abstand der relevanten äußeren Punkte zu allen inneren Punkten und sortiere nach Abstand
diff = np.zeros((len(I_all), dim))
distance = np.zeros((len(I_all),dim))


k=0
if k==1:
    for j in range(len(J_relevant)):
        for i in range(len(I_all)):
            diff[i] = I_all[i]-J_relevant[j]
            distance[i,0] = LA.norm(diff[i])
            distance[i,1] = i
            sort = distance[np.argsort(distance[:,0])]

# Lösche Punkte die Anzahl Nearest Neighbor überschreitet
        i = len(I_all)-1
        while i >= n_neighbors:
            sort = np.delete(sort, i , 0)
            i = i-1

# Bestimme die Nearest Neighbor inneren Punkte 
        NN = np.zeros((len(sort), dim))
        for i in range(len(sort)):
            NN[i] = I_all[int(sort[i,1])]

        plt.scatter(I_all[:,0], I_all[:,1], c='mediumblue',s=50,lw=0)
        plt.scatter(J_relevant[:,0], J_relevant[:,1], c='goldenrod',s=50,lw=0)
        plt.scatter(J_relevant[j,0], J_relevant[j,1], c='cyan',s=50,lw=0) 
        plt.scatter(NN[:,0], NN[:,1], c='limegreen',s=50,lw=0)
else:
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
    plt.scatter(I_all[:,0], I_all[:,1], c='mediumblue',s=50,lw=0)
    plt.scatter(J_relevant[:,0], J_relevant[:,1], c='goldenrod',s=50,lw=0)
    plt.scatter(J_relevant[j,0], J_relevant[j,1], c='cyan',s=50,lw=0)
    plt.scatter(NN[:,0], NN[:,1], c='limegreen',s=50,lw=0)
# Index NN Punkte in Gesamtgitterpunkte x
    index_NN = np.zeros(n_neighbors)
    for i in range(len(NN)):
        for j in range(len(x)):
            if NN[i,0]==x[j,0] and NN[i,1]==x[j,1]:
                index_NN[i] = j
    print('Index_NN = {}'.format(index_NN))
    print(NN)
plt.axis('equal')
#plt.show()
        
# Monome
x_all = x[:,0]
y_all = x[:,1]
size_mon = np.int((degree+1)*(degree+2)/2)

eval_monomials_py = np.zeros((size_mon, gridStorage.getSize()))
#print(eval_monomials_py)
k=0
for i in range(degree+1):
    for j in range (degree+1):
        if i+j<=3:
            #print(pow(x,i)*pow(y,j))
            eval_monomials_py[k]=(pow(x_all,i)*pow(y_all,j))
            k=k+1
        
eval_monomials_py = np.transpose(eval_monomials_py)
#print(eval_monomials_py)

eval_monomials = pysgpp.DataMatrix(gridStorage.getSize(), size_mon)
for j in range(size_mon):
    for i in range(gridStorage.getSize()):
        eval_monomials.set(i,j,eval_monomials_py[i,j])
#print(eval_monomials)

# Interpolation über alle Punkte für Koeffizientenmatrix aller Punkte
printLine()
coeffs_all = pysgpp.DataMatrix(gridStorage.getSize(), size_mon)
hierSLE = pysgpp.OptHierarchisationSLE(grid)
sleSolver = pysgpp.OptAutoSLESolver()
if not sleSolver.solve(hierSLE, eval_monomials, coeffs_all):
    print("Solving failed, exiting.")
    sys.exit(1)

# Result of SLE 
# coeffs_all ist 49x10 Matrix 
#print("coeffs_all solved: {}".format(coeffs_all)) 
printLine()
print('Anzahl coeffs_all = {}'.format(coeffs_all.getSize()))
print('Reihen coeffs_all = {}'.format(coeffs_all.getNrows()))
print('Spalten coeffs_all = {}'.format(coeffs_all.getNcols()))

printLine()
# Index äußere Punkte in Gesamtgitterpunkte x
index_J = np.zeros(len(J_relevant))
for i in range(len(J_relevant)):
    for j in range(len(x)):
        if J_relevant[i,0]==x[j,0] and J_relevant[i,1]==x[j,1]:
            index_J[i] = j
print('Index_J = {}'.format(index_J))

printLine()

# Definiere Extension Koeffizienten 
extension_coeffs = pysgpp.DataMatrix(len(NN),1)

# Definiere Koeffizientenmatrix von Nearest Neighbors Punkten und befüllen der Matrix
coeffs_NN = pysgpp.DataMatrix(len(NN),coeffs_all.getNcols())
c1=pysgpp.DataVector(coeffs_all.getNcols())
for i in range(len(NN)):
    coeffs_all.getRow(int(index_NN[i]),c1)
    coeffs_NN.setRow(i,c1)
printLine()
coeffs_NN.transpose()
#print(coeffs_NN)

# Definiere Koeffizientenmatrix der äußeren relevanten Punkte und befüllen der Matrix(=Zielvektor)
coeffs_J = pysgpp.DataMatrix(coeffs_all.getNcols(),1)
c2 = pysgpp.DataVector(coeffs_all.getNcols())
coeffs_all.getRow(int(index_J[0]), c2)
coeffs_J.setColumn(0,c2)
print(coeffs_NN)
printLine()
print(extension_coeffs)
print(coeffs_J)


fullSLE = pysgpp.OptFullSLE(coeffs_NN)
sleSolver = pysgpp.OptAutoSLESolver()
if not sleSolver.solve(fullSLE, coeffs_J, extension_coeffs):
    print("Solving failed, exiting.")
    sys.exit(1)
    
print(extension_coeffs)


print(J_relevant)

#q=pysgpp.DataMatrix(2,2)
#q.set(0,0,2)
#q.set(0,1,3)
#q.set(1,0,1)
#q.set(1,1,5)
#w=pysgpp.DataVector(2)
#w.set(0,5)
#w.set(1,6)
#e=pysgpp.DataVector(2)


#fullSLE = pysgpp.OptFullSLE(q)
#sleSolver = pysgpp.OptAutoSLESolver()
#if not sleSolver.solve(fullSLE, w, e):
#    print("Solving failed, exiting.")
#    sys.exit(1)

#print(e)





