import pysgpp
import math
import sys
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import weightfunction
from mpl_toolkits import mplot3d
from numpy import linalg as LA


def printLine():
    print("--------------------------------------------------------------------------------------")


# Zielfunktion
def function(x):
    f = 0
    f = np.sin(8 * x[0]) + np.sin(7 * x[1])
    f = f * weightfunction.circle(radius, x)
    return f


# Definiere WEB-Splines
def WEBspline(x, i, l):
    J_I = np.zeros((index_NN.shape[1], len(I_all)))  # x
    #print(J_I.shape)
#    for i in range(len(I_all)):#index_x
    for j in range((index_NN.shape[1])):
        for k in range((index_NN.shape[0])):
            if index_x[i] == index_NN[k, j]:
                # print(i,index_J_relevant[j])
                J_I[j, i] = index_J_relevant[j]

#    for i in range(len(I_all)):
    bi = basis.eval(int(lvl[int(index_I_all[i]), 0]), int(ind[int(index_I_all[i]), 0]), x[l, 0]) * basis.eval(int(lvl[int(index_I_all[i]), 1]), int(ind[int(index_I_all[i]), 1]), x[l, 1])   
    # print(bi)
    sum = 0
    for j in range(index_NN.shape[1]):
        for k in range(index_NN.shape[0]):
            if index_NN[k, j] == index_I_all[i] and J_I[j, i] != 0: 
                # print(k,j)
                sum = sum + extension_coeffs[k, j] * basis.eval(int(lvl[int(J_I[j, i]), 0]), int(ind[int(J_I[j, i]), 0]), x[l, 0]) * basis.eval(int(lvl[int(J_I[j, i]), 1]), int(ind[int(J_I[j, i]), 1]), x[l, 1])
                # print(extension_coeffs[k,j])
                # print(sum)
    extended_Bspline = bi + sum 
    WEBspline = weightfunction.circle(radius, x[l]) * extended_Bspline
    # printLine()
    # print(WEBspline)
    return WEBspline


# Mit WEB-Splines interpolierte Funktion
def function_tilde(p, l):
    f_tilde = 0
    for i in range(len(I_all)):      
        f_tilde = f_tilde + alpha[i] * WEBspline(p, i, l)
    return f_tilde


dim = 2  # Dimension
radius = 0.1  # Radius von Kreis
degree = 3  # Grad von B-Splines (nur ungerade)
#level = 5       # Level von Sparse Grid

p = np.zeros((10000, 2))
counter = 0 
while counter < 10000:
    z = np.random.rand(1, 2)
    if weightfunction.circle(radius, z[0]) > 0:
        p[counter] = z[0]
        counter = counter + 1

error = np.zeros((4,1))
Level = np.zeros((4,1))

          
for v in range(4,8):
    level = v
    
    # Gitter für Kreis erzeugen und auswerten
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
    
    # Vektor 'x' enthält Koordinaten von Gitterpunkten
    # anschl. auswerten von Gewichtsfunktion des Kreises an Gitterpunkten
    x = np.zeros((gridStorage.getSize(), dim))
    index_x = np.zeros((gridStorage.getSize()))
    lvl = np.zeros((gridStorage.getSize(), dim))
    ind = np.zeros((gridStorage.getSize(), dim))
    eval_circle = np.zeros(gridStorage.getSize())
    
    if dim == 2:
        for i in range(gridStorage.getSize()):
            gp = gridStorage.getPoint(i)
            lvl[i] = [gp.getLevel(0), gp.getLevel(1)]
            ind[i] = [gp.getIndex(0), gp.getIndex(1)]
            x[i] = [gp.getStandardCoordinate(0), gp.getStandardCoordinate(1)]
            index_x[i] = i   
            eval_circle[i] = weightfunction.circle(radius, x[i])
    # elif dim == 3:
    #    for i in range(gridStorage.getSize()):
    #        gp = gridStorage.getPoint(i)
    #        x[i] = [gp.getStandardCoordinate(0), gp.getStandardCoordinate(1), gp.getStandardCoordinate(2)]
    #        eval_circle[i]=weightfunction.circle(radius, x[i])            
    
    A = np.zeros((gridStorage.getSize(), gridStorage.getSize()))
    # print(x)
    # print('lvl  : {}'.format(lvl))
    # print('ind: {}'.format(ind))
    
    for i in range(gridStorage.getSize()):
        for j in range(gridStorage.getSize()):
            A[i, j] = basis.eval(int(lvl[j, 0]), int(ind[j, 0]), x[i, 0]) * basis.eval(int(lvl[j, 1]), int(ind[j, 1]), x[i, 1])
    # print(A)
    
    # Überprüfung auf innere und äußere Punkte 
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
    
    # Plot von inneren und äußeren Punkten 
    if dim == 2:
        plt.scatter(I_all[:, 0], I_all[:, 1], c='mediumblue', s=50, lw=0)
        plt.scatter(J_all[:, 0], J_all[:, 1], c='crimson', s=50, lw=0)
    # elif dim == 3:
    #    fig = plt.figure()
    #    ax = fig.add_subplot(111, projection='3d')
    #    ax.scatter(I_all[:,0], I_all[:,1], I_all[:,2], c='mediumblue', s=50, lw=0)
    #    ax.scatter(J_all[:,0], J_all[:,1], J_all[:,2], c='crimson', s=50, lw=0)
    plt.axis('equal')
    #plt.show()
    
    # Bestimme Gitterweite h
    h = 2 ** (-level)
    print("Gitterweite:              {}".format(h))
    
    # Bestimme Eckpunkte von Träger von Punkt (x,y)
    J_relevant = np.zeros(dim)
    for i in range(len(J_all)):
        if weightfunction.circle(radius, J_all[i] - ((degree/2+0.5) * h)) > 0 or weightfunction.circle(radius, [J_all[i, 0] - ((degree/2+0.5) * h), J_all[i, 1] + ((degree/2+0.5) * h)]) > 0 or weightfunction.circle(radius, [J_all[i, 0] + ((degree/2+0.5) * h), J_all[i, 1] - ((degree/2+0.5) * h)]) > 0 or weightfunction.circle(radius, J_all[i] + ((degree/2+0.5) * h)) > 0: 
            J_relevant = np.vstack((J_relevant, J_all[i]))            
    J_relevant = np.delete(J_relevant, 0, 0)
    
    # Index der relevanten äußeren Punkte unter Gesamtpunkten x
    index_J_relevant = np.zeros(len(J_relevant))
    
    for i in range(len(J_relevant)):
        for j in range(len(x)):
            if J_relevant[i, 0] == x[j, 0] and J_relevant[i, 1] == x[j, 1]:
                index_J_relevant[i] = j
    # print(index_J_relevant)    
                
    # Index der inneren Punkte unter Gesamtpunkten x
    index_I_all = np.zeros(len(I_all))
    
    for i in range(len(I_all)):
        for j in range(len(x)):
            if I_all[i, 0] == x[j, 0] and I_all[i, 1] == x[j, 1]:
                index_I_all[i] = j
    # print(index_I_all)
    
    # Plotte relevante Punkte
    if dim == 2:
        plt.scatter(I_all[:, 0], I_all[:, 1], c='mediumblue', s=50, lw=0)
        plt.scatter(J_relevant[:, 0], J_relevant[:, 1], c='goldenrod', s=50, lw=0)
    plt.axis('equal')
    #plt.show()
    
    # Anzahl Nearest Neighbors
    n_neighbors = (degree + 1) ** dim
    print("Anzahl Nearest Neighbors: {}".format(n_neighbors))
    
    # Berechne Abstand der relevanten äußeren Punkte zu allen inneren Punkten und sortiere nach Abstand
    diff = np.zeros((len(I_all), dim))
    distance = np.zeros((len(I_all), dim))
    
    # k=1: nearest neighbors für alle relevanten äußeren Punkten
    # k=0: nearest neighbors für einen bestimmten relevanten äußeren Punkt
    k = 1
    if k == 1:
        NN = np.zeros((n_neighbors, dim * len(J_relevant)))
        for j in range(len(J_relevant)):
            for i in range(len(I_all)):
                diff[i] = I_all[i] - J_relevant[j]
                distance[i, 0] = LA.norm(diff[i])
                distance[i, 1] = i
                sort = distance[np.argsort(distance[:, 0])]
            # print(sort)
    
    # Lösche Punkte die Anzahl Nearest Neighbor überschreitet
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
        # print(J_relevant)
        # print(NN)
        # print(index_NN)                
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
        j = 0  # Setze j auf den Index des zu betrachtenden äußeren Punktes
        for i in range(len(I_all)):
            diff[i] = I_all[i] - J_relevant[j]
            distance[i, 0] = LA.norm(diff[i])
            distance[i, 1] = i
            sort = distance[np.argsort(distance[:, 0])]
    
    # Lösche Punkte die Anzahl Nearest Neighbor überschreitet
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
                    
    # Plot der nearest neighbors für einen Punkt äußeren Punkt j
        plt.scatter(I_all[:, 0], I_all[:, 1], c='mediumblue', s=50, lw=0)
        plt.scatter(J_relevant[:, 0], J_relevant[:, 1], c='goldenrod', s=50, lw=0)
        plt.scatter(J_relevant[j, 0], J_relevant[j, 1], c='cyan', s=50, lw=0)
        plt.scatter(NN[:, 0], NN[:, 1], c='limegreen', s=50, lw=0)
        plt.axis('equal')
        plt.show()
            
    # Monome definieren und an allen Gitterpunkten auswerten
    size_monomials = np.int((degree + 1) * (degree + 2) / 2)
    eval_monomials = np.zeros((size_monomials, gridStorage.getSize()))
    k = 0
    for i in range(degree + 1):
        for j in range (degree + 1):
            if i + j <= degree:
                eval_monomials[k] = (pow(x[:, 0], i) * pow(x[:, 1], j))
                k = k + 1    
    eval_monomials = np.transpose(eval_monomials)
    
    # Löse LGS und erhalte coeffs
    coeffs = np.linalg.solve(A, eval_monomials)
    
    # Test ob Lösen des LGS erfolgreich war
    # error = eval_monomials - np.matmul(A, coeffs)
    # error = LA.norm(error)
    # if error > pow(10, -14):
    #     print('failed. error > 10e-14')
        
    # Definiere Koeffizientenmatrix mit Koeffizienten der nearest neighbor Punkte zum jeweiligen äußeren Punkt
    coeffs_J_relevant = np.zeros((size_monomials))
    extension_coeffs = np.zeros((n_neighbors, len(J_relevant))) 
    
    for j in range(len(J_relevant)):
        # in Zeile i stehen die coeffs des i-ten inneren NN von einem(!) äußerem Punkt j
        coeffs_inner_NN = np.zeros((n_neighbors, size_monomials))
        for i in range(len(coeffs_inner_NN)):
            coeffs_inner_NN[i] = coeffs[int(index_NN[i, j])]
        # in Spalte j stehen die coeffs des j-ten inneren NN von einem(!) äußeren Punkt i    
        coeffs_inner_NN = np.transpose(coeffs_inner_NN)
        coeffs_J_relevant = coeffs[int(index_J_relevant[j])]
        solution = np.linalg.lstsq(coeffs_inner_NN, coeffs_J_relevant)  # löse mit least squares
        # in Spalte j stehen die Extensionkoeffs für äußeren Punkt j
        extension_coeffs[:, j] = solution[0]
    # print(extension_coeffs)
    # print(extension_coeffs.shape)
    
    # print(x)
    
    # WEB Splines 
    # J_I=np.zeros((index_NN.shape[0],len(I_all)))#x
    # for i in range(len(I_all)):#index_x
    #    for j in range((index_NN.shape[1])):
    #        for k in range((index_NN.shape[0])):
    #            if index_x[i] == index_NN[k,j]:
    #                #print(i,index_J_relevant[j])
    #                J_I[j,i]=index_J_relevant[j]
    
    # for i in range(len(I_all)):
    #    bi = basis.eval(int(lvl[int(index_I_all[i]),0]), int(ind[int(index_I_all[i]),0]), x[int(index_I_all[i]),0])*basis.eval(int(lvl[int(index_I_all[i]),1]), int(ind[int(index_I_all[i]),1]), x[int(index_I_all[i]),1])   
    #    #print(bi)
    #    sum=0
    #    for j in range(index_NN.shape[1]):
    #        for k in range(index_NN.shape[0]):
    #            if index_NN[k,j] == index_I_all[i] and J_I[j,i] != 0: 
    #                #print(k,j)
    #                sum = sum + extension_coeffs[k,j]*basis.eval(int(lvl[int(J_I[j,i]),0]), int(ind[int(J_I[j,i]),0]), x[int(index_I_all[i]),0])#*basis.eval(int(lvl[int(J_I[j,i]),1]), int(ind[int(J_I[j,i]),1]), x[int(index_I_all[i]),1])
    #                #print(extension_coeffs[k,j])
    #                #print(sum)
    #    extended_Bspline = bi+sum 
    #    WEBspline = weightfunction.circle(radius,x[int(index_I_all[i])]) * extended_Bspline
    #    #printLine()
    #    print(WEBspline)
    
    # HIER RICHTIG!!!
    # WEB-Splines
    # J_I=np.zeros((index_NN.shape[0],len(I_all)))#x
    # for i in range(len(I_all)):#index_x
    #    for j in range((index_NN.shape[1])):
    #        for k in range((index_NN.shape[0])):
    #            if index_x[i] == index_NN[k,j]:
    #                #print(i,index_J_relevant[j])
    #                J_I[j,i]=index_J_relevant[j]
    
    # Matrix A mit WEB Splines füllen
    # A_WEB = np.zeros((len(I_all), len(I_all)))    
    # for i in range(len(I_all)):
    #    for l in range(len(I_all)):
    #        bi = basis.eval(int(lvl[int(index_I_all[i]),0]), int(ind[int(index_I_all[i]),0]), x[int(index_I_all[l]),0])*basis.eval(int(lvl[int(index_I_all[i]),1]), int(ind[int(index_I_all[i]),1]), x[int(index_I_all[l]),1])   
    #        #print(bi)
    #        sum=0
    #        for j in range(index_NN.shape[1]):
    #            for k in range(index_NN.shape[0]):
    #                if index_NN[k,j] == index_I_all[i] and J_I[j,i] != 0: 
    #                    #print(k,j)
    #                    sum = sum + extension_coeffs[k,j]*basis.eval(int(lvl[int(J_I[j,i]),0]), int(ind[int(J_I[j,i]),0]), x[int(index_I_all[l]),0])*basis.eval(int(lvl[int(J_I[j,i]),1]), int(ind[int(J_I[j,i]),1]), x[int(index_I_all[l]),1])
    #                    #print(extension_coeffs[k,j])
    #                    #print(sum)
    #        extended_Bspline = bi+sum 
    #        WEBspline = weightfunction.circle(radius,x[int(index_I_all[l])]) * extended_Bspline
    #        #printLine()
    #        #print(WEBspline)
    #        A_WEB[l,i] = WEBspline
    
    # print(A_WEB)
    # print(I_all.shape)
    # print(J_relevant.shape)
    # print(J_all.shape) 
    # print(NN.shape)
    # print(A_WEB.shape)
    # print(x.shape)
    # print(index_J_relevant.shape)
    
    # Matrix A_WEB mit WEB Splines ausgewertet an inneren Punkten füllen: a_l,i = WEBspline_i(x_l) für alle i,l in innere Punkte I
    A_WEB = np.zeros((len(I_all), len(I_all)))  
    for i in range(len(I_all)):
        k = 0
        for l in range(len(x)):
            if eval_circle[l] > 0:
                A_WEB[k, i] = WEBspline(x, i, l)
                k = k + 1
    # print(A_WEB)
    
    # Zielfunktion auswerten an inneren Punkten 
    ev_f = np.zeros((len(I_all), 1))
    for i in range(len(I_all)):
        ev_f[i] = function(x[int(index_I_all[i])])
    # print(ev_f) 
    
    # LGS lösen für Interpolationskoeffizient alpha
    alpha = np.linalg.solve(A_WEB, ev_f)
    # print(alpha)
    
    
    
    err = 0
    for i in range(len(p)):
        f = np.sin(8 * p[i,0]) + np.sin(7 * p[i,1])
        f = f * weightfunction.circle(radius, p[i])
        f_tilde = 0
        for j in range(len(I_all)):      
            f_tilde = f_tilde + alpha[j] * WEBspline(p, j, i)
        err = err + (f - f_tilde) ** 2
    err = err ** (1 / 2)
    print('error : {}'.format(err[0]))  

    error[v-4]= err
    Level[v-4]= v

plt.show()
plt.scatter(Level,error, c='mediumblue', s=50, lw=0)      
plt.plot(Level,error,c='mediumblue')
plt.show()    
    
    
    
    # counter = 0 
    # while counter < 10:
    #    z = np.random.rand(1,2)
    # #    print(z)
    # #    print(z[0])
    #    if weightfunction.circle(radius,z[0]) > 0:
    #        err = err + (function(z[0])-function_tilde(z))**2
    #        counter = counter + 1
    # err = err**(1/2)
    # print(err)
