import numpy as np
import weightfunction
import Bspline
import NearestNeighbors




#Definieren der Testfunktionen

# Testfunktion sin(8x)+sin(7y) auf dem Kreis
def function_1(x):
    f = 0
    f = np.sin(8* x[0]) + np.sin(7 * x[1])
    f = f * weightfunction.circle(radius, x)
    return f

# Testfunktion sin(pi*x*y) auf dem Kreis
def function_2(x):
    f = 0
    f = np.sin(np.pi*x[0]*x[1])
    f = f * weightfunction.circle(radius,x)
    return f

# Testfunktion sin(8x)+sin(7y) auf der Ellipse
def function_1_ellipse(x):
    f = 0
    f = np.sin(8* x[0]) + np.sin(7 * x[1])
    f = f * weightfunction.ellipse(radius1, radius2, x)
    return f

# Testfunktion sin(pi*x*y) auf der Ellipse
def function_2_ellipse(x):
    f = 0
    f = np.sin(np.pi*x[0]*x[1])
    f = f * weightfunction.ellipse(radius1,radius2,x)
    return f


# Interpolationsroutine
def interpolation(level_x,level_y, dim, degree):              
 
    # Bei Bedarf kann hier das Gebiet berechnet und als Bild betrachtet werden.    
    #
    # # Gitter fuer Kreis erzeugen und auswerten
    # x0 = np.linspace(0, 1, 50)
    # X = np.meshgrid(x0, x0) 
    # Z = weightfunction.circle(radius, X)
    #   
    # # Plot von Kreis
    # plt.contour(X[0], X[1], Z, 0)
    # plt.axis('equal')
    # plt.show()
    
    
    # Gitterweite
    h_x = 2**(-level_x)
    h_y = 2**(-level_y)
                
    # Not a Knot Knotenfolge
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
    
    # Index (i_1,i_2) der Bsplines auf dem Gitter
    k=0
    index_all_Bsplines = np.zeros((len(index_Bspline_x)*len(index_Bspline_y),dim))
    for i in index_Bspline_x:
        for j in index_Bspline_y:
            index_all_Bsplines[k] = [i,j]
            k=k+1

        
    # Index (i_1,i_2) der Bsplines mit Knotenmittelpunkt im inneren des Gebiets
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
        
    # Pruefe ob genug innere Bsplines vorhanden sind 
    if len(index_inner_Bsplines) < (degree+1)**2:
        print('Nicht genug innere Punkte. Erhoehe Level oder Gebiet.')   
        quit() 
        
    # Definiere Bsplinemittelpunkte als Vektor
    k=0
    midpoints = np.zeros((len(index_Bspline_x)*len(index_Bspline_y), dim))
    for i in index_Bspline_x:
        for j in index_Bspline_y:
            midpoints[k] = [xi[int(i+degree)], yi[int(j+degree)]]
            k=k+1
        
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
    
    # Bei Bedarf Ausgabe der Parameter:   
#     print("dimensionality:           {}".format(dim))
#     print("level:                    {}".format((level_x, level_y)))
#     print("number of Bsplines:       {}".format(len(index_Bspline_x)*len(index_Bspline_y)))
    
    # Bestimme Index der aeusseren relevanten B-Splines    
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
    
    # Festlegen der relevanten aeusseren Bsplines durch Mittelpunkte   
    J_relevant = np.zeros((len(index_outer_relevant_Bsplines), dim))
    k=0
    for j in index_outer_relevant_Bsplines:
        J_relevant[k] = [xi[int(j[0]+degree)], yi[int(j[1]+degree)]]
        k=k+1
        
    # Bei Bedarf koennen hier die inneren, aeusseren und aeusseren relevanten Punkte als Bild betrachet werden.     
    # Beachte, dass dafuer die Berechnung des Gebietes nicht auskommentiert sein darf 
    #
    # plt.contour(X[0], X[1], Z, 0)
    # plt.scatter(J_all[:,0], J_all[:,1], c='crimson', s=50, lw=0)
    # plt.scatter(I_all[:,0], I_all[:,1], c='mediumblue', s=50, lw=0)
    # plt.scatter(J_relevant[:, 0], J_relevant[:, 1], c='goldenrod', s=50, lw=0)
    # plt.show()
    
    
        
    # Index der inneren Bsplines unter Gesamtanzahl (n+1)**2
    index_I_all = np.zeros(len(I_all))
    for i in range(len(I_all)):
        for j in range(len(midpoints)):
            if I_all[i, 0] == midpoints[j, 0] and I_all[i, 1] == midpoints[j, 1]:
                index_I_all[i] = j
        
    # Index der aeusseren Bsplines unter Gesamtanzahl (n+1)**2
    index_J_all = np.zeros(len(J_all))
    for i in range(len(J_all)):
        for j in range(len(midpoints)):
            if J_all[i, 0] == midpoints[j, 0] and J_all[i, 1] == midpoints[j, 1]:
                index_J_all[i] = j
                
    # Index der relevanten aeusseren Bsplines unter Gesamtanzahl (n+1)**2
    index_J_relevant = np.zeros(len(J_relevant))
    for i in range(len(J_relevant)):
        for j in range(len(midpoints)):
            if J_relevant[i, 0] == midpoints[j, 0] and J_relevant[i, 1] == midpoints[j, 1]:
                index_J_relevant[i] = j
        
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
         
    # Aufstellen der Interpolationsmatrix A_ij = b_j(x_i)
    A = np.zeros((len(index_Bspline_x)*len(index_Bspline_y), len(gp)))
    for l in range(len(gp)):
        k=0
        for i in index_Bspline_x:
            for j in index_Bspline_y:
                A[l,k] = Bspline.evalBspline(degree, i, xi, gp[l,0]) * Bspline.evalBspline(degree, j, yi, gp[l,1])
                k=k+1        
    
    # Loesen des LGS A*coeffs = eval_monomials fuer Interpolationskoeffizienten
    coeffs = np.linalg.solve(A, eval_monomials)

    # Festlegen von I(j)
    k=1
    if k == 0:
        # Nearest Neighbors nach Abstand
        distance = np.zeros((len(I_all), dim))
        NN = np.zeros((len(J_relevant), n_neighbors, dim))
        for j in range(len(J_relevant)):
            for i in range(len(I_all)):
                diff = I_all[i] - J_relevant[j]
                distance[i, 0] = np.linalg.norm(diff)
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
        # Nearest Neighbors mit naehestem (n+1)x(n+1) Array
        distance = np.zeros((len(I_all), dim))
        NN = np.zeros((len(J_relevant), n_neighbors, dim))
        for j in range(len(J_relevant)):
            for i in range(len(I_all)):
                diff = I_all[i] - J_relevant[j]
                distance[i, 0] = np.linalg.norm(diff)
                distance[i, 1] = i
                sort = distance[np.argsort(distance[:, 0])]
            NearestNeighbors.NNsearch(degree, sort, j, 0, I_all, h_x, h_y, NN, radius)
        NN=NN
                  
        index_NN = np.zeros((len(J_relevant), n_neighbors))
        for j in range(NN.shape[0]):
            for i in range(NN.shape[1]):
                for k in range(len(gp)):
                    if NN[j, i, 0] == gp[k,0] and NN[j, i, 1] == gp[k,1]:
                        index_NN[j, i] = k

                        
    # Bei Bedarf kann hier I(j) fuer alle j in den aeusseren Indizes betrachtet werden. 
    # Beachte, dass dafuer die Berechnung des Gebietes nicht auskommentiert sein darf  
    #     
    # for i in range(len(J_relevant)):
    #     plt.scatter(J_all[:,0], J_all[:,1], c='crimson', s=50, lw=0)
    #     plt.scatter(I_all[:, 0], I_all[:, 1], c='mediumblue', s=50, lw=0)
    #     plt.scatter(J_relevant[:, 0], J_relevant[:, 1], c='goldenrod', s=50, lw=0)
    #     plt.scatter(J_relevant[i, 0], J_relevant[i, 1], c='cyan', s=50, lw=0) 
    #     plt.scatter(NN[i,:,0], NN[i, :, 1], c='limegreen', s=50, lw=0)
    #     plt.contour(X[0], X[1], Z, 0)
    #     plt.show()
    
    
         
    # Definiere Koeffizientenmatrix der aeusseren relevanten Indizes j
    coeffs_J_relevant = np.zeros((len(J_relevant),1, size_monomials))
    k=0
    for i in index_J_relevant: 
        coeffs_J_relevant[k] = coeffs[int(i)]
        k=k+1
    coeffs_J_relevant = np.transpose(coeffs_J_relevant, [0,2,1])          
          
    # Definiere Koeffizientenmatrix der I(j)
    coeffs_NN = np.zeros((len(J_relevant),n_neighbors, size_monomials))
    for i in range(len(index_NN)):
        k=0
        for j in index_NN[i]:
            coeffs_NN[i,k] = coeffs[int(j)]
            k=k+1
    coeffs_NN = np.transpose(coeffs_NN, [0,2,1])     
    
         
    # Ueberpruefe ob Determinante der Koeffizientenmatrix der I(j) ungleich 0
    if (np.linalg.det(coeffs_NN) == 0).any():
        print('Waehle I(j) so, dass Koeffizientenmatrix der I(j) nicht singulaer')
        quit()
         
    # Loesen des LGS E=C^(-1)D um Erweiterungskoeffizienten zu erhalten      
    extension_coeffs = np.zeros((len(J_relevant), size_monomials, 1))
    for i in range(coeffs_NN.shape[0]):
        extension_coeffs[i] = np.linalg.solve(coeffs_NN[i], coeffs_J_relevant[i])
    
    
    # Definiere hierarch. WEB-Splines und stelle Matrix A_WEB auf
    A_WEB = np.zeros((len(I_all),len(I_all)))
    for p in range(len(I_all)):  
        # Definiere J(i)
        extended_Bspline = 0
        c=0        
        for i in index_I_all: 
            J_i = np.zeros(1)
            index_NN_relevant = np.zeros(1)
            bi = Bspline.evalBspline(degree, index_all_Bsplines[int(i),0], xi, I_all[p,0])*Bspline.evalBspline(degree, index_all_Bsplines[int(i),1], yi, I_all[p,1])
            for k in range(len(index_NN)):
                if (i == index_NN[k]).any():
                    J_i = np.hstack((J_i, index_J_relevant[k]))                                 
                    for l in range(index_NN.shape[1]):
                        if i == index_NN[k,l]:
                            index_NN_relevant = np.hstack((index_NN_relevant, l))
            J_i = np.delete(J_i, 0)
            index_NN_relevant = np.delete(index_NN_relevant, 0)

            g=0
            inner_sum = 0
            for j in J_i:
                for t in range(len(index_J_relevant)):
                    if j == index_J_relevant[t]:
                        inner_sum = inner_sum + extension_coeffs[t, int(index_NN_relevant[g])]*Bspline.evalBspline(degree, index_all_Bsplines[int(j),0], xi, I_all[p,0])*Bspline.evalBspline(degree, index_all_Bsplines[int(j),1], yi, I_all[p,1]) 
                        g=g+1
            extended_Bspline = extended_Bspline + ( bi + inner_sum)
            A_WEB[p,c] = weightfunction.circle(radius, I_all[p])*extended_Bspline
            c=c+1          
               
    # Auswerten der Testfunktion an den inneren Punkten           
    b = np.zeros((len(I_all),1))
    for i in range(len(I_all)):
        b[i] = function_1(gp[int(index_I_all[i])])
    
    # Interpolationskoeffizieten fuer hierar. WEB-Spline Interpolanten       
    alpha = np.linalg.solve(A_WEB, b)
    
    # Interpolation an number_intepolation_points vielen Punkten
    eval_Interpolation = np.zeros((len(interpolation_points),1))  
    for p in range(len(interpolation_points)):  
        # Definiere J(i)
        extended_Bspline = 0
        c=0
        f_tilde = 0        
        for i in index_I_all: 
            J_i = np.zeros(1)
            index_NN_relevant = np.zeros(1)
            bi = Bspline.evalBspline(degree, index_all_Bsplines[int(i),0], xi, interpolation_points[p,0])*Bspline.evalBspline(degree, index_all_Bsplines[int(i),1], yi, interpolation_points[p,1])
            for k in range(len(index_NN)): 
                if (i == index_NN[k]).any():
                    J_i = np.hstack((J_i, index_J_relevant[k]))
                                           
                    for l in range(index_NN.shape[1]):
                        if i == index_NN[k,l]:
                            index_NN_relevant = np.hstack((index_NN_relevant, l))
            J_i = np.delete(J_i, 0)
            index_NN_relevant = np.delete(index_NN_relevant, 0)
            g=0
            inner_sum = 0
            for j in J_i:
                for t in range(len(index_J_relevant)):
                    if j == index_J_relevant[t]:
                        inner_sum = inner_sum + extension_coeffs[t, int(index_NN_relevant[g])]*Bspline.evalBspline(degree, index_all_Bsplines[int(j),0], xi, interpolation_points[p,0])*Bspline.evalBspline(degree, index_all_Bsplines[int(j),1], yi, interpolation_points[p,1]) 
                        g=g+1
                                   
            extended_Bspline = extended_Bspline + ( bi + inner_sum)
            webspline = weightfunction.circle(radius, interpolation_points[p])*extended_Bspline
                   
            f_tilde = f_tilde + alpha[c]*webspline
            c=c+1 
        eval_Interpolation[p] = f_tilde     
    return eval_Interpolation 
                      


#Es wird der Fehler von einem SG zum hierarch. WEB-Spline Interpolaten berechnet. 
#Die Parameter die veraendert werden koennen sind der Grad der B-Splines, das Level des Sparse Grids, die Anzahl der Interpolationspunkte, 
# der Radius und die Gewichtsfunktion. Letztere muss manuell in den Routinen geandert werden.
#Abhaengig vom SG Level muss das Startlevel der vollen Gitter fuer die Kombinationstechnik eingegeben werden.



#Parameter festlegen
dim = 2         # Dimension (funktioniert aktuell nur fuer Grad 2)
degree = 1      # Grad der B-Splines (nur ungerade!)
SGlevel = 3     # Level des Sparse Grids
number_interpolation_points = 5000   # Anzahl der Interpolationspunkte

#Festlegen des Startlevels des kleinsten vollen Gitters fuer die Kombinationstechnik
startlevel = 2      # Minimales Level in x-Richtung
level_y = startlevel
level_x = startlevel   

#Parameter fuer weightfunction_circle
radius = 0.4    # Radius des Kreises

#Parameter fuer weightfunction_ellipse
radius1 = 0.45  # Radius der Ellipse
radius2 = 0.1   # Radius des ausgeschnittenen Kreises


# Beliebige Interpolationspunkte im Gebiet festlegen
interpolation_points = np.zeros((number_interpolation_points, dim))
counter = 0 
while counter < number_interpolation_points:
    z = np.random.rand(1,dim)
    if weightfunction.circle(radius, z[0]) > 0:
        interpolation_points[counter] = z[0]
        counter = counter + 1
        
# Pruefe ob Level des vollen Gitters hoch genug fuer NAK Bedingung
if level_x and level_y < np.log2(degree+1):
    print('Error: Level zu niedrig. Es muss Level >= log2(degree+1) sein ')
    quit()  



# L2 Fehler wird zunachst 0 gesetzt
Error = np.zeros((number_interpolation_points,1))
# Berechnung des Fehlers fuer die einzelnen Gitter im Kombinationsschema fuer SG 
level_y_max = SGlevel+1
for level_x in range(startlevel,SGlevel+1):
    for level_y  in range(startlevel,level_y_max):
        if level_x+level_y == SGlevel+dim: # Gitter die aufaddiert werden
            print(level_x,level_y) # Print aktuelles Level fur bessere Uebersicht welches volle Gitter gerade berechnet wird
            Error = Error + interpolation(level_x, level_y, dim, degree) # aufaddieren des Fehlers auf Gesamtfehler
        elif level_x+level_y == SGlevel+dim-1: # Gitter die subtrahiert werden
            print(level_x,level_y) # Print aktuelles Level fur bessere Uebersicht welches volle Gitter gerade berechnet wird
            Error = Error - interpolation(level_x, level_y, dim, degree) # subtrahieren des Fehlers von Gesamtfehler
            
 
# Auswerten der Testfunktion an Interpolationspunkten    
eval_function = np.zeros((len(interpolation_points),1))
for i in range(len(interpolation_points)):
    eval_function[i,0] = function_1(interpolation_points[i]) 
  
  
# Bestimmung des L2 Fehlers zwischen Testfunktion und interpolierter Funktion
L2error = np.linalg.norm(eval_function-Error)
print(L2error) # Ausgabe des L2-Fehlers

            
            
            
            
            
            
            
            
            
            

