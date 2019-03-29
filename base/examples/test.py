
import pysgpp
import math
import sys
import numpy as np
import matplotlib as plt
from array import array


## Before starting, the function \f$f\f$, which we want to interpolate, is defined.
f = lambda x0: pow(x0,2)

## First, we create a two-dimensional grid (type sgpp::base::Grid)
## with piecewise bilinear basis functions with the help of the factory method
## sgpp::base::Grid.createLinearGrid().
dim = 1 
degree = 2
grid = pysgpp.Grid.createWEBsplineGrid(dim, degree)

## Then we obtain the grid's
## sgpp::base::GridStorage object which allows us, e.g., to access grid
## points, to obtain the dimensionality (which we print) and the
## number of grid points.
gridStorage = grid.getStorage()
print("dimensionality:         {}".format(gridStorage.getDimension()))

## Now, we use a sgpp::base::GridGenerator to
## create a regular sparse grid of level 3.
## Thus, \c gridStorage.getSize() returns 17, the number of grid points
## of a two-dimensional regular sparse grid of level 3.
level = 3
grid.getGenerator().regular(level)
print("number of grid points:  {}".format(gridStorage.getSize()))

## We create an object of type sgpp::base::DataVector
## which is essentially a wrapper around a \c double array.
## The \c DataVector is initialized with as many
## entries as there are grid points. It serves as a coefficient vector for the
## sparse grid interpolant we want to construct. As the entries of a
## freshly created \c DataVector are not initialized, we set them to
## 0.0. (This is superfluous here as we initialize them in the
## next few lines anyway.)
alpha = pysgpp.DataVector(gridStorage.getSize())
alpha.setAll(0.0)
print("length of alpha vector: {}".format(len(alpha)))
#print(alpha)

## The \c for loop iterates over all grid points: For each grid
## point \c gp, the corresponding coefficient \f$\alpha_j\f$ is set to the
## function value at the grid point's coordinates which are obtained by
## \c getStandardCoordinate(dim).
## The current coefficient vector is then printed.
for i in range(gridStorage.getSize()):
  gp = gridStorage.getPoint(i)
  alpha[i] = f(gp.getStandardCoordinate(0))

print("alpha: {}".format(alpha))

coeffs = pysgpp.DataVector(len(alpha))
print("coeffs before SLE: {}".format(coeffs))

hierSLE = pysgpp.OptHierarchisationSLE(grid)
sleSolver = pysgpp.OptAutoSLESolver()

# solve linear system
if not sleSolver.solve(hierSLE, alpha, coeffs):
    print("Solving failed, exiting.")
    sys.exit(1)

# ergebnis
print("coeffs solved: {}".format(coeffs))

#print("alpha: {}".format(alpha))

ft = pysgpp.OptInterpolantScalarFunction(grid, coeffs) #interpolierte funktion

x0 = pysgpp.DataVector(alpha.getSize(),0.0)

# Gitterpunkte sind 1/2; 1/4, 3/4; 1/8, 3/8, 5/8, 7/8 (erstes;zweites;drittes Gritter)
for i in range(gridStorage.getSize()):
  gp = gridStorage.getPoint(i)
  x0[i] = gp.getStandardCoordinate(0) #Auswertungspunkte
    
print("Gitterpunkte: {}".format(x0))

evalPoint = pysgpp.DataVector(1,0.0)
evalPoint[0] = x0[0]
print(evalPoint)
ft.eval(evalPoint) #auswerten an Vektor!! nicht an Punkt
print(ft.eval(evalPoint))

#print("ft1={}".format(ft.eval(x0[1])))


#for i in range (0,len(alpha)):
 #   x0[i]=gridStorage.getCoordinates(gridStorage.getPoint(i))
  #  print(x0)
    
#x0 = gridStorage.getCoordinates(gridStorage.getPoint(0));
#print(x0)
#x1 = gridStorage.getCoordinates(gridStorage.getPoint(1));
#print(x1)
#x2 = gridStorage.getCoordinates(gridStorage.getPoint(2));
#print(x2)
#x3 = gridStorage.getCoordinates(gridStorage.getPoint(3));
#print(x3)



#for i in range(0,len(alpha)):
 #   print("alpha: {}".format(alpha[i]))
  #  print(f(coeffs[i]))
    



# print(alpha[0])






