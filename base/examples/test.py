#import faulthandler; faulthandler.enable()
import pysgpp
import math
import sys
import numpy as np
import matplotlib as plt
from numpy import linalg as LA


def printLine():
    print("--------------------------------------------------------------------------------------")
    

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
alpha = pysgpp.DataVector(gridStorage.getSize(),0.0)
print("length of alpha vector: {}".format(len(alpha)))

## The \c for loop iterates over all grid points: For each grid
## point \c gp, the corresponding coefficient \f$\alpha_j\f$ is set to the
## function value at the grid point's coordinates which are obtained by
## \c getStandardCoordinate(dim).
## The current coefficient vector is then printed.
printLine()
for i in range(gridStorage.getSize()):
  gp = gridStorage.getPoint(i)
  alpha[i] = f(gp.getStandardCoordinate(0))

print("alpha: {}".format(alpha))
#print("gp = {}".format(gp.getStandardCoordinate(0)))

## Define the coefficient vector
printLine()
coeffs = pysgpp.DataVector(alpha.getSize(),0.0)
#print("coeffs before SLE: {}".format(coeffs))

# Define and solve SLE
hierSLE = pysgpp.OptHierarchisationSLE(grid)
sleSolver = pysgpp.OptAutoSLESolver()
if not sleSolver.solve(hierSLE, alpha, coeffs):
    print("Solving failed, exiting.")
    sys.exit(1)

# Result of SLE 
print("coeffs solved: {}".format(coeffs))

printLine()
# Interpolated function
ft = pysgpp.OptInterpolantScalarFunction(grid, coeffs)

# Define gridpoints as vector for evaluation
# Gitterpunkte sind 1/2; 1/4, 3/4; 1/8, 3/8, 5/8, 7/8 (erstes;zweites;drittes Gritter)
#Auswertungspunkte sind Gitterpunkte
x0 = pysgpp.DataVector(alpha.getSize(),0.0)
for i in range(gridStorage.getSize()):
  gp = gridStorage.getPoint(i)
  x0[i] = gp.getStandardCoordinate(0) #Auswertungspunkte
    
print("Gridpoints: {}".format(x0))

printLine()
# Eval interpolated function at gridpoints \gp
beta = pysgpp.DataVector(alpha.getSize(),0.0)
evalPoint = pysgpp.DataVector(1,0.0)
for i in range(gridStorage.getSize()):
    evalPoint[0] = x0[i]
    beta[i] = ft.eval(evalPoint) #auswerten an Vektor!! nicht an Punkt
print("beta = {}".format(beta))
print("alpha = {}".format(alpha))

printLine()
# Compare results of interpolated function with exact function

#print(np.array_equal(alpha, beta)) Error: Segmentation fault ???

printLine()
# Estimate at (random) points
a = pysgpp.DataVector(np.linspace(0, 1, 7))
a0 = a
print("length of a vector: {}".format(len(a0)))

# Estimate f(a0)
for i in range(len(a0)):
    a[i] = f(a0[i])  

# Define coefficient vector \c and solve SLE
c = pysgpp.DataVector(a.getSize(),0.0)
hierSLE = pysgpp.OptHierarchisationSLE(grid)
sleSolver = pysgpp.OptAutoSLESolver()
if not sleSolver.solve(hierSLE, a, c):
    print("Solving failed, exiting.")
    sys.exit(1)

# Result of SLE 
print("c solved: {}".format(c))

printLine()
# Interpolated function
ft_new = pysgpp.OptInterpolantScalarFunction(grid, c)

# Define gridpoints as vector for evaluation
# Gitterpunkte sind 1/2; 1/4, 3/4; 1/8, 3/8, 5/8, 7/8 (erstes;zweites;drittes Gritter)
#Auswertungspunkte sind Gitterpunkte
x1 = pysgpp.DataVector(a.getSize(),0.0)
for i in range(gridStorage.getSize()):
  x1[i] = a0[i] #Auswertungspunkte
   
print("Interpolationpoints: {}".format(x1))

printLine()
# Eval interpolated function at gridpoints \gp
b = pysgpp.DataVector(a.getSize(),0.0)
evalPoint = pysgpp.DataVector(1,0.0)
for i in range(gridStorage.getSize()):
    evalPoint[0] = x1[i]
    b[i] = ft_new.eval(evalPoint) #auswerten an Vektor!! nicht an Punkt
print("b = {}".format(b))
print("a = {}".format(a))

# calculate difference a-b
diff = pysgpp.DataVector(a.getSize(),0.0)
for i in range(len(a)):
    diff[i] = a[i]-b[i]
    
l2 = [3, 4]
print(l2)
print(LA.norm(l2, 2))
#print(diff)


# L2 Norm of difference
error = LA.norm(diff, 2)


