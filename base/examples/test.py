
import pysgpp
import math
import sys
import numpy as np
import matplotlib as plt


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

print("alpha before hierarchization: {}".format(alpha))

coeffs = pysgpp.DataVector(len(alpha))
print("coeffs: {}".format(coeffs))

#hierSLE = pysgpp.OptHierarchisationSLE(grid)
#sleSolver = pysgpp.OptAutoSLESolver()







