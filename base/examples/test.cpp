// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include <sgpp/base/datatypes/DataVector.hpp>
#include <sgpp/base/grid/Grid.hpp>
#include <sgpp/base/grid/GridStorage.hpp>
#include <sgpp/base/grid/generation/GridGenerator.hpp>
#include <sgpp/base/operation/BaseOpFactory.hpp>
#include <sgpp/base/operation/hash/OperationEval.hpp>

#include <sgpp_base.hpp>
#include <sgpp_optimization.hpp>

#include <algorithm>
#include <iostream>
#include <iterator>


void printLine() {
  std::cout << "----------------------------------------"
               "----------------------------------------\n";
}

/**
 * Before starting with the <tt>main</tt> function,
 * the function \f$f\f$, which we want to interpolate, is defined.
 */
double f(double x0) { return pow(x0, 2); }

int main() {
  /**
   * First, we create a two-dimensional grid (type sgpp::base::Grid)
   * with piecewise bilinear basis functions with the help of the factory method
   * sgpp::base::Grid.createWEBsplineGrid().
   */
  size_t dim = 1;
  size_t degree = 2;
  // sgpp::base::WEBsplineGrid grid(dim, degree);
  std::unique_ptr<sgpp::base::Grid> grid(sgpp::base::Grid::createWEBsplineGrid(dim, degree));
  // ist pointer


  /**
   * Then we obtain a reference to the grid's
   * sgpp::base::GridStorage object which allows us, e.g., to access grid
   * points, to obtain the dimensionality (which we print) and the
   * number of grid points.
   */
  sgpp::base::GridStorage& gridStorage = grid->getStorage();  // grid.getStorage wenn kein pointer
  std::cout << "dimensionality:         " << gridStorage.getDimension() << std::endl;

  /**
   * Now, we use a sgpp::base::GridGenerator to
   * create a regular sparse grid of level 3.
   * Thus, \c gridStorage.getSize() returns 17, the number of grid points
   * of a two-dimensional regular sparse grid of level 3.
   */
  size_t level = 3;
  grid->getGenerator().regular(level);  // grid.getGenerator().regual(level); wenn kein pointer
  std::cout << "number of grid points:  " << gridStorage.getSize() << std::endl;

  //-----------------------------------------------------------------------------------------------

  /**
    * We create an object of type sgpp::base::DataVector
    * which is essentially a wrapper around a \c double array.
    * The \c DataVector is initialized with as many
    * entries as there are grid points. It serves as a coefficient vector for the
    * sparse grid interpolant we want to construct. As the entries of a
    * freshly created \c DataVector are not initialized, we set them to
    * 0.0. (This is superfluous here as we initialize them in the
    * next few lines anyway.)
    */
  sgpp::base::DataVector alpha(gridStorage.getSize());
  alpha.setAll(0.0);
  std::cout << "length of alpha vector: " << alpha.getSize() << std::endl;

   /**
    * The \c for loop iterates over all grid points: For each grid
    * point \c gp, the corresponding coefficient \f$\alpha_j\f$ is set to the
    * function value at the grid point's coordinates which are obtained by
    * \c getStandardCoordinate(dim).
    * The current coefficient vector is then printed.
    */
  /**
   * Diese schleife anstelle von sgpp::base::DataVector functionValues(gridGen.getFunctionValues());
   * in optimization.cpp
   */
  for (size_t i = 0; i < gridStorage.getSize(); i++) {
      sgpp::base::GridPoint& gp = gridStorage.getPoint(i);
      alpha[i] = f(gp.getStandardCoordinate(0));
  }
  std::cout << "alpha before hierarchization: " << alpha.toString() << std::endl;

  /**
     * Then, we hierarchize the function values to get hierarchical B-spline
     * coefficients of the B-spline sparse grid interpolant
     * \f$\tilde{f}\colon [0, 1]^d \to \mathbb{R}\f$.
     */
    printLine();
    std::cout << "Hierarchizing...\n\n";
    sgpp::base::DataVector coeffs(alpha.getSize());
    std::cout << "coeffs: " << coeffs.toString() << std::endl;

//    sgpp::optimization::HierarchisationSLE hierSLE(*grid);
//    sgpp::optimization::sle_solver::Auto sleSolver;
//
//
//     // solve linear system, lösen von gleichungssystem. ausgabe hinzufügen
//      if (!sleSolver.solve(hierSLE, alpha, coeffs)) {
//         std::cout << "Solving failed, exiting.\n";
//          return 1;
//     }
//
//     sgpp::optimization::InterpolantScalarFunction ft(*grid, coeffs);  // function erstellen
}
