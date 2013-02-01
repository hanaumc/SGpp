/******************************************************************************
* Copyright (C) 2009 Technische Universitaet Muenchen                         *
* This file is part of the SG++ project. For conditions of distribution and   *
* use, please see the copyright notice at http://www5.in.tum.de/SGpp          *
******************************************************************************/
// @author A. Mo-Hellenbrand

#ifndef OPERATIONDENSITYMARGTO1D_HPP_
#define OPERATIONDENSITYMARGTO1D_HPP_

#include "base/grid/Grid.hpp"

namespace sg
{
namespace datadriven
{

  /**
   * Marginalize Probability Density Function
   */

  class OperationDensityMargTo1D
  {
    public:
      OperationDensityMargTo1D() {}
      virtual ~OperationDensityMargTo1D() {}

	  /**
	   * Keep applying marginalizes to (Density) Functions, until it's reduced to 1 dimension (dim_x)
	   *
	   * @param alpha Coefficient vector for current grid
	   * @param grid_x Referenz of grid pointer
	   * @param alpha_x Coefficient vector for new grid (grid_x). Will be resized.
	   * @param dim_x Target dimension number, all dimensions higher and lower than dim_x will be removed
	   * @param dim_h Highest dimension number of grid
	   */
      virtual void margToDimX(base::DataVector* alpha, base::Grid* &grid_x, base::DataVector* &alpha_x, unsigned int dim_x, unsigned int dim_h) = 0;
  };

}
}
#endif /* OPERATIONDENSITYMARGTO1D_HPP_ */
