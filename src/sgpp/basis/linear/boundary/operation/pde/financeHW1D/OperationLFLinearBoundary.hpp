/******************************************************************************
* Copyright (C) 2009 Technische Universitaet Muenchen                         *
* This file is part of the SG++ project. For conditions of distribution and   *
* use, please see the copyright notice at http://www5.in.tum.de/SGpp          *
******************************************************************************/
//@author Chao qi(qic@in.tum.de)

#ifndef OPERATIONLFLINEARBOUNDARY_HPP
#define OPERATIONLFLINEARBOUNDARY_HPP

#include "algorithm/pde/StdUpDown.hpp"

namespace sg
{

/**
 * Implements the \f$(x\phi_i(x),d\phi_j(x))\f$ operator on linear boundary grids
 *
 * @version $HEAD$
 */
class OperationLFLinearBoundary: public StdUpDown
{
public:
	/**
	 * Constructor
	 *
	 * @param storage the grid's GridStorage object
	 */
	OperationLFLinearBoundary(GridStorage* storage);

	/**
	 * Destructor
	 */
	virtual ~OperationLFLinearBoundary();

protected:
	/**
	 * Up-step in dimension <i>dim</i> for \f$(x\phi_i(x),d\phi_j(x))\f$.
	 * Applies the up-part of the one-dimensional mass matrix in one dimension.
	 * Computes \f[\int_{x=0}^1  x \phi_i(x) d\phi_j(x) dx.\f]
	 *
	 * @param dim dimension in which to apply the up-part
	 * @param alpha vector of coefficients
	 * @param result vector to store the results in
	 */
	virtual void up(DataVector& alpha, DataVector& result, size_t dim);

	/**
	 * Down-step in dimension <i>dim</i> for \f$(x\phi_i(x),d\phi_j(x))\f$.
	 * Applies the down-part of the one-dimensional mass matrix in one dimension.
	 * Computes \f[\int_{x=0}^1  x \phi_i(x) d\phi_j(x) dx.\f]
	 *
	 * @param dim dimension in which to apply the down-part
	 * @param alpha vector of coefficients
	 * @param result vector to store the results in
	 */
	virtual void down(DataVector& alpha, DataVector& result, size_t dim);
};

}

#endif /* OPERATIONLFLINEARBOUNDARY_HPP */
