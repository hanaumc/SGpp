/******************************************************************************
* Copyright (C) 2009 Technische Universitaet Muenchen                         *
* This file is part of the SG++ project. For conditions of distribution and   *
* use, please see the copyright notice at http://www5.in.tum.de/SGpp          *
******************************************************************************/
// @author Jörg Blank (blankj@in.tum.de), Alexander Heinecke (Alexander.Heinecke@mytum.de)

#include "basis/modlinear/algorithm_sweep/PhiPhiUpModLinear.hpp"
using namespace sg::base;

namespace sg
{

namespace detail
{

PhiPhiUpModLinear::PhiPhiUpModLinear(GridStorage* storage) : storage(storage)
{
}

PhiPhiUpModLinear::~PhiPhiUpModLinear()
{
}

void PhiPhiUpModLinear::operator()(DataVector& source, DataVector& result, grid_iterator& index, size_t dim)
{
	double fl = 0.0;
	double fr = 0.0;
	rec(source, result, index, dim, fl, fr);
}

void PhiPhiUpModLinear::rec(DataVector& source, DataVector& result, grid_iterator& index, size_t dim, double& fl, double& fr)
{
	size_t seq = index.seq();

	double alpha_value = source[seq];

	GridStorage::index_type::level_type l;
	GridStorage::index_type::index_type i;

	index.get(dim, l, i);

	double h = 1/pow(2.0, static_cast<int>(l));
	double fml = 0.0;
	double fmr = 0.0;

	if(!index.hint())
	{
		index.left_child(dim);
		if(!storage->end(index.seq()))
		{
			rec(source, result, index, dim, fl, fml);
		}

		index.step_right(dim);
		if(!storage->end(index.seq()))
		{
			rec(source, result, index, dim, fmr, fr);
		}

		index.up(dim);
	}

	double fm = fml + fmr;

	// level 1, constant function
	if(l == 1)
	{
		result[seq] = fl + fm + fr;

		fl += fm/2.0 + alpha_value;
		fr += fm/2.0 + alpha_value;
	}
	// left boundary
	else if(i == 1)
	{
		result[seq] = 2.0 * fl + fm;

		fl += fm/2.0 + 4.0/3.0*h*alpha_value;
		fr += fm/2.0 + 2.0/3.0*h*alpha_value;
	}
	// right boundary
	else if(static_cast<int>(i) == static_cast<int>((1 << l)-1))
	{
		result[seq] = 2.0 * fr + fm;

		fl += fm/2.0 + 2.0/3.0*h*alpha_value;
		fr += fm/2.0 + 4.0/3.0*h*alpha_value;
	}
	// inner functions
	else
	{
		result[seq] = fm;

		fl += fm/2.0 + h/2.0*alpha_value;
		fr += fm/2.0 + h/2.0*alpha_value;
	}
}

}

}
