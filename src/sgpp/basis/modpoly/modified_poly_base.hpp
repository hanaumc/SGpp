/*****************************************************************************/
/* This file is part of sg++, a program package making use of spatially      */
/* adaptive sparse grids to solve numerical problems                         */
/*                                                                           */
/* Copyright (C) 2008 Jörg Blank (blankj@in.tum.de)                          */
/* Copyright (C) 2009 Alexander Heinecke (Alexander.Heinecke@mytum.de)       */
/*                                                                           */
/* sg++ is free software; you can redistribute it and/or modify              */
/* it under the terms of the GNU General Public License as published by      */
/* the Free Software Foundation; either version 3 of the License, or         */
/* (at your option) any later version.                                       */
/*                                                                           */
/* sg++ is distributed in the hope that it will be useful,                   */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/* GNU General Public License for more details.                              */
/*                                                                           */
/* You should have received a copy of the GNU General Public License         */
/* along with sg++; if not, write to the Free Software                       */
/* Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA */
/* or see <http://www.gnu.org/licenses/>.                                    */
/*****************************************************************************/

#ifndef MODIFIED_POLY_BASE_HPP
#define MODIFIED_POLY_BASE_HPP

#include <cmath>
#include <vector>

#include "exception/factory_exception.hpp"

namespace sg
{

/**
 * Modified polynomial base functions.
 * Special polynomial functions to cover values unequal 0 at the border. Implemented as seen in AWR 2 paper
 * by Prof. Bungartz (http://www5.in.tum.de/wiki/index.php/Algorithmen_des_Wissenschaftlichen_Rechnens_II_-_Winter_08)
 */
template<class LT, class IT>
class modified_poly_base
{
protected:
	double* polynoms;
	size_t degree;

public:
	modified_poly_base(size_t degree) : polynoms(NULL), degree(degree)
	{
		if(degree < 0)
		{
			throw factory_exception("poly_base: degree < 0");
		}

		int polycount = (1 << (degree+1)) - 1;
		std::vector<double> x;

		// degree + 1 for the polynom, +1 for the integral value, +1 for the base-point
		polynoms = new double[(degree + 1 + 2) * polycount];
		initPolynoms(x, 1, 1);
	}

	~modified_poly_base()
	{
		if(polynoms)
		{
			delete [] polynoms;
		}
	}

	double eval(LT level, IT index, double p)
	{
		size_t deg = degree + 1 < level ? degree + 1 : level;

		size_t idMask = (1 << deg) - 1;
		size_t id = (((index & idMask) >> 1) | (1 << (deg - 1))) - 1;

		// scale p to a value in [-1.0,1.0]
		double val = (1 << level)*p - index;
		return evalPolynom(id, deg, val);
	}

private:
	double evalPolynom(size_t id, size_t deg, double val)
	{
		double* x_store = this->polynoms + (degree + 1 + 2) * id;
		double* y_store = x_store + 2;

		double y_val = y_store[deg-1]; // TODO
		// scale val back into the right range
		double x_val = x_store[0] + val * pow(2.0, -(double)(deg));

		//Horner
		for(int i = deg-2; i >= 0; i--)
		{
			y_val = y_val * x_val + y_store[i];
		}

		return y_val;
	}

/**
 * recursively creates polynomial values
 */
	void initPolynoms(std::vector<double>& x, LT level, IT index)
	{
		// Add new point
		x.push_back(index * pow(2.0, -(double)level));

		std::vector<double> y;
		std::vector<double> intpoly;

		for(int i = 0; i < level - 1; i++)
		{
			y.push_back(0.0);
			intpoly.push_back(0.0);
		}
		y.push_back(1.0);
		intpoly.push_back(0.0);

		// Every poly has a unique id similiar to sparse grid level/index pairs
		size_t id = ((index >> 1) | (1 << (level-1))) - 1;

		int n = level;
		std::vector<std::vector<double> > lagpoly;

		/**
		 * Fill lagpoly with multiplied lagrange polynomials
		 * Add lagrange polynomials together into intpoly
		 */
		for(int i = 0; i < n; i++)
		{
			lagpoly.push_back(std::vector<double>());
			lagpoly[i].push_back(1.0);
			double fac = y[i];

			int j = 0;
			for(int k = 0; k < n; k++)
			{
				if(k == i)
				{
					continue;
				}
				lagpoly[i].push_back(lagpoly[i][j]);
				for(int jj = j; jj > 0; jj--)
				{
					lagpoly[i][jj] = lagpoly[i][jj-1] - lagpoly[i][jj]*x[k];
				}
				lagpoly[i][0] *= -x[k];
            	j += 1;
           		fac /= (x[i] - x[k]);
			}

			for(int l = 0; l < n; l++)
			{
				lagpoly[i][l] *= fac;
            	intpoly[l] += lagpoly[i][l];
			}
		}

		//determine position in storage. (degree + 1) polynomial factors and 2 values for integral and x-value
		double* x_store = this->polynoms + (degree + 3) * id;
		double* y_store = x_store + 2;

		// Copy values into storage
		for(int i = 0; i < n; i++)
		{
			y_store[i] = intpoly[i];
		}

		x_store[0] = x.back();


		if((level) < degree+1)
		{
			initPolynoms(x, level+1, index*2 - 1);
			initPolynoms(x, level+1, index*2 + 1);
		}
		x.pop_back();

	}
};

}

#endif /* MODIFIED_POLY_BASE_HPP */
