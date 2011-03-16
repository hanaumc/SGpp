/******************************************************************************
* Copyright (C) 2009 Technische Universitaet Muenchen                         *
* This file is part of the SG++ project. For conditions of distribution and   *
* use, please see the copyright notice at http://www5.in.tum.de/SGpp          *
******************************************************************************/
// @author Alexander Heinecke (Alexander.Heinecke@mytum.de)

#ifndef HEATEQUATIONPARABOLICPDESOLVERSYSTEM_HPP
#define HEATEQUATIONPARABOLICPDESOLVERSYSTEM_HPP

#include "data/DataVector.hpp"
#include "grid/Grid.hpp"
#include "operation/pde/OperationParabolicPDESolverSystemDirichlet.hpp"
using namespace sg::base;

namespace sg
{

/**
 * This class implements the ParabolicPDESolverSystem for the
 * Heat Equation.
 */
class HeatEquationParabolicPDESolverSystem : public OperationParabolicPDESolverSystemDirichlet
{
private:
	/// the heat coefficient
	double a;
	/// the Laplace Operation (Stiffness Matrix), on boundary grid
	OperationMatrix* OpLaplaceBound;
	/// the LTwoDotProduct Operation (Mass Matrix), on boundary grid
	OperationMatrix* OpMassBound;
	/// the Laplace Operation (Stiffness Matrix), on inner grid
	OperationMatrix* OpLaplaceInner;
	/// the LTwoDotProduct Operation (Mass Matrix), on inner grid
	OperationMatrix* OpMassInner;

	void applyMassMatrixComplete(DataVector& alpha, DataVector& result);

	void applyLOperatorComplete(DataVector& alpha, DataVector& result);

	void applyMassMatrixInner(DataVector& alpha, DataVector& result);

	void applyLOperatorInner(DataVector& alpha, DataVector& result);

public:
	/**
	 * Std-Constructor
	 *
	 * @param SparseGrid reference to the sparse grid
	 * @param a the heat coefficient
	 * @param TimestepSize the size of one timestep used in the ODE Solver
	 * @param OperationMode specifies in which solver this matrix is used, valid values are: ExEul for explicit Euler,
	 *  							ImEul for implicit Euler, CrNic for Crank Nicolson solver
	 */
	HeatEquationParabolicPDESolverSystem(Grid& SparseGrid, DataVector& alpha, double a, double TimestepSize, std::string OperationMode = "ExEul");

	/**
	 * Std-Destructor
	 */
	virtual ~HeatEquationParabolicPDESolverSystem();

	void finishTimestep(bool isLastTimestep = false);

	void startTimestep();
};

}

#endif /* HEATEQUATIONPARABOLICPDESOLVERSYSTEM_HPP */
