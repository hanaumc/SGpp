/*
 * ProlongationRestriction.hpp
 *
 *  Created on: May 16, 2011
 *      Author: benk
 */

#ifndef PROLONGATIONRESTRICTION_HPP_
#define PROLONGATIONRESTRICTION_HPP_

#include "combigrid.hpp"
#include "combigrid/combigrid/AbstractCombiGrid.hpp"

namespace combigrid {

/** class which has two static methods for restrictions and prolongation of the multigrid method. <br>
 * */
class ProlongationRestriction {
public:

    /** empty Ctor*/
	ProlongationRestriction() {;}

	/** empty Dtor*/
	virtual ~ProlongationRestriction() {;}

	/** prolongation with the specified coefficients
	 * @param nrSpace [IN] number of unknowns per node*/
	static void prolongation(const FullGridD* fgFine ,
			                 std::vector<double>& vectFine ,
			                 double coefFine ,
			                 const FullGridD* fgCoarse ,
			                 const std::vector<double>& vectCoarse ,
			                 double coefCoarse ,
			                 int nrSpace) ;

	/** restriction with the specified coefficients
	 * @param nrSpace [IN] number of unknowns per node*/
	static void restriction( const FullGridD* fgFine ,
			                 const std::vector<double>& vectFine ,
			                 double coefFine ,
			                 const FullGridD* fgCoarse ,
			                 std::vector<double>& vectCoarse ,
			                 double coefCoarse ,
			                 int nrSpace) ;

};

}

#endif /* PROLONGATIONRESTRICTION_HPP_ */