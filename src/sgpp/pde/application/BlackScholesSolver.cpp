/******************************************************************************
* Copyright (C) 2009 Technische Universitaet Muenchen                         *
* This file is part of the SG++ project. For conditions of distribution and   *
* use, please see the copyright notice at http://www5.in.tum.de/SGpp          *
******************************************************************************/
// @author Alexander Heinecke (Alexander.Heinecke@mytum.de)

#include "pde/algorithm/BlackScholesParabolicPDESolverSystem.hpp"
#include "pde/algorithm/BlackScholesParabolicPDESolverSystemEuroAmer.hpp"
#include "pde/algorithm/BlackScholesParabolicPDESolverSystemEuroAmerParallelOMP.hpp"
#include "pde/algorithm/BlackScholesPATParabolicPDESolverSystem.hpp"
#include "pde/algorithm/BlackScholesPATParabolicPDESolverSystemEuroAmer.hpp"
#include "pde/algorithm/BlackScholesPATParabolicPDESolverSystemEuroAmerParallelOMP.hpp"
#include "pde/application/BlackScholesSolver.hpp"
#include "solver/ode/Euler.hpp"
#include "solver/ode/CrankNicolson.hpp"
#include "solver/ode/AdamsBashforth.hpp"
#include "solver/ode/VarTimestep.hpp"
#include "solver/ode/StepsizeControlH.hpp"
#include "solver/ode/StepsizeControlBDF.hpp"
#include "solver/ode/StepsizeControlEJ.hpp"
#include "solver/sle/BiCGStab.hpp"
#include "base/grid/Grid.hpp"
#include "base/exception/application_exception.hpp"

#include "solver/sle/ConjugateGradients.hpp"

#include "base/operation/BaseOpFactory.hpp"
#include "pde/operation/PdeOpFactory.hpp"

#include <cstdlib>
#include <sstream>
#include <cmath>
#include <fstream>
#include <iomanip>
using namespace sg::pde;
using namespace sg::solver;
using namespace sg::base;

namespace sg
{
namespace finance
{

  BlackScholesSolver::BlackScholesSolver(bool useLogTransform, bool usePAT) : ParabolicPDESolver()
  {
    this->bStochasticDataAlloc = false;
    this->bGridConstructed = false;
    this->myScreen = NULL;
    this->useCoarsen = false;
    this->coarsenThreshold = 0.0;
    this->adaptSolveMode = "none";
    this->refineMode = "classic";
    this->numCoarsenPoints = -1;
    this->useLogTransform = useLogTransform;
    this->usePAT = usePAT;
    if (this->usePAT == true)
      {
        this->useLogTransform = true;
      }
    this->refineMaxLevel = 0;
    this->nNeededIterations = 0;
    this->dNeededTime = 0.0;
    this->staInnerGridSize = 0;
    this->finInnerGridSize = 0;
    this->avgInnerGridSize = 0;
    this->current_time = 0.0;
    this->tBoundaryType = "freeBoundaries";
  }

  BlackScholesSolver::~BlackScholesSolver()
  {
    if (this->bStochasticDataAlloc)
      {
        delete this->mus;
        delete this->sigmas;
        delete this->rhos;
        delete this->eigval_covar;
        delete this->eigvec_covar;
        delete this->mu_hat;
      }
    if (this->myScreen != NULL)
      {
        delete this->myScreen;
      }
  }

  void BlackScholesSolver::getGridNormalDistribution(DataVector& alpha, std::vector<double>& norm_mu, std::vector<double>& norm_sigma)
  {
    if (this->bGridConstructed)
      {
        double tmp;
        double value;
        StdNormalDistribution myNormDistr;
        double* s_coords = new double[this->dim];

        for (size_t i = 0; i < this->myGrid->getStorage()->size(); i++)
          {
            std::string coords = this->myGridStorage->get(i)->getCoordsStringBB(*(this->myBoundingBox));
            std::stringstream coordsStream(coords);

            for (size_t j = 0; j < this->dim; j++)
              {
                double tmp_load;
                coordsStream >> tmp_load;
                s_coords[j] = tmp_load;
              }

            value = 1.0;
            for (size_t j = 0; j < this->dim; j++)
              {
                if (this->useLogTransform == false)
                  {
                    value *= myNormDistr.getDensity(s_coords[j], norm_mu[j], norm_sigma[j]);
                  }
                else
                  {
                    if (this->usePAT == true)
                      {
                        double inner_tmp = 0.0;
                        for (size_t l = 0; l < dim; l++)
                          {
                            inner_tmp += this->eigvec_covar->get(j, l)*(s_coords[l]-(this->current_time*this->mu_hat->get(l)));
                          }
                        value *= myNormDistr.getDensity(exp(inner_tmp), norm_mu[j], norm_sigma[j]);
                      }
                    else
                      {
                        value *= myNormDistr.getDensity(exp(s_coords[j]), norm_mu[j], norm_sigma[j]);
                      }
                  }
              }

            alpha[i] = value;
          }
        delete[] s_coords;
      }
    else
      {
        throw new application_exception("BlackScholesSolver::getGridNormalDistribution : The grid wasn't initialized before!");
      }
  }

  void BlackScholesSolver::constructGrid(BoundingBox& BoundingBox, size_t level)
  {
    this->dim = BoundingBox.getDimensions();
    this->levels = level;

    this->myGrid = new LinearTrapezoidBoundaryGrid(BoundingBox);

    GridGenerator* myGenerator = this->myGrid->createGridGenerator();
    myGenerator->regular(this->levels);
    delete myGenerator;

    this->myBoundingBox = this->myGrid->getBoundingBox();
    this->myGridStorage = this->myGrid->getStorage();

    //std::string serGrid;
    //myGrid->serialize(serGrid);
    //std::cout << serGrid << std::endl;

    this->bGridConstructed = true;
  }

  void BlackScholesSolver::refineInitialGridWithPayoff(DataVector& alpha, double strike, std::string payoffType, double dStrikeDistance)
  {
    size_t nRefinements = 0;

    this->dStrike = strike;
    this->payoffType = payoffType;

    if (this->useLogTransform == false)
      {
        if (this->bGridConstructed)
          {

            DataVector refineVector(alpha.getSize());

            if (payoffType == "std_euro_call" || payoffType == "std_euro_put" || payoffType == "std_amer_put")
              {
                this->tBoundaryType = "Dirichlet";
                double tmp;
                double* dblFuncValues = new double[dim];
                double dDistance = 0.0;

                for (size_t i = 0; i < this->myGrid->getStorage()->size(); i++)
                  {
                    std::string coords = this->myGridStorage->get(i)->getCoordsStringBB(*(this->myBoundingBox));
                    std::stringstream coordsStream(coords);

                    for (size_t j = 0; j < this->dim; j++)
                      {
                        coordsStream >> tmp;

                        dblFuncValues[j] = tmp;
                      }

                    tmp = 0.0;
                    for (size_t j = 0; j < this->dim; j++)
                      {
                        tmp += dblFuncValues[j];
                      }

                    if (payoffType == "std_euro_call")
                      {
                        dDistance = fabs(((tmp/static_cast<double>(this->dim))-strike));
                      }
                    if (payoffType == "std_euro_put" || payoffType == "std_amer_put")
                      {
                        dDistance = fabs((strike-(tmp/static_cast<double>(this->dim))));
                      }

                    if (dDistance <= dStrikeDistance)
                      {
                        refineVector[i] = dDistance;
                        nRefinements++;
                      }
                    else
                      {
                        refineVector[i] = 0.0;
                      }
                  }

                delete[] dblFuncValues;

                SurplusRefinementFunctor* myRefineFunc = new SurplusRefinementFunctor(&refineVector, nRefinements, 0.0);

                this->myGrid->createGridGenerator()->refine(myRefineFunc);

                delete myRefineFunc;

                alpha.resize(this->myGridStorage->size());

                // reinit the grid with the payoff function
                initGridWithPayoff(alpha, strike, payoffType);
              }
            else
              {
                throw new application_exception("BlackScholesSolver::refineInitialGridWithPayoff : An unsupported payoffType was specified!");
              }
          }
        else
          {
            throw new application_exception("BlackScholesSolver::refineInitialGridWithPayoff : The grid wasn't initialized before!");
          }
      }
  }

  void BlackScholesSolver::refineInitialGridWithPayoffToMaxLevel(DataVector& alpha, double strike, std::string payoffType, double dStrikeDistance, size_t maxLevel)
  {
    size_t nRefinements = 0;

    this->dStrike = strike;
    this->payoffType = payoffType;

    if (this->useLogTransform == false)
      {
        if (this->bGridConstructed)
          {

            DataVector refineVector(alpha.getSize());

            if (payoffType == "std_euro_call" || payoffType == "std_euro_put" || payoffType == "std_amer_put")
              {
                double tmp;
                double* dblFuncValues = new double[dim];
                double dDistance = 0.0;

                this->tBoundaryType = "Dirichlet";

                for (size_t i = 0; i < this->myGrid->getStorage()->size(); i++)
                  {
                    std::string coords = this->myGridStorage->get(i)->getCoordsStringBB(*this->myBoundingBox);
                    std::stringstream coordsStream(coords);

                    for (size_t j = 0; j < this->dim; j++)
                      {
                        coordsStream >> tmp;

                        dblFuncValues[j] = tmp;
                      }

                    tmp = 0.0;
                    for (size_t j = 0; j < this->dim; j++)
                      {
                        tmp += dblFuncValues[j];
                      }

                    if (payoffType == "std_euro_call")
                      {
                        dDistance = fabs(((tmp/static_cast<double>(this->dim))-strike));
                      }
                    if (payoffType == "std_euro_put" || payoffType == "std_amer_put")
                      {
                        dDistance = fabs((strike-(tmp/static_cast<double>(this->dim))));
                      }

                    if (dDistance <= dStrikeDistance)
                      {
                        refineVector[i] = dDistance;
                        nRefinements++;
                      }
                    else
                      {
                        refineVector[i] = 0.0;
                      }
                  }

                delete[] dblFuncValues;

                SurplusRefinementFunctor* myRefineFunc = new SurplusRefinementFunctor(&refineVector, nRefinements, 0.0);

                this->myGrid->createGridGenerator()->refineMaxLevel(myRefineFunc, maxLevel);

                delete myRefineFunc;

                alpha.resize(this->myGridStorage->size());

                // reinit the grid with the payoff function
                initGridWithPayoff(alpha, strike, payoffType);
              }
            else
              {
                throw new application_exception("BlackScholesSolver::refineInitialGridWithPayoffToMaxLevel : An unsupported payoffType was specified!");
              }
          }
        else
          {
            throw new application_exception("BlackScholesSolver::refineInitialGridWithPayoffToMaxLevel : The grid wasn't initialized before!");
          }
      }
  }

  void BlackScholesSolver::setStochasticData(DataVector& mus, DataVector& sigmas, DataMatrix& rhos, double r)
  {
    this->mus = new sg::base::DataVector(mus);
    this->sigmas = new sg::base::DataVector(sigmas);
    this->rhos = new sg::base::DataMatrix(rhos);
    this->r = r;

    // calculate eigenvalues, eigenvectors and mu_hat from stochastic data for PAT
    size_t mydim = this->mus->getSize();
    this->eigval_covar = new sg::base::DataVector(mydim);
    this->eigvec_covar = new sg::base::DataMatrix(mydim,mydim);
    this->mu_hat = new sg::base::DataVector(mydim);

    // 1d test case
    if (mydim == 1)
      {
        this->eigval_covar->set(0, this->sigmas->get(0)*this->sigmas->get(0));
        this->eigvec_covar->set(0, 0, 1.0);
      }
    // 2d test case
    if (mydim == 2)
      {
        // correlation -0.5, sigma_1 0.3, sigma_2 0.4
        this->eigval_covar->set(0, 0.0555377800527509792);
        this->eigval_covar->set(1, 0.194462219947249021);

        this->eigvec_covar->set(0, 0, -0.867142152569025494);
        this->eigvec_covar->set(0, 1, 0.498060726456078796);
        this->eigvec_covar->set(1, 0, -0.498060726456078796);
        this->eigvec_covar->set(1, 1, -0.867142152569025494);

        // correlation 0.1, sigma 0.4
        //		this->eigval_covar->set(0, 0.176);
        //		this->eigval_covar->set(1, 0.144);
        //
        //		this->eigvec_covar->set(0, 0, 0.707106781186547351);
        //		this->eigvec_covar->set(0, 1, -0.707106781186547573);
        //		this->eigvec_covar->set(1, 0, 0.707106781186547573);
        //		this->eigvec_covar->set(1, 1, 0.707106781186547351);

        // correlation 0.25, sigma 0.4
        //		this->eigval_covar->set(0, 0.20);
        //		this->eigval_covar->set(1, 0.12);
        //
        //		this->eigvec_covar->set(0, 0, 0.707106781186547573);
        //		this->eigvec_covar->set(0, 1, -0.707106781186547462);
        //		this->eigvec_covar->set(1, 0, 0.707106781186547462);
        //		this->eigvec_covar->set(1, 1, 0.707106781186547573);

        // correlation 0.5, sigma 0.4
        //		this->eigval_covar->set(0, 0.24);
        //		this->eigval_covar->set(1, 0.08);
        //
        //		this->eigvec_covar->set(0, 0, 0.707106781186547462);
        //		this->eigvec_covar->set(0, 1, -0.707106781186547462);
        //		this->eigvec_covar->set(1, 0, 0.707106781186547462);
        //		this->eigvec_covar->set(1, 1, 0.707106781186547462);

        // correlation -0.5, sigma 0.4
        //		this->eigval_covar->set(0, 0.24);
        //		this->eigval_covar->set(1, 0.08);
        //
        //		this->eigvec_covar->set(0, 0, 0.707106781186547462);
        //		this->eigvec_covar->set(0, 1, 0.707106781186547462);
        //		this->eigvec_covar->set(1, 0, -0.707106781186547462);
        //		this->eigvec_covar->set(1, 1, 0.707106781186547462);

        // correlation 0.0, sigma 0.4
        //		this->eigval_covar->set(0, 0.16);
        //		this->eigval_covar->set(1, 0.16);
        //
        //		this->eigvec_covar->set(0, 0, 1);
        //		this->eigvec_covar->set(0, 1, 0);
        //		this->eigvec_covar->set(1, 0, 0);
        //		this->eigvec_covar->set(1, 1, 1);
      }
    // 3d test case
    if (mydim == 3)
      {
        this->eigval_covar->set(0, 0.0161152062670340546);
        this->eigval_covar->set(1, 0.109759129882028184);
        this->eigval_covar->set(2, 0.164125663850937770);


        this->eigvec_covar->set(0, 0, -0.869836297894464927);
        this->eigvec_covar->set(0, 1, -0.472520156399458380);
        this->eigvec_covar->set(0, 2, -0.141808027493095845);

        this->eigvec_covar->set(1, 0, -0.493287590771869233);
        this->eigvec_covar->set(1, 1, 0.837245861704536964);
        this->eigvec_covar->set(1, 2, 0.235980337844304722);

        this->eigvec_covar->set(2, 0, -0.00722271802968979613);
        this->eigvec_covar->set(2, 1, -0.275216403680555832);
        this->eigvec_covar->set(2, 2, 0.961355170313971441);

        // american put test case, steffi
        //		this->eigval_covar->set(0, 0.0941905506508006474);
        //		this->eigval_covar->set(1, 0.211230588426859212);
        //		this->eigval_covar->set(2, 0.174578860922339874);
        //
        //
        //		this->eigvec_covar->set(0, 0, 0.592851303426220277);
        //		this->eigvec_covar->set(0, 1, -0.610834162257256286);
        //		this->eigvec_covar->set(0, 2, 0.524794205613311915);
        //
        //		this->eigvec_covar->set(1, 0, -0.658103087561460964);
        //		this->eigvec_covar->set(1, 1, -0.743068675017919733);
        //		this->eigvec_covar->set(1, 2, -0.121446574052896814);
        //
        //		this->eigvec_covar->set(2, 0, -0.464141851342780687);
        //		this->eigvec_covar->set(2, 1, 0.273368927324575406);
        //		this->eigvec_covar->set(2, 2, 0.842521080689087154);
      }
    // 4d test case
    if (mydim == 4)
      {
        this->eigval_covar->set(0, 0.203896808612126890);
        this->eigval_covar->set(1, 0.143228838600683389);
        this->eigval_covar->set(2, 0.0369289052706551282);
        this->eigval_covar->set(3, 0.884454475165345477e-1);


        this->eigvec_covar->set(0, 0, -0.758784156527507303);
        this->eigvec_covar->set(0, 1, -0.441609644035335480);
        this->eigvec_covar->set(0, 2, -0.329723883556484409);
        this->eigvec_covar->set(0, 3, 0.347145051398191185);

        this->eigvec_covar->set(1, 0, 0.0381555338704252095);
        this->eigvec_covar->set(1, 1, -0.502978780370165746e-1);
        this->eigvec_covar->set(1, 2, 0.713853106787664671);
        this->eigvec_covar->set(1, 3, 0.697443919343796126);

        this->eigvec_covar->set(2, 0, 0.327315044904916141);
        this->eigvec_covar->set(2, 1, 0.376967846478724333);
        this->eigvec_covar->set(2, 2, -0.600862700798999505);
        this->eigvec_covar->set(2, 3, 0.624278879098610684);

        this->eigvec_covar->set(3, 0, -0.561832377508447611);
        this->eigvec_covar->set(3, 1, 0.812616938342507478);
        this->eigvec_covar->set(3, 2, 0.143735581296216858);
        this->eigvec_covar->set(3, 3, -0.577769311359955864e-1);
      }
    // 5d test case
    if (mydim == 5)
      {
        this->eigval_covar->set(0, 0.248090694157781677);
        this->eigval_covar->set(1, 0.181003240820949346);
        this->eigval_covar->set(2, 0.0132179416451147155);
        this->eigval_covar->set(3, 0.0939549786605669707);
        this->eigval_covar->set(4, 0.0587331447155870698);


        this->eigvec_covar->set(0, 0, 0.263790550378285305);
        this->eigvec_covar->set(0, 1, 0.859923642135395405);
        this->eigvec_covar->set(0, 2, 0.334323851170300113);
        this->eigvec_covar->set(0, 3, -0.280069559202008711);
        this->eigvec_covar->set(0, 4, 0.0271012873267933163);

        this->eigvec_covar->set(1, 0, -0.460592496016481306e-1);
        this->eigvec_covar->set(1, 1, 0.127520536615915014e-1);
        this->eigvec_covar->set(1, 2, -0.377657780213884409);
        this->eigvec_covar->set(1, 3, -0.528472957191017390);
        this->eigvec_covar->set(1, 4, -0.758819389061222926);

        this->eigvec_covar->set(2, 0, -0.164069200805689486);
        this->eigvec_covar->set(2, 1, -0.378822519988808670);
        this->eigvec_covar->set(2, 2, 0.470800766942319815);
        this->eigvec_covar->set(2, 3, -0.728869775137000020);
        this->eigvec_covar->set(2, 4, 0.276893994941337263);

        this->eigvec_covar->set(3, 0, 0.749451659161123107);
        this->eigvec_covar->set(3, 1, -0.143542996441549164);
        this->eigvec_covar->set(3, 2, -0.467549699477188219);
        this->eigvec_covar->set(3, 3, -0.257715994147244276);
        this->eigvec_covar->set(3, 4, 0.364276493384776856);

        this->eigvec_covar->set(4, 0, -0.582834967194725273);
        this->eigvec_covar->set(4, 1, 0.310254123817747751);
        this->eigvec_covar->set(4, 2, -0.552581288090625233);
        this->eigvec_covar->set(4, 3, -0.211207700566551748);
        this->eigvec_covar->set(4, 4, 0.462699694124277694);
      }
    // 6d test case
    if (mydim == 6)
      {
        this->eigval_covar->set(0, 0.295078790670591784);
        this->eigval_covar->set(1, 0.208971923682093358);
        this->eigval_covar->set(2, 0.0128769346283334751);
        this->eigval_covar->set(3, 0.0498094136802732304);
        this->eigval_covar->set(4, 0.0774284377364329035);
        this->eigval_covar->set(5, 0.110834499602274789);


        this->eigvec_covar->set(0, 0, 0.0901970509435878476);
        this->eigvec_covar->set(0, 1, 0.825424523396568799);
        this->eigvec_covar->set(0, 2, 0.355559506316694030);
        this->eigvec_covar->set(0, 3, 0.198721715730790532);
        this->eigvec_covar->set(0, 4, 0.363046763193636213);
        this->eigvec_covar->set(0, 5, 0.113238743705900313);

        this->eigvec_covar->set(1, 0, 0.0365722370191640458);
        this->eigvec_covar->set(1, 1, -0.0144109917849721478);
        this->eigvec_covar->set(1, 2, -0.385578116247248359);
        this->eigvec_covar->set(1, 3, -0.505921195549970393);
        this->eigvec_covar->set(1, 4, 0.742524001100560604);
        this->eigvec_covar->set(1, 5, -0.206121718286344846);

        this->eigvec_covar->set(2, 0, 0.0739556327669701891);
        this->eigvec_covar->set(2, 1, -0.312594988132788687);
        this->eigvec_covar->set(2, 2, 0.484247830906325782);
        this->eigvec_covar->set(2, 3, 0.317369431761546084);
        this->eigvec_covar->set(2, 4, 0.263216340455829090);
        this->eigvec_covar->set(2, 5, -0.701650039506314549);

        this->eigvec_covar->set(3, 0, -.593995441134266944);
        this->eigvec_covar->set(3, 1, 0.333166368988965789);
        this->eigvec_covar->set(3, 2, -0.468239680986198614);
        this->eigvec_covar->set(3, 3, 0.270759276883645317);
        this->eigvec_covar->set(3, 4, -0.153188802604424607);
        this->eigvec_covar->set(3, 5, -0.469194834130158833);

        this->eigvec_covar->set(4, 0, 0.575964981307517609);
        this->eigvec_covar->set(4, 1, -0.0554637711185953924);
        this->eigvec_covar->set(4, 2, -0.516994216391904682);
        this->eigvec_covar->set(4, 3, 0.613357677157853276);
        this->eigvec_covar->set(4, 4, 0.135825834003240276);
        this->eigvec_covar->set(4, 5, 0.0569981868738118971);

        this->eigvec_covar->set(5, 0, 0.548170853002664993);
        this->eigvec_covar->set(5, 1, 0.326611380012385655);
        this->eigvec_covar->set(5, 2, -0.0622862213401522563);
        this->eigvec_covar->set(5, 3, -0.392825458984802756);
        this->eigvec_covar->set(5, 4, -0.453494043197070207);
        this->eigvec_covar->set(5, 5, -0.478524252837880804);
      }
    // 7d test case
    if (mydim == 7)
      {
        this->eigval_covar->set(0, 0.297277846593271722);
        this->eigval_covar->set(1, 0.208971930701126907);
        this->eigval_covar->set(2, 0.127627200649076539);
        this->eigval_covar->set(3, 0.0128255347503872102);
        this->eigval_covar->set(4, 0.0306260409692614491);
        this->eigval_covar->set(5, 0.0811043629942213989);
        this->eigval_covar->set(6, 0.0590670833426543418);


        this->eigvec_covar->set(0, 0, 0.0843288248079574199);
        this->eigvec_covar->set(0, 1, 0.825397719934479079);
        this->eigvec_covar->set(0, 2, -0.168709288751078074);
        this->eigvec_covar->set(0, 3, -0.363307928519013612);
        this->eigvec_covar->set(0, 4, -0.202137777735263119);
        this->eigvec_covar->set(0, 5, -0.331775600546961924);
        this->eigvec_covar->set(0, 6, -0.0147325453249897778);

        this->eigvec_covar->set(1, 0, 0.0388837766210140823);
        this->eigvec_covar->set(1, 1, -0.0144026678855673478);
        this->eigvec_covar->set(1, 2, 0.138436093520141523);
        this->eigvec_covar->set(1, 3, 0.391084279235813126);
        this->eigvec_covar->set(1, 4, 0.196762223947347770);
        this->eigvec_covar->set(1, 5, -0.670298189519680343);
        this->eigvec_covar->set(1, 6, 0.581510201090990941);

        this->eigvec_covar->set(2, 0, 0.0816282868053433697);
        this->eigvec_covar->set(2, 1, -0.312561527401625017);
        this->eigvec_covar->set(2, 2, 0.544757290477682066);
        this->eigvec_covar->set(2, 3, -0.481767290108688295);
        this->eigvec_covar->set(2, 4, -0.0150731089385307045);
        this->eigvec_covar->set(2, 5, -0.480781834905232630);
        this->eigvec_covar->set(2, 6, -0.367972166945297441);

        this->eigvec_covar->set(3, 0, -0.583758726401433226);
        this->eigvec_covar->set(3, 1, 0.333233472084286164);
        this->eigvec_covar->set(3, 2, 0.441249303136342808);
        this->eigvec_covar->set(3, 3, 0.468515871811986007);
        this->eigvec_covar->set(3, 4, -0.0558306699252684793);
        this->eigvec_covar->set(3, 5, -0.00669728666705872531);
        this->eigvec_covar->set(3, 6, -0.361678331424167054);

        this->eigvec_covar->set(4, 0, 0.567646273735049278);
        this->eigvec_covar->set(4, 1, -0.0555178986758357862);
        this->eigvec_covar->set(4, 2, -0.156878448552707023);
        this->eigvec_covar->set(4, 3, 0.504829137586297549);
        this->eigvec_covar->set(4, 4, -0.401843814498330509);
        this->eigvec_covar->set(4, 5, -0.202551001291546379);
        this->eigvec_covar->set(4, 6, -0.439006612353535552);

        this->eigvec_covar->set(5, 0, 0.558697722750611581);
        this->eigvec_covar->set(5, 1, 0.326633776766320549);
        this->eigvec_covar->set(5, 2, 0.476573019522854413);
        this->eigvec_covar->set(5, 3, 0.0821587688846129316);
        this->eigvec_covar->set(5, 4, 0.494200581047168797);
        this->eigvec_covar->set(5, 5, 0.320995293884535349);
        this->eigvec_covar->set(5, 6, 0.00480949306111711893);

        this->eigvec_covar->set(6, 0, 0.0978459803650174598);
        this->eigvec_covar->set(6, 1, 0.000224850494160951678);
        this->eigvec_covar->set(6, 2, 0.457360339661911197);
        this->eigvec_covar->set(6, 3, -0.0430240652502010823);
        this->eigvec_covar->set(6, 4, -0.715100005708458997);
        this->eigvec_covar->set(6, 5, 0.255664574715722459);
        this->eigvec_covar->set(6, 6, 0.450182374325719670);
      }

    for (size_t i = 0; i < mydim; i++)
      {
        double tmp = 0.0;
        for (size_t j = 0; j < mydim; j++)
          {
            tmp += ((this->mus->get(j) - (0.5*this->sigmas->get(j)*this->sigmas->get(j))) * this->eigvec_covar->get(j, i));
          }
        this->mu_hat->set(i, tmp);
      }

    bStochasticDataAlloc = true;
  }

  void BlackScholesSolver::solveExplicitEuler(size_t numTimesteps, double timestepsize, size_t maxCGIterations, double epsilonCG, DataVector& alpha, bool verbose, bool generateAnimation, size_t numEvalsAnimation)
  {
    if (this->bGridConstructed && this->bStochasticDataAlloc)
      {
        Euler* myEuler = new Euler("ExEul", numTimesteps, timestepsize, generateAnimation, numEvalsAnimation, myScreen);
        SLESolver* myCG = NULL;
        OperationParabolicPDESolverSystem* myBSSystem = NULL;

        if (this->tBoundaryType == "Dirichlet")
          {
            if (this->usePAT == true)
              {
                myCG = new ConjugateGradients(maxCGIterations, epsilonCG);
#ifdef _OPENMP
                myBSSystem = new BlackScholesPATParabolicPDESolverSystemEuroAmerParallelOMP(*this->myGrid, alpha, *this->eigval_covar, *this->eigvec_covar, *this->mu_hat, timestepsize, "ExEul", this->dStrike, this->payoffType, this->r, this->useCoarsen, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
#else
                myBSSystem = new BlackScholesPATParabolicPDESolverSystemEuroAmer(*this->myGrid, alpha, *this->eigval_covar, *this->eigvec_covar, *this->mu_hat, timestepsize, "ExEul", this->dStrike, this->payoffType, this->r, this->useCoarsen, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
#endif
              }
            else
              {
                myCG = new BiCGStab(maxCGIterations, epsilonCG);
#ifdef _OPENMP
                myBSSystem = new BlackScholesParabolicPDESolverSystemEuroAmerParallelOMP(*this->myGrid, alpha, *this->mus, *this->sigmas, *this->rhos, this->r, timestepsize, "ExEul", this->dStrike, this->payoffType, this->useLogTransform, this->useCoarsen, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
#else
                myBSSystem = new BlackScholesParabolicPDESolverSystemEuroAmer(*this->myGrid, alpha, *this->mus, *this->sigmas, *this->rhos, this->r, timestepsize, "ExEul", this->dStrike, this->payoffType, this->useLogTransform, this->useCoarsen, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
#endif
              }
          }
        else
          {
            if (this->usePAT == true)
              {
                myCG = new ConjugateGradients(maxCGIterations, epsilonCG);
                myBSSystem = new BlackScholesPATParabolicPDESolverSystem(*this->myGrid, alpha, *this->eigval_covar, *this->eigvec_covar, *this->mu_hat, timestepsize, "ExEul", this->dStrike, this->payoffType, this->useCoarsen, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
              }
            else
              {
                myCG = new BiCGStab(maxCGIterations, epsilonCG);
                myBSSystem = new BlackScholesParabolicPDESolverSystem(*this->myGrid, alpha, *this->mus, *this->sigmas, *this->rhos, this->r, timestepsize, "ExEul", this->dStrike, this->payoffType, this->useLogTransform, this->useCoarsen, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
              }
          }

        SGppStopwatch* myStopwatch = new SGppStopwatch();
        this->staInnerGridSize = getNumberInnerGridPoints();

        std::cout << "Using Explicit Euler to solve " << numTimesteps << " timesteps:" << std::endl;
        myStopwatch->start();
        myEuler->solve(*myCG, *myBSSystem, true, verbose);
        this->dNeededTime = myStopwatch->stop();

        std::cout << std::endl << "Final Grid size: " << getNumberGridPoints() << std::endl;
        std::cout << "Final Grid size (inner): " << getNumberInnerGridPoints() << std::endl << std::endl << std::endl;

        std::cout << "Average Grid size: " << static_cast<double>(myBSSystem->getSumGridPointsComplete())/static_cast<double>(numTimesteps) << std::endl;
        std::cout << "Average Grid size (Inner): " << static_cast<double>(myBSSystem->getSumGridPointsInner())/static_cast<double>(numTimesteps) << std::endl << std::endl << std::endl;

        if (this->myScreen != NULL)
          {
            std::cout << "Time to solve: " << this->dNeededTime << " seconds" << std::endl;
            this->myScreen->writeEmptyLines(2);
          }

        this->finInnerGridSize = getNumberInnerGridPoints();
        this->avgInnerGridSize = static_cast<size_t>((static_cast<double>(myBSSystem->getSumGridPointsInner())/static_cast<double>(numTimesteps))+0.5);
        this->nNeededIterations = myEuler->getNumberIterations();

        delete myBSSystem;
        delete myCG;
        delete myEuler;
        delete myStopwatch;

        this->current_time += (static_cast<double>(numTimesteps)*timestepsize);
      }
    else
      {
        throw new application_exception("BlackScholesSolver::solveExplicitEuler : A grid wasn't constructed before or stochastic parameters weren't set!");
      }
  }

  void BlackScholesSolver::solveImplicitEuler(size_t numTimesteps, double timestepsize, size_t maxCGIterations, double epsilonCG, DataVector& alpha, bool verbose, bool generateAnimation, size_t numEvalsAnimation)
  {
    if (this->bGridConstructed && this->bStochasticDataAlloc)
      {
        Euler* myEuler = new Euler("ImEul", numTimesteps, timestepsize, generateAnimation, numEvalsAnimation, myScreen);
        SLESolver* myCG = NULL;
        OperationParabolicPDESolverSystem* myBSSystem = NULL;

        if (this->tBoundaryType == "Dirichlet")
          {
            if (this->usePAT == true)
              {
                myCG = new ConjugateGradients(maxCGIterations, epsilonCG);
#ifdef _OPENMP
                myBSSystem = new BlackScholesPATParabolicPDESolverSystemEuroAmerParallelOMP(*this->myGrid, alpha, *this->eigval_covar, *this->eigvec_covar, *this->mu_hat, timestepsize, "ImEul", this->dStrike, this->payoffType, this->r, this->useCoarsen, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
#else
                myBSSystem = new BlackScholesPATParabolicPDESolverSystemEuroAmer(*this->myGrid, alpha, *this->eigval_covar, *this->eigvec_covar, *this->mu_hat, timestepsize, "ImEul", this->dStrike, this->payoffType, this->r, this->useCoarsen, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
#endif
              }
            else
              {
                myCG = new BiCGStab(maxCGIterations, epsilonCG);
#ifdef _OPENMP
                myBSSystem = new BlackScholesParabolicPDESolverSystemEuroAmerParallelOMP(*this->myGrid, alpha, *this->mus, *this->sigmas, *this->rhos, this->r, timestepsize, "ImEul", this->dStrike, this->payoffType, this->useLogTransform, this->useCoarsen, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
#else
                myBSSystem = new BlackScholesParabolicPDESolverSystemEuroAmer(*this->myGrid, alpha, *this->mus, *this->sigmas, *this->rhos, this->r, timestepsize, "ImEul", this->dStrike, this->payoffType, this->useLogTransform, this->useCoarsen, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
#endif
              }
          }
        else
          {
            if (this->usePAT == true)
              {
                myCG = new ConjugateGradients(maxCGIterations, epsilonCG);
                myBSSystem = new BlackScholesPATParabolicPDESolverSystem(*this->myGrid, alpha, *this->eigval_covar, *this->eigvec_covar, *this->mu_hat, timestepsize, "ImEul", this->dStrike, this->payoffType, this->useCoarsen, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
              }
            else
              {
                myCG = new BiCGStab(maxCGIterations, epsilonCG);
                myBSSystem = new BlackScholesParabolicPDESolverSystem(*this->myGrid, alpha, *this->mus, *this->sigmas, *this->rhos, this->r, timestepsize, "ImEul", this->dStrike, this->payoffType, this->useLogTransform, this->useCoarsen, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
              }
          }

        SGppStopwatch* myStopwatch = new SGppStopwatch();
        this->staInnerGridSize = getNumberInnerGridPoints();

        std::cout << "Using Implicit Euler to solve " << numTimesteps << " timesteps:" << std::endl;
        myStopwatch->start();
        myEuler->solve(*myCG, *myBSSystem, true, verbose);
        this->dNeededTime = myStopwatch->stop();

        std::cout << std::endl << "Final Grid size: " << getNumberGridPoints() << std::endl;
        std::cout << "Final Grid size (inner): " << getNumberInnerGridPoints() << std::endl << std::endl << std::endl;

        std::cout << "Average Grid size: " << static_cast<double>(myBSSystem->getSumGridPointsComplete())/static_cast<double>(numTimesteps) << std::endl;
        std::cout << "Average Grid size (Inner): " << static_cast<double>(myBSSystem->getSumGridPointsInner())/static_cast<double>(numTimesteps) << std::endl << std::endl << std::endl;

        if (this->myScreen != NULL)
          {
            std::cout << "Time to solve: " << this->dNeededTime << " seconds" << std::endl;
            this->myScreen->writeEmptyLines(2);
          }

        this->finInnerGridSize = getNumberInnerGridPoints();
        this->avgInnerGridSize = static_cast<size_t>((static_cast<double>(myBSSystem->getSumGridPointsInner())/static_cast<double>(numTimesteps))+0.5);
        this->nNeededIterations = myEuler->getNumberIterations();

        delete myBSSystem;
        delete myCG;
        delete myEuler;
        delete myStopwatch;

        this->current_time += (static_cast<double>(numTimesteps)*timestepsize);
      }
    else
      {
        throw new application_exception("BlackScholesSolver::solveImplicitEuler : A grid wasn't constructed before or stochastic parameters weren't set!");
      }
  }

  void BlackScholesSolver::solveCrankNicolson(size_t numTimesteps, double timestepsize, size_t maxCGIterations, double epsilonCG, DataVector& alpha, size_t NumImEul)
  {
    if (this->bGridConstructed && this->bStochasticDataAlloc)
      {
        SLESolver* myCG = NULL;
        OperationParabolicPDESolverSystem* myBSSystem = NULL;

        if (this->tBoundaryType == "Dirichlet")
          {
            if (this->usePAT == true)
              {
                myCG = new ConjugateGradients(maxCGIterations, epsilonCG);
#ifdef _OPENMP
                myBSSystem = new BlackScholesPATParabolicPDESolverSystemEuroAmerParallelOMP(*this->myGrid, alpha, *this->eigval_covar, *this->eigvec_covar, *this->mu_hat, timestepsize, "ImEul", this->dStrike, this->payoffType, this->r, this->useCoarsen, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
#else
                myBSSystem = new BlackScholesPATParabolicPDESolverSystemEuroAmer(*this->myGrid, alpha, *this->eigval_covar, *this->eigvec_covar, *this->mu_hat, timestepsize, "ImEul", this->dStrike, this->payoffType, this->r, this->useCoarsen, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
#endif
              }
            else
              {
                myCG = new BiCGStab(maxCGIterations, epsilonCG);
#ifdef _OPENMP
                myBSSystem = new BlackScholesParabolicPDESolverSystemEuroAmerParallelOMP(*this->myGrid, alpha, *this->mus, *this->sigmas, *this->rhos, this->r, timestepsize, "ImEul", this->dStrike, this->payoffType, this->useLogTransform, this->useCoarsen, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
#else
                myBSSystem = new BlackScholesParabolicPDESolverSystemEuroAmer(*this->myGrid, alpha, *this->mus, *this->sigmas, *this->rhos, this->r, timestepsize, "ImEul", this->dStrike, this->payoffType, this->useLogTransform, this->useCoarsen, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
#endif
              }
          }
        else
          {
            if (this->usePAT == true)
              {
                myCG = new ConjugateGradients(maxCGIterations, epsilonCG);
                myBSSystem = new BlackScholesPATParabolicPDESolverSystem(*this->myGrid, alpha, *this->eigval_covar, *this->eigvec_covar, *this->mu_hat, timestepsize, "ImEul", this->dStrike, this->payoffType, this->useCoarsen, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
              }
            else
              {
                myCG = new BiCGStab(maxCGIterations, epsilonCG);
                myBSSystem = new BlackScholesParabolicPDESolverSystem(*this->myGrid, alpha, *this->mus, *this->sigmas, *this->rhos, this->r, timestepsize, "ImEul", this->dStrike, this->payoffType, this->useLogTransform, this->useCoarsen, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
              }
          }

        SGppStopwatch* myStopwatch = new SGppStopwatch();
        this->staInnerGridSize = getNumberInnerGridPoints();

        size_t numCNSteps;
        size_t numIESteps;

        numCNSteps = numTimesteps;
        if (numTimesteps > NumImEul)
          {
            numCNSteps = numTimesteps - NumImEul;
          }
        numIESteps = NumImEul;

        Euler* myEuler = new Euler("ImEul", numIESteps, timestepsize, false, 0, this->myScreen);
        CrankNicolson* myCN = new CrankNicolson(numCNSteps, timestepsize, this->myScreen);

        myStopwatch->start();
        if (numIESteps > 0)
          {
            std::cout << "Using Implicit Euler to solve " << numIESteps << " timesteps:" << std::endl;
            myBSSystem->setODESolver("ImEul");
            myEuler->solve(*myCG, *myBSSystem, false, false);
          }
        myBSSystem->setODESolver("CrNic");
        std::cout << "Using Crank Nicolson to solve " << numCNSteps << " timesteps:" << std::endl << std::endl << std::endl << std::endl;
        myCN->solve(*myCG, *myBSSystem, true, false);
        this->dNeededTime = myStopwatch->stop();

        std::cout << std::endl << "Final Grid size: " << getNumberGridPoints() << std::endl;
        std::cout << "Final Grid size (inner): " << getNumberInnerGridPoints() << std::endl << std::endl << std::endl;

        std::cout << "Average Grid size: " << static_cast<double>(myBSSystem->getSumGridPointsComplete())/static_cast<double>(numTimesteps) << std::endl;
        std::cout << "Average Grid size (Inner): " << static_cast<double>(myBSSystem->getSumGridPointsInner())/static_cast<double>(numTimesteps) << std::endl << std::endl << std::endl;

        if (this->myScreen != NULL)
          {
            std::cout << "Time to solve: " << this->dNeededTime << " seconds" << std::endl;
            this->myScreen->writeEmptyLines(2);
          }

        this->finInnerGridSize = getNumberInnerGridPoints();
        this->avgInnerGridSize = static_cast<size_t>((static_cast<double>(myBSSystem->getSumGridPointsInner())/static_cast<double>(numTimesteps))+0.5);
        this->nNeededIterations = myEuler->getNumberIterations() + myCN->getNumberIterations();

        delete myBSSystem;
        delete myCG;
        delete myCN;
        delete myEuler;
        delete myStopwatch;

        this->current_time += (static_cast<double>(numTimesteps)*timestepsize);
      }
    else
      {
        throw new application_exception("BlackScholesSolver::solveCrankNicolson : A grid wasn't constructed before or stochastic parameters weren't set!");
      }
  }


  void BlackScholesSolver::solveAdamsBashforth(size_t numTimesteps, double timestepsize, size_t maxCGIterations, double epsilonCG, DataVector& alpha, bool verbose)
  {
    ODESolver* myODESolver = new AdamsBashforth(numTimesteps, timestepsize, myScreen);
    BlackScholesSolver::solveX(numTimesteps, timestepsize, maxCGIterations, epsilonCG, alpha, verbose, myODESolver, "AdBas");
    delete myODESolver;
  }

  void BlackScholesSolver::solveSCAC(size_t numTimesteps, double timestepsize, double epsilon, size_t maxCGIterations, double epsilonCG, DataVector& alpha, bool verbose)
  {
    ODESolver* myODESolver = new VarTimestep(numTimesteps, timestepsize, epsilon, myScreen);
    BlackScholesSolver::solveX(numTimesteps, timestepsize, maxCGIterations, epsilonCG, alpha, verbose, myODESolver, "CrNic");
    delete myODESolver;
  }

  void BlackScholesSolver::solveSCH(size_t numTimesteps, double timestepsize, double epsilon, size_t maxCGIterations, double epsilonCG, DataVector& alpha, bool verbose)
  {
    ODESolver* myODESolver = new StepsizeControlH(numTimesteps, timestepsize, epsilon, myScreen);
    BlackScholesSolver::solveX(numTimesteps, timestepsize, maxCGIterations, epsilonCG, alpha, verbose, myODESolver, "CrNic");
    delete myODESolver;
  }

  void BlackScholesSolver::solveSCBDF(size_t numTimesteps, double timestepsize, double epsilon, size_t maxCGIterations, double epsilonCG, DataVector& alpha, bool verbose)
  {
    ODESolver* myODESolver = new StepsizeControlBDF(numTimesteps, timestepsize, epsilon, myScreen);
    BlackScholesSolver::solveX(numTimesteps, timestepsize, maxCGIterations, epsilonCG, alpha, verbose, myODESolver, "SCBDF");
    delete myODESolver;
  }

  void BlackScholesSolver::solveSCEJ(size_t numTimesteps, double timestepsize, double epsilon, double myAlpha, size_t maxCGIterations, double epsilonCG, DataVector& alpha, bool verbose)
  {
    ODESolver* myODESolver = new StepsizeControlEJ(numTimesteps, timestepsize, epsilon, myAlpha,  myScreen);
    BlackScholesSolver::solveX(numTimesteps, timestepsize, maxCGIterations, epsilonCG, alpha, verbose, myODESolver, "SCEJ");
    delete myODESolver;
  }

  void BlackScholesSolver::solveX(size_t numTimesteps, double timestepsize, size_t maxCGIterations, double epsilonCG, DataVector& alpha, bool verbose, void *myODESolverV, std::string Solver)
  {
    ODESolver *myODESolver = (ODESolver *)myODESolverV;
    if (this->bGridConstructed && this->bStochasticDataAlloc)
      {		BiCGStab* myCG = new BiCGStab(maxCGIterations, epsilonCG);
        OperationParabolicPDESolverSystem* myBSSystem = NULL;

        if (this->tBoundaryType == "Dirichlet")
          {
#ifdef _OPENMP
            myBSSystem = new BlackScholesParabolicPDESolverSystemEuroAmerParallelOMP(*this->myGrid, alpha, *this->mus, *this->sigmas, *this->rhos, this->r, timestepsize, Solver, this->dStrike, this->payoffType, this->useLogTransform, false, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
#else
            myBSSystem = new BlackScholesParabolicPDESolverSystemEuroAmer(*this->myGrid, alpha, *this->mus, *this->sigmas, *this->rhos, this->r, timestepsize, Solver, this->dStrike, this->payoffType, this->useLogTransform, false, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
#endif
          }
        else
          {
            myBSSystem = new BlackScholesParabolicPDESolverSystem(*this->myGrid, alpha, *this->mus, *this->sigmas, *this->rhos, this->r, timestepsize, Solver, this->dStrike, this->payoffType, this->useLogTransform, false, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
          }

        SGppStopwatch* myStopwatch = new SGppStopwatch();
        this->staInnerGridSize = getNumberInnerGridPoints();

        myStopwatch->start();
        myODESolver->solve(*myCG, *myBSSystem, false, verbose);
        this->dNeededTime = myStopwatch->stop();

        if (this->myScreen != NULL)
          {
            std::cout << "Time to solve: " << this->dNeededTime << " seconds" << std::endl;
            this->myScreen->writeEmptyLines(2);
          }

        this->finInnerGridSize = getNumberInnerGridPoints();
        this->avgInnerGridSize = static_cast<size_t>((static_cast<double>(myBSSystem->getSumGridPointsInner())/static_cast<double>(numTimesteps))+0.5);
        this->nNeededIterations = myODESolver->getNumberIterations();

        delete myBSSystem;
        delete myCG;

        this->current_time += (static_cast<double>(numTimesteps)*timestepsize);
      }
    else
      {
        throw new application_exception("BlackScholesSolver::solveX : A grid wasn't constructed before or stochastic parameters weren't set!");
      }
  }


  void BlackScholesSolver::initGridWithPayoff(DataVector& alpha, double strike, std::string payoffType)
  {
    this->dStrike = strike;
    this->payoffType = payoffType;

    if (payoffType == "std_euro_call" || payoffType == "std_euro_put" || payoffType == "std_amer_put")
      {
        this->tBoundaryType = "Dirichlet";
      }

    if (this->useLogTransform)
      {
        if (this->usePAT)
          {
            initPATTransformedGridWithPayoff(alpha, strike, payoffType);
          }
        else
          {
            initLogTransformedGridWithPayoff(alpha, strike, payoffType);
          }
      }
    else
      {
        initCartesianGridWithPayoff(alpha, strike, payoffType);
      }
  }

  double BlackScholesSolver::get1DEuroCallPayoffValue(double assetValue, double strike)
  {
    if (assetValue <= strike)
      {
        return 0.0;
      }
    else
      {
        return assetValue - strike;
      }
  }

  double BlackScholesSolver::getAnalyticSolution1D(double stock, bool isCall, double t, double vola, double r, double strike)
  {
    StdNormalDistribution* myStdNDis = new StdNormalDistribution();

    double dOne = (log((stock/strike)) + ((r + (vola*vola*0.5))*(t)))/(vola*sqrt(t));
    double dTwo = dOne - (vola*sqrt(t));

    double prem;
    if (isCall)
      {
        return (stock*myStdNDis->getCumulativeDensity(dOne)) - (strike*myStdNDis->getCumulativeDensity(dTwo)*(exp((-1.0)*r*t)));
      }
    else
      {
        return (strike*myStdNDis->getCumulativeDensity(dTwo*(-1.0))*(exp((-1.0)*r*t))) - (stock*myStdNDis->getCumulativeDensity(dOne*(-1.0)));
      }

    delete myStdNDis;
  }


  void BlackScholesSolver::solve1DAnalytic(std::vector< std::pair<double, double> >& premiums, double minStock, double maxStock, double StockInc, double strike, double t, bool isCall)
  {
    if (bStochasticDataAlloc)
      {
        double stock = 0.0;
        double vola = this->sigmas->get(0);

        for (stock = minStock; stock <= maxStock; stock += StockInc)
          {
            double prem = getAnalyticSolution1D(stock, isCall, t, vola, this->r, strike);
            premiums.push_back(std::make_pair(stock, prem));
          }
      }
    else
      {
        throw new application_exception("BlackScholesSolver::solve1DAnalytic : Stochastic parameters weren't set!");
      }
  }

  void BlackScholesSolver::print1DAnalytic(std::vector< std::pair<double, double> >& premiums, std::string tfilename)
  {
    typedef std::vector< std::pair<double, double> > printVector;
    std::ofstream fileout;

    fileout.open(tfilename.c_str());
    for(printVector::iterator iter = premiums.begin(); iter != premiums.end(); iter++)
      {
        fileout << iter->first << " " << iter->second << " " << std::endl;
      }
    fileout.close();
  }


  void BlackScholesSolver::getAnalyticAlpha1D(DataVector& alpha_analytic, double strike, double t, std::string payoffType, bool hierarchized)
  {
    double coord;

    if(dim!=1)
      {
        throw new application_exception("BlackScholesSolver::getAnalyticAlpha1D : A grid wasn't constructed before!");
      }
    if (!this->bGridConstructed)
      {
        throw new application_exception("BlackScholesSolver::getAnalyticAlpha1D : function only available for dim = 1!");
      }

    // compute values of analytic solution on given grid
    for (size_t i = 0; i < this->myGridStorage->size(); i++)
      {
        std::string coords = this->myGridStorage->get(i)->getCoordsStringBB(*this->myBoundingBox);
        std::stringstream coordsStream(coords);
        coordsStream >> coord;
        if(useLogTransform)
          {
            coord = exp(coord);
          }
        if (payoffType == "std_euro_call")
          {
            alpha_analytic[i] = getAnalyticSolution1D(coord, true, t, this->sigmas->get(0), this->r, strike);
          }
        else if (payoffType == "std_euro_put")
          {
            alpha_analytic[i] = getAnalyticSolution1D(coord, false, t, this->sigmas->get(0), this->r, strike);
          }
      }

    if(hierarchized)
      {
        // hierarchize computed values
        OperationHierarchisation* myHier = sg::op_factory::createOperationHierarchisation(*this->myGrid);
        myHier->doHierarchisation(alpha_analytic);

        delete myHier;
      }
  }


  void BlackScholesSolver::evaluate1DAnalyticCuboid(sg::base::DataVector& AnalyticOptionPrices, sg::base::DataMatrix& EvaluationPoints, double strike, double vola, double r, double t, bool isCall)
  {
    int n = EvaluationPoints.getNrows();

    if (AnalyticOptionPrices.getSize() != n)
      {
        throw new sg::base::application_exception("PDESolver::evaluate1DAnalyticCuboid : The size of the price vector doesn't match the size of the evaluation points' vector!");
      }

    for(int k=0; k<n; k++)
      {
        double x = EvaluationPoints.get(k,0); // get first coordinate

        if(this->useLogTransform)
          {
            x = exp(x);
          }
        double price = getAnalyticSolution1D(x, isCall, t, vola, r, strike);
        AnalyticOptionPrices.set(k,price);
      }
  }

  std::vector<size_t> BlackScholesSolver::getAlgorithmicDimensions()
  {
    return this->myGrid->getAlgorithmicDimensions();
  }

  void BlackScholesSolver::setAlgorithmicDimensions(std::vector<size_t> newAlgoDims)
  {
    if (this->tBoundaryType == "freeBoundaries")
      {
        this->myGrid->setAlgorithmicDimensions(newAlgoDims);
      }
    else
      {
        throw new application_exception("BlackScholesSolver::setAlgorithmicDimensions : Set algorithmic dimensions is only supported when choosing option type all!");
      }
  }

  void BlackScholesSolver::initScreen()
  {
    this->myScreen = new ScreenOutput();
    this->myScreen->writeTitle("SGpp - Black Scholes Solver, 2.0.0", "TUM (C) 2009-2010, by Alexander Heinecke");
    this->myScreen->writeStartSolve("Multidimensional Black Scholes Solver");
  }

  void BlackScholesSolver::setEnableCoarseningData(std::string adaptSolveMode, std::string refineMode, size_t refineMaxLevel, int numCoarsenPoints, double coarsenThreshold, double refineThreshold)
  {
    this->useCoarsen = true;
    this->coarsenThreshold = coarsenThreshold;
    this->refineThreshold = refineThreshold;
    this->refineMaxLevel = refineMaxLevel;
    this->adaptSolveMode = adaptSolveMode;
    this->refineMode = refineMode;
    this->numCoarsenPoints = numCoarsenPoints;
  }

  void BlackScholesSolver::printPayoffInterpolationError2D(DataVector& alpha, std::string tFilename, size_t numTestpoints, double strike)
  {
    if (this->useLogTransform == false)
      {
        if (this->bGridConstructed)
          {
            if (this->myGrid->getStorage()->getBoundingBox()->getDimensions() == 2)
              {
                if (numTestpoints < 2)
                  numTestpoints = 2;

                double dInc = (2.0*strike)/static_cast<double>(numTestpoints-1);

                double dX = 0.0;
                double dY = 2*strike;

                std::ofstream file;
                file.open(tFilename.c_str());

                OperationEval* myEval = sg::op_factory::createOperationEval(*this->myGrid);

                for (size_t i = 0; i < numTestpoints; i++)
                  {
                    std::vector<double> point;

                    point.push_back(dX);
                    point.push_back(dY);

                    double result = myEval->eval(alpha, point);

                    file << std::scientific << std::setprecision( 16 ) << dX << " " << dY << " " << result << std::endl;

                    dX += dInc;
                    dY -= dInc;
                  }

                delete myEval;

                file.close();
              }
          }
        else
          {
            throw new application_exception("BlackScholesSolver::getPayoffInterpolationError : A grid wasn't constructed before!");
          }
      }
  }

  size_t BlackScholesSolver::getGridPointsAtMoney(std::string payoffType, double strike, double eps)
  {
    size_t nPoints = 0;

    if (this->useLogTransform == false)
      {
        if (this->bGridConstructed)
          {
            for (size_t i = 0; i < this->myGrid->getStorage()->size(); i++)
              {
                bool isAtMoney = true;
                DataVector coords(this->dim);
                this->myGridStorage->get(i)->getCoordsBB(coords, *this->myBoundingBox);

                if (payoffType == "std_euro_call" || payoffType == "std_euro_put" || payoffType == "std_amer_put")
                  {
                    for (size_t d = 0; d < this->dim; d++)
                      {
                        if ( ((coords.sum()/static_cast<double>(this->dim)) < (strike-eps)) || ((coords.sum()/static_cast<double>(this->dim)) > (strike+eps)) )
                          {
                            isAtMoney = false;
                          }

                      }
                  }
                else
                  {
                    throw new application_exception("BlackScholesSolver::getGridPointsAtMoney : An unknown payoff-type was specified!");
                  }

                if (isAtMoney == true)
                  {
                    nPoints++;
                  }
              }
          }
        else
          {
            throw new application_exception("BlackScholesSolver::getGridPointsAtMoney : A grid wasn't constructed before!");
          }
      }

    return nPoints;
  }

  void BlackScholesSolver::initCartesianGridWithPayoff(DataVector& alpha, double strike, std::string payoffType)
  {
    double tmp;

    if (this->bGridConstructed)
      {
        for (size_t i = 0; i < this->myGrid->getStorage()->size(); i++)
          {
            std::string coords = this->myGridStorage->get(i)->getCoordsStringBB(*this->myBoundingBox);
            std::stringstream coordsStream(coords);
            double* dblFuncValues = new double[dim];

            for (size_t j = 0; j < this->dim; j++)
              {
                coordsStream >> tmp;

                dblFuncValues[j] = tmp;
              }

            if (payoffType == "std_euro_call")
              {
                tmp = 0.0;
                for (size_t j = 0; j < dim; j++)
                  {
                    tmp += dblFuncValues[j];
                  }
                alpha[i] = std::max<double>(((tmp/static_cast<double>(dim))-strike), 0.0);
              }
            else if (payoffType == "std_euro_put" || payoffType == "std_amer_put")
              {
                tmp = 0.0;
                for (size_t j = 0; j < dim; j++)
                  {
                    tmp += dblFuncValues[j];
                  }
                alpha[i] = std::max<double>(strike-((tmp/static_cast<double>(dim))), 0.0);
              }
            else
              {
                throw new application_exception("BlackScholesSolver::initCartesianGridWithPayoff : An unknown payoff-type was specified!");
              }

            delete[] dblFuncValues;
          }

        OperationHierarchisation* myHierarchisation = sg::op_factory::createOperationHierarchisation(*this->myGrid);
        myHierarchisation->doHierarchisation(alpha);
        delete myHierarchisation;
      }
    else
      {
        throw new application_exception("BlackScholesSolver::initCartesianGridWithPayoff : A grid wasn't constructed before!");
      }
  }

  void BlackScholesSolver::initLogTransformedGridWithPayoff(DataVector& alpha, double strike, std::string payoffType)
  {
    double tmp;

    if (this->bGridConstructed)
      {
        for (size_t i = 0; i < this->myGrid->getStorage()->size(); i++)
          {
            std::string coords = this->myGridStorage->get(i)->getCoordsStringBB(*this->myBoundingBox);
            std::stringstream coordsStream(coords);
            double* dblFuncValues = new double[dim];

            for (size_t j = 0; j < this->dim; j++)
              {
                coordsStream >> tmp;

                dblFuncValues[j] = tmp;
              }

            if (payoffType == "std_euro_call")
              {
                tmp = 0.0;
                for (size_t j = 0; j < dim; j++)
                  {
                    tmp += exp(dblFuncValues[j]);
                  }
                alpha[i] = std::max<double>(((tmp/static_cast<double>(dim))-strike), 0.0);
              }
            else if (payoffType == "std_euro_put" || payoffType == "std_amer_put")
              {
                tmp = 0.0;
                for (size_t j = 0; j < dim; j++)
                  {
                    tmp += exp(dblFuncValues[j]);
                  }
                alpha[i] = std::max<double>(strike-((tmp/static_cast<double>(dim))), 0.0);
              }
            else
              {
                throw new application_exception("BlackScholesSolver::initLogTransformedGridWithPayoff : An unknown payoff-type was specified!");
              }

            delete[] dblFuncValues;
          }

        OperationHierarchisation* myHierarchisation = sg::op_factory::createOperationHierarchisation(*this->myGrid);
        myHierarchisation->doHierarchisation(alpha);
        delete myHierarchisation;
      }
    else
      {
        throw new application_exception("BlackScholesSolver::initLogTransformedGridWithPayoff : A grid wasn't constructed before!");
      }
  }

  void BlackScholesSolver::initPATTransformedGridWithPayoff(DataVector& alpha, double strike, std::string payoffType)
  {
    double tmp;

    if (this->bGridConstructed)
      {
        for (size_t i = 0; i < this->myGrid->getStorage()->size(); i++)
          {
            std::string coords = this->myGridStorage->get(i)->getCoordsStringBB(*this->myBoundingBox);
            std::stringstream coordsStream(coords);
            double* dblFuncValues = new double[dim];

            for (size_t j = 0; j < this->dim; j++)
              {
                coordsStream >> tmp;

                dblFuncValues[j] = tmp;
              }

            if (payoffType == "std_euro_call")
              {
                tmp = 0.0;
                for (size_t j = 0; j < dim; j++)
                  {
                    double inner_tmp = 0.0;
                    for (size_t l = 0; l < dim; l++)
                      {
                        inner_tmp += this->eigvec_covar->get(j, l)*dblFuncValues[l];
                      }
                    tmp += exp(inner_tmp);
                  }
                alpha[i] = std::max<double>(((tmp/static_cast<double>(dim))-strike), 0.0);
              }
            else if (payoffType == "std_euro_put" || payoffType == "std_amer_put")
              {
                tmp = 0.0;
                for (size_t j = 0; j < dim; j++)
                  {
                    double inner_tmp = 0.0;
                    for (size_t l = 0; l < dim; l++)
                      {
                        inner_tmp += this->eigvec_covar->get(j, l)*dblFuncValues[l];
                      }
                    tmp += exp(inner_tmp);
                  }
                alpha[i] = std::max<double>(strike-((tmp/static_cast<double>(dim))), 0.0);
              }
            else
              {
                throw new application_exception("BlackScholesSolver::initPATTransformedGridWithPayoff : An unknown payoff-type was specified!");
              }

            delete[] dblFuncValues;
          }

        OperationHierarchisation* myHierarchisation = sg::op_factory::createOperationHierarchisation(*this->myGrid);
        myHierarchisation->doHierarchisation(alpha);
        delete myHierarchisation;
      }
    else
      {
        throw new application_exception("BlackScholesSolver::initPATTransformedGridWithPayoff : A grid wasn't constructed before!");
      }
  }

  double BlackScholesSolver::evalOption(std::vector<double>& eval_point, sg::base::DataVector& alpha)
  {
    std::vector<double> trans_eval = eval_point;

    // apply needed coordinate transformations
    if (this->useLogTransform)
      {
        if (this->usePAT)
          {
            for (size_t i = 0; i < eval_point.size(); i++)
              {
                double trans_point = 0.0;
                for (size_t j = 0; j < this->dim; j++)
                  {
                    trans_point += (this->eigvec_covar->get(j,i)*(log(eval_point[j])));
                  }
                trans_point += (this->current_time*this->mu_hat->get(i));

                trans_eval[i] = trans_point;
              }
          }
        else
          {
            for (size_t i = 0; i < eval_point.size(); i++)
              {
                trans_eval[i] = log(trans_eval[i]);
              }
          }
      }

    sg::base::OperationEval* myEval = sg::op_factory::createOperationEval(*this->myGrid);
    double result = myEval->eval(alpha, trans_eval);
    delete myEval;

    // discounting, if PAT is used
    if (this->usePAT == true && this->payoffType != "std_amer_put")
    {
    	result *= exp(((-1.0)*(this->r*this->current_time)));
    }

    return result;
  }

  void BlackScholesSolver::transformPoint(sg::base::DataVector& point)
  {
    sg::base::DataVector tmp_point(point);

    // apply needed coordinate transformations
    if (this->useLogTransform)
      {
        if (this->usePAT)
          {
            for (size_t i = 0; i < point.getSize(); i++)
              {
                double trans_point = 0.0;
                for (size_t j = 0; j < point.getSize(); j++)
                  {
                    trans_point += (this->eigvec_covar->get(j,i)*(log(point[j])));
                  }
                trans_point += (this->current_time*this->mu_hat->get(i));

                tmp_point[i] = trans_point;
              }
          }
        else
          {
            for (size_t i = 0; i < point.getSize(); i++)
              {
                tmp_point[i] = log(point[i]);
              }
          }
      }

    point = tmp_point;
  }

  void BlackScholesSolver::printSparseGridPAT(sg::base::DataVector& alpha, std::string tfilename, bool bSurplus) const
  {
    DataVector temp(alpha);
    double tmp = 0.0;
    size_t dim = myGrid->getStorage()->dim();
    std::ofstream fileout;

    // Do Dehierarchisation, is specified
    if (bSurplus == false)
      {
        OperationHierarchisation* myHier = sg::op_factory::createOperationHierarchisation(*myGrid);
        myHier->doDehierarchisation(temp);
        delete myHier;
      }

    // Open filehandle
    fileout.open(tfilename.c_str());
    for (size_t i = 0; i < myGrid->getStorage()->size(); i++)
      {
        std::string coords =  myGrid->getStorage()->get(i)->getCoordsStringBB(*myGrid->getBoundingBox());
        std::stringstream coordsStream(coords);

        double* dblFuncValues = new double[dim];

        for (size_t j = 0; j < dim; j++)
          {
            coordsStream >> tmp;
            dblFuncValues[j] = tmp;
          }

        for (size_t l = 0; l < dim; l++)
          {
            double trans_point = 0.0;
            for (size_t j = 0; j < dim; j++)
              {
                trans_point += this->eigvec_covar->get(l,j)*(dblFuncValues[j] - (this->current_time*this->mu_hat->get(j)));
              }
            fileout << exp(trans_point) << " ";
          }

        fileout << temp[i] << std::endl;

        delete[] dblFuncValues;
      }
    fileout.close();
  }

  void BlackScholesSolver::resetSolveTime()
  {
    this->current_time = 0.0;
  }

  size_t BlackScholesSolver::getNeededIterationsToSolve()
  {
    return this->nNeededIterations;
  }

  double BlackScholesSolver::getNeededTimeToSolve()
  {
    return this->dNeededTime;
  }

  size_t BlackScholesSolver::getStartInnerGridSize()
  {
    return this->staInnerGridSize;
  }

  size_t BlackScholesSolver::getFinalInnerGridSize()
  {
    return this->finInnerGridSize;
  }

  size_t BlackScholesSolver::getAverageInnerGridSize()
  {
    return this->avgInnerGridSize;
  }

  void BlackScholesSolver::storeInnerMatrix(DataVector& alpha, std::string tFilename, double timestepsize)
  {
    if (this->bGridConstructed)
      {
        OperationParabolicPDESolverSystemDirichlet* myBSSystem = new BlackScholesParabolicPDESolverSystemEuroAmer(*this->myGrid, alpha, *this->mus, *this->sigmas, *this->rhos, this->r, timestepsize, "ImEul", this->dStrike, this->payoffType, this->useLogTransform, this->useCoarsen, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
        SGppStopwatch* myStopwatch = new SGppStopwatch();

        std::string mtx = "";

        myStopwatch->start();
        std::cout << "Generating matrix in MatrixMarket format..." << std::endl;
        myBSSystem->getInnerMatrix(mtx);

        std::ofstream outfile(tFilename.c_str());
        outfile << mtx;
        outfile.close();
        std::cout << "Generating matrix in MatrixMarket format... DONE! (" << myStopwatch->stop() << " s)" << std::endl << std::endl << std::endl;

        delete myStopwatch;
        delete myBSSystem;
      }
    else
      {
        throw new application_exception("BlackScholesSolver::storeInnerMatrix : A grid wasn't constructed before!");
      }
  }

  void BlackScholesSolver::storeInnerMatrixDiagonal(DataVector& alpha, std::string tFilename, double timestepsize)
  {
    if (this->bGridConstructed)
      {
        OperationParabolicPDESolverSystemDirichlet* myBSSystem = new BlackScholesParabolicPDESolverSystemEuroAmer(*this->myGrid, alpha, *this->mus, *this->sigmas, *this->rhos, this->r, timestepsize, "ImEul", this->dStrike, this->payoffType, this->useLogTransform, this->useCoarsen, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
        SGppStopwatch* myStopwatch = new SGppStopwatch();

        std::string mtx = "";

        myStopwatch->start();
        std::cout << "Generating systemmatrix's diagonal in MatrixMarket format..." << std::endl;
        myBSSystem->getInnerMatrixDiagonal(mtx);

        std::ofstream outfile(tFilename.c_str());
        outfile << mtx;
        outfile.close();
        std::cout << "Generating systemmatrix's diagonal in MatrixMarket format... DONE! (" << myStopwatch->stop() << " s)" << std::endl << std::endl << std::endl;

        delete myStopwatch;
        delete myBSSystem;
      }
    else
      {
        throw new application_exception("BlackScholesSolver::storeInnerMatrix : A grid wasn't constructed before!");
      }
  }

  void BlackScholesSolver::storeInnerMatrixDiagonalRowSum(DataVector& alpha, std::string tFilename, double timestepsize)
  {
    if (this->bGridConstructed)
      {
        OperationParabolicPDESolverSystemDirichlet* myBSSystem = new BlackScholesParabolicPDESolverSystemEuroAmer(*this->myGrid, alpha, *this->mus, *this->sigmas, *this->rhos, this->r, timestepsize, "ImEul", this->dStrike, this->payoffType, this->useLogTransform, this->useCoarsen, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
        SGppStopwatch* myStopwatch = new SGppStopwatch();

        std::string mtx = "";

        myStopwatch->start();
        std::cout << "Generating systemmatrix rowsum as diagonal matrix in MatrixMarket format..." << std::endl;
        myBSSystem->getInnerMatrixDiagonalRowSum(mtx);

        std::ofstream outfile(tFilename.c_str());
        outfile << mtx;
        outfile.close();
        std::cout << "Generating systemmatrix rowsum as diagonal matrix in MatrixMarket format... DONE! (" << myStopwatch->stop() << " s)" << std::endl << std::endl << std::endl;

        delete myStopwatch;
        delete myBSSystem;
      }
    else
      {
        throw new application_exception("BlackScholesSolver::storeInnerMatrix : A grid wasn't constructed before!");
      }
  }

  void BlackScholesSolver::storeInnerRHS(DataVector& alpha, std::string tFilename, double timestepsize)
  {
    if (this->bGridConstructed)
      {
        OperationParabolicPDESolverSystemDirichlet* myBSSystem = new BlackScholesParabolicPDESolverSystemEuroAmer(*this->myGrid, alpha, *this->mus, *this->sigmas, *this->rhos, this->r, timestepsize, "ImEul", this->dStrike, this->payoffType, this->useLogTransform, this->useCoarsen, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
        SGppStopwatch* myStopwatch = new SGppStopwatch();

        myStopwatch->start();
        std::cout << "Exporting inner right-hand-side..." << std::endl;
        DataVector* rhs_inner = myBSSystem->generateRHS();

        size_t nCoefs = rhs_inner->getSize();
        std::ofstream outfile(tFilename.c_str());
        for (size_t i = 0; i < nCoefs; i++)
          {
            outfile << std::scientific << rhs_inner->get(i) << std::endl;
          }
        outfile.close();
        std::cout << "Exporting inner right-hand-side... DONE! (" << myStopwatch->stop() << " s)" << std::endl << std::endl << std::endl;

        delete myStopwatch;
        delete myBSSystem;
      }
    else
      {
        throw new application_exception("BlackScholesSolver::storeInnerMatrix : A grid wasn't constructed before!");
      }
  }

  void BlackScholesSolver::storeInnerSolution(DataVector& alpha, size_t numTimesteps, double timestepsize, size_t maxCGIterations, double epsilonCG, std::string tFilename)
  {
    if (this->bGridConstructed)
      {
        Euler* myEuler = new Euler("ImEul", numTimesteps, timestepsize, false, 0, myScreen);
        BiCGStab* myCG = new BiCGStab(maxCGIterations, epsilonCG);
        OperationParabolicPDESolverSystemDirichlet* myBSSystem = new BlackScholesParabolicPDESolverSystemEuroAmer(*this->myGrid, alpha, *this->mus, *this->sigmas, *this->rhos, this->r, timestepsize, "ImEul", this->dStrike, this->payoffType, this->useLogTransform, this->useCoarsen, this->coarsenThreshold, this->adaptSolveMode, this->numCoarsenPoints, this->refineThreshold, this->refineMode, this->refineMaxLevel);
        SGppStopwatch* myStopwatch = new SGppStopwatch();

        myStopwatch->start();
        std::cout << "Exporting inner solution..." << std::endl;
        myEuler->solve(*myCG, *myBSSystem, false);

        DataVector* alpha_solve = myBSSystem->getGridCoefficientsForCG();
        size_t nCoefs = alpha_solve->getSize();
        std::ofstream outfile(tFilename.c_str());
        for (size_t i = 0; i < nCoefs; i++)
          {
            outfile << std::scientific << alpha_solve->get(i) << std::endl;
          }
        outfile.close();

        std::cout << "Exporting inner solution... DONE!" << std::endl;

        delete myStopwatch;
        delete myBSSystem;
        delete myCG;
        delete myEuler;
      }
    else
      {
        throw new application_exception("BlackScholesSolver::solveImplicitEuler : A grid wasn't constructed before!");
      }

  }

}
}
