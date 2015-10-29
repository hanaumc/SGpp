// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#ifndef SGPP_OPTIMIZATION_OPTIMIZER_UNCONSTRAINED_MULTISTART_HPP
#define SGPP_OPTIMIZATION_OPTIMIZER_UNCONSTRAINED_MULTISTART_HPP

#include <sgpp/globaldef.hpp>

#include <sgpp/optimization/optimizer/unconstrained/UnconstrainedOptimizer.hpp>
#include <sgpp/optimization/optimizer/unconstrained/NelderMead.hpp>

namespace SGPP {
  namespace optimization {
    namespace optimizer {

      /**
       * Meta optimization algorithm calling local algorithm multiple times.
       * MultiStart generates a random population of a given number of
       * starting points, and then runs a local optimization algorithm
       * for each of the starting point.
       * The best point wins.
       */
      class MultiStart : public UnconstrainedOptimizer {
        public:
          /// default maximal number of function evaluations
          static const size_t DEFAULT_MAX_FCN_EVAL_COUNT = 1000;

          /**
           * Constructor.
           * By default, Nelder-Mead is used as optimization algorithm.
           *
           * @param f               objective function
           * @param maxFcnEvalCount maximal number of function evaluations
           * @param populationSize  number of individual points
           *                        (default: \f$\min(10d, 100)\f$)
           */
          MultiStart(ObjectiveFunction& f,
                     size_t maxFcnEvalCount = DEFAULT_MAX_FCN_EVAL_COUNT,
                     size_t populationSize = 0);

          /**
           * Constructor with custom optimization algorithm.
           * The current values of the optimizer's N and starting point
           * properties will not be used.
           *
           * @param optimizer        optimization algorithm and
           *                         objective function
           * @param maxFcnEvalCount  maximal number of function evaluations
           * @param populationSize   number of individual points
           *                         (default: \f$\min(10d, 100)\f$)
           */
          MultiStart(UnconstrainedOptimizer& optimizer,
                     size_t maxFcnEvalCount = DEFAULT_MAX_FCN_EVAL_COUNT,
                     size_t populationSize = 0);

          /**
           * @param[out] xOpt optimal point
           * @return          optimal objective function value
           */
          float_t optimize(base::DataVector& xOpt);

          /**
           * @return                  number of individual points
           */
          size_t getPopulationSize() const;

          /**
           * @param populationSize    number of individual points
           */
          void setPopulationSize(size_t populationSize);

        protected:
          /// default optimization algorithm
          NelderMead defaultOptimizer;
          /// optimization algorithm
          UnconstrainedOptimizer& optimizer;
          /// number of individual points
          size_t populationSize;

          /**
           * Initializes populationSize.
           *
           * @param populationSize     number of individual points
           *                           (zero to use default value)
           */
          void initialize(size_t populationSize);
      };

    }
  }
}

#endif /* SGPP_OPTIMIZATION_OPTIMIZER_UNCONSTRAINED_MULTISTART_HPP */