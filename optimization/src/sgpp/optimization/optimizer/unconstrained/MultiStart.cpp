// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include <sgpp/globaldef.hpp>

#include <sgpp/optimization/optimizer/unconstrained/MultiStart.hpp>
#include <sgpp/optimization/tools/Printer.hpp>
#include <sgpp/optimization/tools/RandomNumberGenerator.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>

namespace SGPP {
  namespace optimization {
    namespace optimizer {

      MultiStart::MultiStart(ObjectiveFunction& f,
                             size_t maxFcnEvalCount,
                             size_t populationSize) :
        UnconstrainedOptimizer(f, maxFcnEvalCount),
        defaultOptimizer(NelderMead(f)),
        optimizer(defaultOptimizer) {
        initialize(populationSize);
      }

      MultiStart::MultiStart(UnconstrainedOptimizer& optimizer,
                             size_t maxFcnEvalCount,
                             size_t populationSize) :
        UnconstrainedOptimizer(optimizer.getObjectiveFunction(), maxFcnEvalCount),
        defaultOptimizer(NelderMead(f)),
        optimizer(optimizer) {
        initialize(populationSize);
      }

      void MultiStart::initialize(size_t populationSize) {
        this->populationSize = (populationSize > 0) ? populationSize :
                               std::min(10 * f.getDimension(),
                                        static_cast<size_t>(100));
      }

      float_t MultiStart::optimize(base::DataVector& xOpt) {
        printer.printStatusBegin("Optimizing (random search)...");

        const size_t d = f.getDimension();
        std::vector<base::DataVector> x0(populationSize, base::DataVector(d));
        std::vector<size_t> roundN(populationSize, 0);
        size_t remainingN = N;

        // split the number of function evaluations evenly up for all points,
        // generate pseudorandom starting points
        for (size_t k = 0; k < populationSize; k++) {
          roundN[k] = static_cast<size_t>(
                        std::ceil(static_cast<float_t>(remainingN) /
                                  static_cast<float_t>(populationSize - k)));
          remainingN -= roundN[k];

          for (size_t t = 0; t < d; t++) {
            x0[k][t] = randomNumberGenerator.getUniformRN();
          }
        }

        float_t fOpt = INFINITY;

        /*printer.disableStatusPrinting();

        #pragma omp parallel shared(d, x0, roundN, xOpt, fOpt, printer) \
        default(none)
        {
          Optimizer* curOptimizerPtr = &optimizer;
        #ifdef _OPENMP
          std::unique_ptr<Optimizer> curOptimizer;

          if (omp_get_max_threads() > 1) {
            optimizer.clone(curOptimizer);
            curOptimizerPtr = curOptimizer.get();
          }

        #endif ** _OPENMP **

          base::DataVector curXOpt(d);
          float_t curFOpt;

          #pragma omp for ordered schedule(dynamic)

          for (size_t k = 0; k < populationSize; k++) {
            // optimize with k-th starting point
            curOptimizerPtr->setStartingPoint(x0[k]);
            curOptimizerPtr->setN(roundN[k]);
            curOptimizerPtr->optimize(curXOpt);

            curFOpt = curOptimizerPtr->getObjectiveFunction().eval(curXOpt);

            #pragma omp critical
            {
              if (curFOpt < fOpt) {
                // this point is the best so far
                fOpt = curFOpt;
                xOpt = curXOpt;
              }
            }

            // status printing
            #pragma omp ordered
            {
              char str[10];
              snprintf(str, 10, "%.1f%%",
                       static_cast<float_t>(k) /
                       static_cast<float_t>(populationSize) * 100.0);
              printer.getMutex().lock();
              printer.enableStatusPrinting();
              printer.printStatusUpdate(std::string(str) +
                                        ", f(x) = " + std::to_string(fOpt));
              printer.disableStatusPrinting();
              printer.getMutex().unlock();
            }
          }
        }

        printer.enableStatusPrinting();*/

        base::DataVector curXOpt(d);
        float_t curFOpt;

        xOpt.resize(d);

        // temporarily save x0 and N (will be overwritten by the loop)
        const base::DataVector tmpX0(optimizer.getStartingPoint());
        const size_t tmpN = optimizer.getN();
        const bool statusPrintingEnabled = printer.isStatusPrintingEnabled();

        for (size_t k = 0; k < populationSize; k++) {
          // optimize with k-th starting point
          optimizer.setStartingPoint(x0[k]);
          optimizer.setN(roundN[k]);

          if (statusPrintingEnabled) {
            printer.disableStatusPrinting();
          }

          optimizer.optimize(curXOpt);

          if (statusPrintingEnabled) {
            printer.enableStatusPrinting();
          }

          curFOpt = f.eval(curXOpt);

          if (curFOpt < fOpt) {
            // this point is the best so far
            fOpt = curFOpt;
            xOpt = curXOpt;
          }

          // status printing
          {
            char str[10];
            snprintf(str, 10, "%.1f%%",
                     static_cast<float_t>(k) /
                     static_cast<float_t>(populationSize) * 100.0);
            printer.printStatusUpdate(std::string(str) +
                                      ", f(x) = " + std::to_string(fOpt));
          }
        }

        // set x0 and N to initial values
        optimizer.setStartingPoint(tmpX0);
        optimizer.setN(tmpN);

        printer.printStatusUpdate("100.0%, f(x) = " + std::to_string(fOpt));
        printer.printStatusEnd();

        return fOpt;
      }

      size_t MultiStart::getPopulationSize() const {
        return populationSize;
      }

      void MultiStart::setPopulationSize(size_t populationSize) {
        this->populationSize = populationSize;
      }

    }
  }
}
