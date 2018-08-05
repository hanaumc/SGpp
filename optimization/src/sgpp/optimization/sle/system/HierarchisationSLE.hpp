// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#ifndef SGPP_OPTIMIZATION_SLE_SYSTEM_HIERARCHISATIONSLE_HPP
#define SGPP_OPTIMIZATION_SLE_SYSTEM_HIERARCHISATIONSLE_HPP

#include <sgpp/globaldef.hpp>

#include <sgpp/optimization/sle/system/CloneableSLE.hpp>
#include <sgpp/base/grid/Grid.hpp>
#include <sgpp/base/grid/GridStorage.hpp>

#include <sgpp/base/operation/hash/common/basis/BsplineBasis.hpp>
#include <sgpp/base/operation/hash/common/basis/BsplineBoundaryBasis.hpp>
#include <sgpp/base/operation/hash/common/basis/BsplineClenshawCurtisBasis.hpp>
#include <sgpp/base/operation/hash/common/basis/BsplineModifiedBasis.hpp>
#include <sgpp/base/operation/hash/common/basis/BsplineModifiedClenshawCurtisBasis.hpp>
#include <sgpp/base/operation/hash/common/basis/FundamentalNotAKnotSplineBasis.hpp>
#include <sgpp/base/operation/hash/common/basis/FundamentalSplineBasis.hpp>
#include <sgpp/base/operation/hash/common/basis/FundamentalSplineModifiedBasis.hpp>
#include <sgpp/base/operation/hash/common/basis/LagrangeNotAKnotSplineBasis.hpp>
#include <sgpp/base/operation/hash/common/basis/LagrangeNotAKnotSplineModifiedBasis.hpp>
#include <sgpp/base/operation/hash/common/basis/LagrangeSplineBasis.hpp>
#include <sgpp/base/operation/hash/common/basis/LinearBasis.hpp>
#include <sgpp/base/operation/hash/common/basis/LinearBoundaryBasis.hpp>
#include <sgpp/base/operation/hash/common/basis/LinearClenshawCurtisBasis.hpp>
#include <sgpp/base/operation/hash/common/basis/LinearModifiedBasis.hpp>
#include <sgpp/base/operation/hash/common/basis/NaturalBsplineBasis.hpp>
#include <sgpp/base/operation/hash/common/basis/NotAKnotBsplineBasis.hpp>
#include <sgpp/base/operation/hash/common/basis/NotAKnotBsplineModifiedBasis.hpp>
#include <sgpp/base/operation/hash/common/basis/WaveletBasis.hpp>
#include <sgpp/base/operation/hash/common/basis/WaveletBoundaryBasis.hpp>
#include <sgpp/base/operation/hash/common/basis/WaveletModifiedBasis.hpp>

#include <sgpp/base/grid/type/LinearClenshawCurtisGrid.hpp>
#include <sgpp/base/grid/type/BsplineGrid.hpp>
#include <sgpp/base/grid/type/BsplineBoundaryGrid.hpp>
#include <sgpp/base/grid/type/BsplineClenshawCurtisGrid.hpp>
#include <sgpp/base/grid/type/FundamentalNotAKnotSplineBoundaryGrid.hpp>
#include <sgpp/base/grid/type/FundamentalSplineGrid.hpp>
#include <sgpp/base/grid/type/FundamentalSplineBoundaryGrid.hpp>
#include <sgpp/base/grid/type/LagrangeNotAKnotSplineBoundaryGrid.hpp>
#include <sgpp/base/grid/type/LagrangeSplineBoundaryGrid.hpp>
#include <sgpp/base/grid/type/ModBsplineGrid.hpp>
#include <sgpp/base/grid/type/ModBsplineClenshawCurtisGrid.hpp>
#include <sgpp/base/grid/type/ModFundamentalSplineGrid.hpp>
#include <sgpp/base/grid/type/ModLagrangeNotAKnotSplineGrid.hpp>
#include <sgpp/base/grid/type/ModNotAKnotBsplineGrid.hpp>
#include <sgpp/base/grid/type/NaturalBsplineBoundaryGrid.hpp>
#include <sgpp/base/grid/type/NotAKnotBsplineBoundaryGrid.hpp>

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <memory>

namespace sgpp {
namespace optimization {

/**
 * Linear system of the hierarchization in a sparse grid.
 */
class HierarchisationSLE : public CloneableSLE {
 public:
  /**
   * Constructor.
   * Do not destruct the grid before this object!
   *
   * @param grid              sparse grid
   */
  explicit HierarchisationSLE(base::Grid& grid) : HierarchisationSLE(grid, grid.getStorage()) {}

  /**
   * Constructor.
   * Do not destruct the grid before this object!
   *
   * @param grid              sparse grid
   * @param gridStorage       custom grid storage (use basis function
   *                          according to grid, but use another set of
   *                          grid points according to gridStorage)
   */
  HierarchisationSLE(base::Grid& grid, base::GridStorage& gridStorage)
      : CloneableSLE(), grid(grid), gridStorage(gridStorage), basisType(INVALID) {
    // initialize the correct basis (according to the grid)
    if (grid.getType() == base::GridType::Bspline) {
      bsplineBasis = std::unique_ptr<base::SBsplineBase>(
          new base::SBsplineBase(dynamic_cast<base::BsplineGrid&>(grid).getDegree()));
      basisType = BSPLINE;
    } else if (grid.getType() == base::GridType::BsplineBoundary) {
      bsplineBoundaryBasis =
          std::unique_ptr<base::SBsplineBoundaryBase>(new base::SBsplineBoundaryBase(
              dynamic_cast<base::BsplineBoundaryGrid&>(grid).getDegree()));
      basisType = BSPLINE_BOUNDARY;
    } else if (grid.getType() == base::GridType::BsplineClenshawCurtis) {
      bsplineClenshawCurtisBasis =
          std::unique_ptr<base::SBsplineClenshawCurtisBase>(new base::SBsplineClenshawCurtisBase(
              dynamic_cast<base::BsplineClenshawCurtisGrid&>(grid).getDegree()));
      basisType = BSPLINE_CLENSHAW_CURTIS;
    } else if (grid.getType() == base::GridType::ModBspline) {
      modBsplineBasis = std::unique_ptr<base::SBsplineModifiedBase>(
          new base::SBsplineModifiedBase(dynamic_cast<base::ModBsplineGrid&>(grid).getDegree()));
      basisType = BSPLINE_MODIFIED;
    } else if (grid.getType() == base::GridType::ModBsplineClenshawCurtis) {
      modBsplineClenshawCurtisBasis = std::unique_ptr<base::SBsplineModifiedClenshawCurtisBase>(
          new base::SBsplineModifiedClenshawCurtisBase(
              dynamic_cast<base::ModBsplineClenshawCurtisGrid&>(grid).getDegree()));
      basisType = BSPLINE_MODIFIED_CLENSHAW_CURTIS;
    } else if (grid.getType() == base::GridType::FundamentalNotAKnotSplineBoundary) {
      fundamentalNotAKnotSplineBasis =
          std::unique_ptr<base::SFundamentalNotAKnotSplineBase>(
            new base::SFundamentalNotAKnotSplineBase(
              dynamic_cast<base::FundamentalNotAKnotSplineBoundaryGrid&>(grid).getDegree()));
      basisType = FUNDAMENTAL_NOTAKNOT_SPLINE;
    } else if (grid.getType() == base::GridType::FundamentalSpline) {
      fundamentalSplineBasis =
          std::unique_ptr<base::SFundamentalSplineBase>(new base::SFundamentalSplineBase(
              dynamic_cast<base::FundamentalSplineGrid&>(grid).getDegree()));
      basisType = FUNDAMENTAL_SPLINE;
    } else if (grid.getType() == base::GridType::FundamentalSplineBoundary) {
      fundamentalSplineBasis =
          std::unique_ptr<base::SFundamentalSplineBase>(new base::SFundamentalSplineBase(
              dynamic_cast<base::FundamentalSplineBoundaryGrid&>(grid).getDegree()));
      basisType = FUNDAMENTAL_SPLINE;
    } else if (grid.getType() == base::GridType::ModFundamentalSpline) {
      modFundamentalSplineBasis = std::unique_ptr<base::SFundamentalSplineModifiedBase>(
          new base::SFundamentalSplineModifiedBase(
              dynamic_cast<base::ModFundamentalSplineGrid&>(grid).getDegree()));
      basisType = FUNDAMENTAL_SPLINE_MODIFIED;
    } else if (grid.getType() == base::GridType::LagrangeNotAKnotSplineBoundary) {
      lagrangeNotAKnotSplineBasis =
          std::unique_ptr<base::SLagrangeNotAKnotSplineBase>(new base::SLagrangeNotAKnotSplineBase(
              dynamic_cast<base::LagrangeNotAKnotSplineBoundaryGrid&>(grid).getDegree()));
      basisType = LAGRANGE_NOTAKNOT_SPLINE;
    } else if (grid.getType() == base::GridType::ModLagrangeNotAKnotSpline) {
      modLagrangeNotAKnotSplineBasis =
          std::unique_ptr<base::SLagrangeNotAKnotSplineModifiedBase>(
              new base::SLagrangeNotAKnotSplineModifiedBase(
              dynamic_cast<base::ModLagrangeNotAKnotSplineGrid&>(grid).getDegree()));
      basisType = LAGRANGE_NOTAKNOT_SPLINE_MODIFIED;
    } else if (grid.getType() == base::GridType::LagrangeSplineBoundary) {
      lagrangeSplineBasis =
          std::unique_ptr<base::SLagrangeSplineBase>(new base::SLagrangeSplineBase(
              dynamic_cast<base::LagrangeSplineBoundaryGrid&>(grid).getDegree()));
      basisType = LAGRANGE_SPLINE;
    } else if (grid.getType() == base::GridType::Linear) {
      linearBasis = std::unique_ptr<base::SLinearBase>(new base::SLinearBase());
      basisType = LINEAR;
    } else if (grid.getType() == base::GridType::LinearBoundary) {
      linearL0BoundaryBasis =
          std::unique_ptr<base::SLinearBoundaryBase>(new base::SLinearBoundaryBase());
      basisType = LINEAR_BOUNDARY;
    } else if (grid.getType() == base::GridType::LinearClenshawCurtis) {
      linearClenshawCurtisBasis =
          std::unique_ptr<base::SLinearClenshawCurtisBase>(new base::SLinearClenshawCurtisBase());
      basisType = LINEAR_CLENSHAW_CURTIS;
    } else if (grid.getType() == base::GridType::ModLinear) {
      modLinearBasis = std::unique_ptr<base::SLinearModifiedBase>(new base::SLinearModifiedBase());
      basisType = LINEAR_MODIFIED;
    } else if (grid.getType() == base::GridType::NaturalBsplineBoundary) {
      naturalBsplineBasis =
          std::unique_ptr<base::SNaturalBsplineBase>(new base::SNaturalBsplineBase(
              dynamic_cast<base::NaturalBsplineBoundaryGrid&>(grid).getDegree()));
      basisType = NATURAL_BSPLINE;
    } else if (grid.getType() == base::GridType::NotAKnotBsplineBoundary) {
      notAKnotBsplineBasis =
          std::unique_ptr<base::SNotAKnotBsplineBase>(new base::SNotAKnotBsplineBase(
              dynamic_cast<base::NotAKnotBsplineBoundaryGrid&>(grid).getDegree()));
      basisType = NOTAKNOT_BSPLINE;
    } else if (grid.getType() == base::GridType::ModNotAKnotBspline) {
      modNotAKnotBsplineBasis =
          std::unique_ptr<base::SNotAKnotBsplineModifiedBase>(
              new base::SNotAKnotBsplineModifiedBase(
              dynamic_cast<base::ModNotAKnotBsplineGrid&>(grid).getDegree()));
      basisType = NOTAKNOT_BSPLINE_MODIFIED;
    } else if (grid.getType() == base::GridType::Wavelet) {
      waveletBasis = std::unique_ptr<base::SWaveletBase>(new base::SWaveletBase());
      basisType = WAVELET;
    } else if (grid.getType() == base::GridType::WaveletBoundary) {
      waveletBoundaryBasis =
          std::unique_ptr<base::SWaveletBoundaryBase>(new base::SWaveletBoundaryBase());
      basisType = WAVELET_BOUNDARY;
    } else if (grid.getType() == base::GridType::ModWavelet) {
      modWaveletBasis =
          std::unique_ptr<base::SWaveletModifiedBase>(new base::SWaveletModifiedBase());
      basisType = WAVELET_MODIFIED;
    } else {
      throw std::invalid_argument("Grid type not supported.");
    }
  }

  /**
   * @param i     row index
   * @param j     column index
   * @return      whether the i-th grid point lies in the support of
   *              the j-th basis function
   */
  inline bool isMatrixEntryNonZero(size_t i, size_t j) override {
    return (evalBasisFunctionAtGridPoint(j, i) != 0.0);
  }

  /**
   * @param i     row index
   * @param j     column index
   * @return      value of the j-th basis function at the i-th grid point
   */
  inline double getMatrixEntry(size_t i, size_t j) override {
    return evalBasisFunctionAtGridPoint(j, i);
  }

  /**
   * @return          sparse grid
   */
  base::Grid& getGrid() { return grid; }

  /**
   * @return              grid storage
   */
  base::GridStorage& getGridStorage() { return gridStorage; }

  size_t getDimension() const override { return gridStorage.getSize(); }

  /**
   * @param[out] clone pointer to cloned object
   */
  void clone(std::unique_ptr<CloneableSLE>& clone) const override {
    clone = std::unique_ptr<CloneableSLE>(new HierarchisationSLE(grid, gridStorage));
  }

 protected:
  /// sparse grid
  base::Grid& grid;
  /// grid storage
  base::GridStorage& gridStorage;

  /// B-spline basis
  std::unique_ptr<base::SBsplineBase> bsplineBasis;
  /// B-spline boundary basis
  std::unique_ptr<base::SBsplineBoundaryBase> bsplineBoundaryBasis;
  /// B-spline Clenshaw-Curtis basis
  std::unique_ptr<base::SBsplineClenshawCurtisBase> bsplineClenshawCurtisBasis;
  /// modified B-spline basis
  std::unique_ptr<base::SBsplineModifiedBase> modBsplineBasis;
  /// modified B-spline Clenshaw-Curtis basis
  std::unique_ptr<base::SBsplineModifiedClenshawCurtisBase> modBsplineClenshawCurtisBasis;
  /// fundamental not-a-knot spline basis
  std::unique_ptr<base::SFundamentalNotAKnotSplineBase> fundamentalNotAKnotSplineBasis;
  /// fundamental spline basis
  std::unique_ptr<base::SFundamentalSplineBase> fundamentalSplineBasis;
  /// modified fundamental spline basis
  std::unique_ptr<base::SFundamentalSplineModifiedBase> modFundamentalSplineBasis;
  /// Lagrange not-a-knot spline basis
  std::unique_ptr<base::SLagrangeNotAKnotSplineBase> lagrangeNotAKnotSplineBasis;
  /// modified Lagrange not-a-knot spline basis
  std::unique_ptr<base::SLagrangeNotAKnotSplineModifiedBase> modLagrangeNotAKnotSplineBasis;
  /// Lagrange spline basis
  std::unique_ptr<base::SLagrangeSplineBase> lagrangeSplineBasis;
  /// linear basis
  std::unique_ptr<base::SLinearBase> linearBasis;
  /// linear boundary basis
  std::unique_ptr<base::SLinearBoundaryBase> linearL0BoundaryBasis;
  /// linear Clenshaw-Curtis basis
  std::unique_ptr<base::SLinearClenshawCurtisBase> linearClenshawCurtisBasis;
  /// modified linear basis
  std::unique_ptr<base::SLinearModifiedBase> modLinearBasis;
  /// natural B-spline basis
  std::unique_ptr<base::SNaturalBsplineBase> naturalBsplineBasis;
  /// not-a-knot B-spline basis
  std::unique_ptr<base::SNotAKnotBsplineBase> notAKnotBsplineBasis;
  /// modified not-a-knot B-spline basis
  std::unique_ptr<base::SNotAKnotBsplineModifiedBase> modNotAKnotBsplineBasis;
  /// wavelet basis
  std::unique_ptr<base::SWaveletBase> waveletBasis;
  /// wavelet boundary basis
  std::unique_ptr<base::SWaveletBoundaryBase> waveletBoundaryBasis;
  /// modified wavelet basis
  std::unique_ptr<base::SWaveletModifiedBase> modWaveletBasis;

  /// type of grid/basis functions
  enum {
    INVALID,
    BSPLINE,
    BSPLINE_BOUNDARY,
    BSPLINE_CLENSHAW_CURTIS,
    BSPLINE_MODIFIED,
    BSPLINE_MODIFIED_CLENSHAW_CURTIS,
    FUNDAMENTAL_NOTAKNOT_SPLINE,
    FUNDAMENTAL_SPLINE,
    FUNDAMENTAL_SPLINE_MODIFIED,
    LAGRANGE_NOTAKNOT_SPLINE,
    LAGRANGE_NOTAKNOT_SPLINE_MODIFIED,
    LAGRANGE_SPLINE,
    LINEAR,
    LINEAR_BOUNDARY,
    LINEAR_CLENSHAW_CURTIS,
    LINEAR_MODIFIED,
    NATURAL_BSPLINE,
    NOTAKNOT_BSPLINE,
    NOTAKNOT_BSPLINE_MODIFIED,
    WAVELET,
    WAVELET_BOUNDARY,
    WAVELET_MODIFIED
  } basisType;

  /**
   * @param basisI    basis function index
   * @param pointJ    grid point index
   * @return          value of the basisI-th basis function at the
   *                  pointJ-th grid point
   */
  inline double evalBasisFunctionAtGridPoint(size_t basisI, size_t pointJ) {
    if (basisType == BSPLINE) {
      return evalBsplineFunctionAtGridPoint(basisI, pointJ);
    } else if (basisType == BSPLINE_BOUNDARY) {
      return evalBsplineBoundaryFunctionAtGridPoint(basisI, pointJ);
    } else if (basisType == BSPLINE_CLENSHAW_CURTIS) {
      return evalBsplineClenshawCurtisFunctionAtGridPoint(basisI, pointJ);
    } else if (basisType == BSPLINE_MODIFIED) {
      return evalBsplineModifiedFunctionAtGridPoint(basisI, pointJ);
    } else if (basisType == BSPLINE_MODIFIED_CLENSHAW_CURTIS) {
      return evalBsplineModifiedClenshawCurtisFunctionAtGridPoint(basisI, pointJ);
    } else if (basisType == FUNDAMENTAL_NOTAKNOT_SPLINE) {
      return evalFundamentalNotAKnotSplineFunctionAtGridPoint(basisI, pointJ);
    } else if (basisType == FUNDAMENTAL_SPLINE) {
      return evalFundamentalSplineFunctionAtGridPoint(basisI, pointJ);
    } else if (basisType == FUNDAMENTAL_SPLINE_MODIFIED) {
      return evalFundamentalSplineModifiedFunctionAtGridPoint(basisI, pointJ);
    } else if (basisType == LAGRANGE_NOTAKNOT_SPLINE) {
      return evalLagrangeNotAKnotSplineFunctionAtGridPoint(basisI, pointJ);
    } else if (basisType == LAGRANGE_NOTAKNOT_SPLINE_MODIFIED) {
      return evalLagrangeNotAKnotSplineModifiedFunctionAtGridPoint(basisI, pointJ);
    } else if (basisType == LAGRANGE_SPLINE) {
      return evalLagrangeSplineFunctionAtGridPoint(basisI, pointJ);
    } else if (basisType == LINEAR) {
      return evalLinearFunctionAtGridPoint(basisI, pointJ);
    } else if (basisType == LINEAR_BOUNDARY) {
      return evalLinearBoundaryFunctionAtGridPoint(basisI, pointJ);
    } else if (basisType == LINEAR_CLENSHAW_CURTIS) {
      return evalLinearClenshawCurtisFunctionAtGridPoint(basisI, pointJ);
    } else if (basisType == LINEAR_MODIFIED) {
      return evalLinearModifiedFunctionAtGridPoint(basisI, pointJ);
    } else if (basisType == NATURAL_BSPLINE) {
      return evalNaturalBsplineFunctionAtGridPoint(basisI, pointJ);
    } else if (basisType == NOTAKNOT_BSPLINE) {
      return evalNotAKnotBsplineFunctionAtGridPoint(basisI, pointJ);
    } else if (basisType == NOTAKNOT_BSPLINE_MODIFIED) {
      return evalNotAKnotBsplineModifiedFunctionAtGridPoint(basisI, pointJ);
    } else if (basisType == WAVELET) {
      return evalWaveletFunctionAtGridPoint(basisI, pointJ);
    } else if (basisType == WAVELET_BOUNDARY) {
      return evalWaveletBoundaryFunctionAtGridPoint(basisI, pointJ);
    } else if (basisType == WAVELET_MODIFIED) {
      return evalWaveletModifiedFunctionAtGridPoint(basisI, pointJ);
    } else {
      return 0.0;
    }
  }

  /**
   * @param basisI    basis function index
   * @param pointJ    grid point index
   * @return          value of the basisI-th B-spline basis function
   *                  at the pointJ-th grid point
   */
  inline double evalBsplineFunctionAtGridPoint(size_t basisI, size_t pointJ) {
    const base::GridPoint& gpBasis = gridStorage[basisI];
    const base::GridPoint& gpPoint = gridStorage[pointJ];
    double result = 1.0;

    for (size_t t = 0; t < gridStorage.getDimension(); t++) {
      const double result1d = bsplineBasis->eval(
          gpBasis.getLevel(t), gpBasis.getIndex(t), gridStorage.getCoordinate(gpPoint, t));

      if (result1d == 0.0) {
        return 0.0;
      }

      result *= result1d;
    }

    return result;
  }

  /**
   * @param basisI    basis function index
   * @param pointJ    grid point index
   * @return          value of the basisI-th B-spline boundary
   *                  basis function at the pointJ-th grid point
   */
  inline double evalBsplineBoundaryFunctionAtGridPoint(size_t basisI, size_t pointJ) {
    const base::GridPoint& gpBasis = gridStorage[basisI];
    const base::GridPoint& gpPoint = gridStorage[pointJ];
    double result = 1.0;

    for (size_t t = 0; t < gridStorage.getDimension(); t++) {
      const double result1d = bsplineBoundaryBasis->eval(
          gpBasis.getLevel(t), gpBasis.getIndex(t), gridStorage.getCoordinate(gpPoint, t));

      if (result1d == 0.0) {
        return 0.0;
      }

      result *= result1d;
    }

    return result;
  }

  /**
   * @param basisI    basis function index
   * @param pointJ    grid point index
   * @return          value of the basisI-th B-spline Clenshaw-Curtis
   *                  basis function at the pointJ-th grid point
   */
  inline double evalBsplineClenshawCurtisFunctionAtGridPoint(size_t basisI, size_t pointJ) {
    const base::GridPoint& gpBasis = gridStorage[basisI];
    const base::GridPoint& gpPoint = gridStorage[pointJ];
    double result = 1.0;

    for (size_t t = 0; t < gridStorage.getDimension(); t++) {
      const double result1d = bsplineClenshawCurtisBasis->eval(
          gpBasis.getLevel(t), gpBasis.getIndex(t), gridStorage.getCoordinate(gpPoint, t));

      if (result1d == 0.0) {
        return 0.0;
      }

      result *= result1d;
    }

    return result;
  }

  /**
   * @param basisI    basis function index
   * @param pointJ    grid point index
   * @return          value of the basisI-th modified B-spline
   *                  basis function at the pointJ-th grid point
   */
  inline double evalBsplineModifiedFunctionAtGridPoint(size_t basisI, size_t pointJ) {
    const base::GridPoint& gpBasis = gridStorage[basisI];
    const base::GridPoint& gpPoint = gridStorage[pointJ];
    double result = 1.0;

    for (size_t t = 0; t < gridStorage.getDimension(); t++) {
      const double result1d = modBsplineBasis->eval(
          gpBasis.getLevel(t), gpBasis.getIndex(t), gridStorage.getCoordinate(gpPoint, t));

      if (result1d == 0.0) {
        return 0.0;
      }

      result *= result1d;
    }

    return result;
  }

  /**
   * @param basisI    basis function index
   * @param pointJ    grid point index
   * @return          value of the basisI-th modified Clenshaw-Curtis
   *                  B-spline basis function at the pointJ-th grid point
   */
  inline double evalBsplineModifiedClenshawCurtisFunctionAtGridPoint(size_t basisI,
                                                                      size_t pointJ) {
    const base::GridPoint& gpBasis = gridStorage[basisI];
    const base::GridPoint& gpPoint = gridStorage[pointJ];
    double result = 1.0;

    for (size_t t = 0; t < gridStorage.getDimension(); t++) {
      const double result1d = modBsplineClenshawCurtisBasis->eval(
          gpBasis.getLevel(t), gpBasis.getIndex(t), gridStorage.getCoordinate(gpPoint, t));

      if (result1d == 0.0) {
        return 0.0;
      }

      result *= result1d;
    }

    return result;
  }

  /**
   * @param basisI    basis function index
   * @param pointJ    grid point index
   * @return          value of the basisI-th fundamental not-a-knot spline basis
   *                  function at the pointJ-th grid point
   */
  inline double evalFundamentalNotAKnotSplineFunctionAtGridPoint(size_t basisI, size_t pointJ) {
    const base::GridPoint& gpBasis = gridStorage[basisI];
    const base::GridPoint& gpPoint = gridStorage[pointJ];
    double result = 1.0;

    for (size_t t = 0; t < gridStorage.getDimension(); t++) {
      if (gpPoint.getLevel(t) < gpBasis.getLevel(t)) {
        return 0.0;
      } else if (gpPoint.getLevel(t) == gpBasis.getLevel(t)) {
        if (gpPoint.getIndex(t) != gpBasis.getIndex(t)) {
          return 0.0;
        }
      } else {
        const double result1d = fundamentalNotAKnotSplineBasis->eval(
            gpBasis.getLevel(t), gpBasis.getIndex(t), gridStorage.getCoordinate(gpPoint, t));

        if (result1d == 0.0) {
          return 0.0;
        }

        result *= result1d;
      }
    }

    return result;
  }

  /**
   * @param basisI    basis function index
   * @param pointJ    grid point index
   * @return          value of the basisI-th fundamental spline basis
   *                  function at the pointJ-th grid point
   */
  inline double evalFundamentalSplineFunctionAtGridPoint(size_t basisI, size_t pointJ) {
    const base::GridPoint& gpBasis = gridStorage[basisI];
    const base::GridPoint& gpPoint = gridStorage[pointJ];
    double result = 1.0;

    for (size_t t = 0; t < gridStorage.getDimension(); t++) {
      if (gpPoint.getLevel(t) < gpBasis.getLevel(t)) {
        return 0.0;
      } else if (gpPoint.getLevel(t) == gpBasis.getLevel(t)) {
        if (gpPoint.getIndex(t) != gpBasis.getIndex(t)) {
          return 0.0;
        }
      } else {
        const double result1d = fundamentalSplineBasis->eval(
            gpBasis.getLevel(t), gpBasis.getIndex(t), gridStorage.getCoordinate(gpPoint, t));

        if (result1d == 0.0) {
          return 0.0;
        }

        result *= result1d;
      }
    }

    return result;
  }

  /**
   * @param basisI    basis function index
   * @param pointJ    grid point index
   * @return          value of the basisI-th modified fundamental spline
   *                  basis function at the pointJ-th grid point
   */
  inline double evalFundamentalSplineModifiedFunctionAtGridPoint(size_t basisI, size_t pointJ) {
    const base::GridPoint& gpBasis = gridStorage[basisI];
    const base::GridPoint& gpPoint = gridStorage[pointJ];
    double result = 1.0;

    for (size_t t = 0; t < gridStorage.getDimension(); t++) {
      if (gpPoint.getLevel(t) < gpBasis.getLevel(t)) {
        return 0.0;
      } else if (gpPoint.getLevel(t) == gpBasis.getLevel(t)) {
        if (gpPoint.getIndex(t) != gpBasis.getIndex(t)) {
          return 0.0;
        }
      } else {
        const double result1d = modFundamentalSplineBasis->eval(
            gpBasis.getLevel(t), gpBasis.getIndex(t), gridStorage.getCoordinate(gpPoint, t));

        if (result1d == 0.0) {
          return 0.0;
        }

        result *= result1d;
      }
    }

    return result;
  }

  /**
   * @param basisI    basis function index
   * @param pointJ    grid point index
   * @return          value of the basisI-th Lagrange not-a-knot spline basis
   *                  function at the pointJ-th grid point
   */
  inline double evalLagrangeNotAKnotSplineFunctionAtGridPoint(size_t basisI, size_t pointJ) {
    const base::GridPoint& gpBasis = gridStorage[basisI];
    const base::GridPoint& gpPoint = gridStorage[pointJ];
    double result = 1.0;

    for (size_t t = 0; t < gridStorage.getDimension(); t++) {
      if (gpPoint.getLevel(t) < gpBasis.getLevel(t)) {
        return 0.0;
      } else {
        const double result1d = lagrangeNotAKnotSplineBasis->eval(
            gpBasis.getLevel(t), gpBasis.getIndex(t), gridStorage.getCoordinate(gpPoint, t));

        if (result1d == 0.0) {
          return 0.0;
        }

        result *= result1d;
      }
    }

    return result;
  }

  /**
   * @param basisI    basis function index
   * @param pointJ    grid point index
   * @return          value of the basisI-th modified Lagrange not-a-knot spline basis
   *                  function at the pointJ-th grid point
   */
  inline double evalLagrangeNotAKnotSplineModifiedFunctionAtGridPoint(
      size_t basisI, size_t pointJ) {
    const base::GridPoint& gpBasis = gridStorage[basisI];
    const base::GridPoint& gpPoint = gridStorage[pointJ];
    double result = 1.0;

    for (size_t t = 0; t < gridStorage.getDimension(); t++) {
      if (gpPoint.getLevel(t) < gpBasis.getLevel(t)) {
        return 0.0;
      } else {
        const double result1d = modLagrangeNotAKnotSplineBasis->eval(
            gpBasis.getLevel(t), gpBasis.getIndex(t), gridStorage.getCoordinate(gpPoint, t));

        if (result1d == 0.0) {
          return 0.0;
        }

        result *= result1d;
      }
    }

    return result;
  }

  /**
   * @param basisI    basis function index
   * @param pointJ    grid point index
   * @return          value of the basisI-th Lagrange spline basis
   *                  function at the pointJ-th grid point
   */
  inline double evalLagrangeSplineFunctionAtGridPoint(size_t basisI, size_t pointJ) {
    const base::GridPoint& gpBasis = gridStorage[basisI];
    const base::GridPoint& gpPoint = gridStorage[pointJ];
    double result = 1.0;

    for (size_t t = 0; t < gridStorage.getDimension(); t++) {
      if (gpPoint.getLevel(t) < gpBasis.getLevel(t)) {
        return 0.0;
      } else {
        const double result1d = lagrangeSplineBasis->eval(
            gpBasis.getLevel(t), gpBasis.getIndex(t), gridStorage.getCoordinate(gpPoint, t));

        if (result1d == 0.0) {
          return 0.0;
        }

        result *= result1d;
      }
    }

    return result;
  }

  /**
   * @param basisI    basis function index
   * @param pointJ    grid point index
   * @return          value of the basisI-th linear
   *                  basis function at the pointJ-th grid point
   */
  inline double evalLinearFunctionAtGridPoint(size_t basisI, size_t pointJ) {
    const base::GridPoint& gpBasis = gridStorage[basisI];
    const base::GridPoint& gpPoint = gridStorage[pointJ];
    double result = 1.0;

    for (size_t t = 0; t < gridStorage.getDimension(); t++) {
      const double result1d = linearBasis->eval(
          gpBasis.getLevel(t), gpBasis.getIndex(t), gridStorage.getCoordinate(gpPoint, t));

      if (result1d == 0.0) {
        return 0.0;
      }

      result *= result1d;
    }

    return result;
  }

  /**
   * @param basisI    basis function index
   * @param pointJ    grid point index
   * @return          value of the basisI-th linear boundary
   *                  basis function at the pointJ-th grid point
   */
  inline double evalLinearBoundaryFunctionAtGridPoint(size_t basisI, size_t pointJ) {
    const base::GridPoint& gpBasis = gridStorage[basisI];
    const base::GridPoint& gpPoint = gridStorage[pointJ];
    double result = 1.0;

    for (size_t t = 0; t < gridStorage.getDimension(); t++) {
      const double result1d = linearL0BoundaryBasis->eval(
          gpBasis.getLevel(t), gpBasis.getIndex(t), gridStorage.getCoordinate(gpPoint, t));

      if (result1d == 0.0) {
        return 0.0;
      }

      result *= result1d;
    }

    return result;
  }

  /**
   * @param basisI    basis function index
   * @param pointJ    grid point index
   * @return          value of the basisI-th linear Clenshaw-Curtis
   *                  basis function at the pointJ-th grid point
   */
  inline double evalLinearClenshawCurtisFunctionAtGridPoint(size_t basisI, size_t pointJ) {
    const base::GridPoint& gpBasis = gridStorage[basisI];
    const base::GridPoint& gpPoint = gridStorage[pointJ];
    double result = 1.0;

    for (size_t t = 0; t < gridStorage.getDimension(); t++) {
      const double result1d = linearClenshawCurtisBasis->eval(
          gpBasis.getLevel(t), gpBasis.getIndex(t), gridStorage.getCoordinate(gpPoint, t));

      if (result1d == 0.0) {
        return 0.0;
      }

      result *= result1d;
    }

    return result;
  }

  /**
   * @param basisI    basis function index
   * @param pointJ    grid point index
   * @return          value of the basisI-th modified linear
   *                  basis function at the pointJ-th grid point
   */
  inline double evalLinearModifiedFunctionAtGridPoint(size_t basisI, size_t pointJ) {
    const base::GridPoint& gpBasis = gridStorage[basisI];
    const base::GridPoint& gpPoint = gridStorage[pointJ];
    double result = 1.0;

    for (size_t t = 0; t < gridStorage.getDimension(); t++) {
      const double result1d = modLinearBasis->eval(
          gpBasis.getLevel(t), gpBasis.getIndex(t), gridStorage.getCoordinate(gpPoint, t));

      if (result1d == 0.0) {
        return 0.0;
      }

      result *= result1d;
    }

    return result;
  }

  /**
   * @param basisI    basis function index
   * @param pointJ    grid point index
   * @return          value of the basisI-th natural B-spline basis function
   *                  at the pointJ-th grid point
   */
  inline double evalNaturalBsplineFunctionAtGridPoint(size_t basisI, size_t pointJ) {
    const base::GridPoint& gpBasis = gridStorage[basisI];
    const base::GridPoint& gpPoint = gridStorage[pointJ];
    double result = 1.0;

    for (size_t t = 0; t < gridStorage.getDimension(); t++) {
      const double result1d = naturalBsplineBasis->eval(
          gpBasis.getLevel(t), gpBasis.getIndex(t), gridStorage.getCoordinate(gpPoint, t));

      if (result1d == 0.0) {
        return 0.0;
      }

      result *= result1d;
    }

    return result;
  }

  /**
   * @param basisI    basis function index
   * @param pointJ    grid point index
   * @return          value of the basisI-th not-a-knot B-spline basis function
   *                  at the pointJ-th grid point
   */
  inline double evalNotAKnotBsplineFunctionAtGridPoint(size_t basisI, size_t pointJ) {
    const base::GridPoint& gpBasis = gridStorage[basisI];
    const base::GridPoint& gpPoint = gridStorage[pointJ];
    double result = 1.0;

    for (size_t t = 0; t < gridStorage.getDimension(); t++) {
      const double result1d = notAKnotBsplineBasis->eval(
          gpBasis.getLevel(t), gpBasis.getIndex(t), gridStorage.getCoordinate(gpPoint, t));

      if (result1d == 0.0) {
        return 0.0;
      }

      result *= result1d;
    }

    return result;
  }

  /**
   * @param basisI    basis function index
   * @param pointJ    grid point index
   * @return          value of the basisI-th modified not-a-knot B-spline basis function
   *                  at the pointJ-th grid point
   */
  inline double evalNotAKnotBsplineModifiedFunctionAtGridPoint(size_t basisI, size_t pointJ) {
    const base::GridPoint& gpBasis = gridStorage[basisI];
    const base::GridPoint& gpPoint = gridStorage[pointJ];
    double result = 1.0;

    for (size_t t = 0; t < gridStorage.getDimension(); t++) {
      const double result1d = modNotAKnotBsplineBasis->eval(
          gpBasis.getLevel(t), gpBasis.getIndex(t), gridStorage.getCoordinate(gpPoint, t));

      if (result1d == 0.0) {
        return 0.0;
      }

      result *= result1d;
    }

    return result;
  }

  /**
   * @param basisI    basis function index
   * @param pointJ    grid point index
   * @return          value of the basisI-th wavelet
   *                  basis function at the pointJ-th grid point
   */
  inline double evalWaveletFunctionAtGridPoint(size_t basisI, size_t pointJ) {
    const base::GridPoint& gpBasis = gridStorage[basisI];
    const base::GridPoint& gpPoint = gridStorage[pointJ];
    double result = 1.0;

    for (size_t t = 0; t < gridStorage.getDimension(); t++) {
      const double result1d = waveletBasis->eval(
          gpBasis.getLevel(t), gpBasis.getIndex(t), gridStorage.getCoordinate(gpPoint, t));

      if (result1d == 0.0) {
        return 0.0;
      }

      result *= result1d;
    }

    return result;
  }

  /**
   * @param basisI    basis function index
   * @param pointJ    grid point index
   * @return          value of the basisI-th wavelet boundary
   *                  basis function at the pointJ-th grid point
   */
  inline double evalWaveletBoundaryFunctionAtGridPoint(size_t basisI, size_t pointJ) {
    const base::GridPoint& gpBasis = gridStorage[basisI];
    const base::GridPoint& gpPoint = gridStorage[pointJ];
    double result = 1.0;

    for (size_t t = 0; t < gridStorage.getDimension(); t++) {
      const double result1d = waveletBoundaryBasis->eval(
          gpBasis.getLevel(t), gpBasis.getIndex(t), gridStorage.getCoordinate(gpPoint, t));

      if (result1d == 0.0) {
        return 0.0;
      }

      result *= result1d;
    }

    return result;
  }

  /**
   * @param basisI    basis function index
   * @param pointJ    grid point index
   * @return          value of the basisI-th modified wavelet
   *                  basis function at the pointJ-th grid point
   */
  inline double evalWaveletModifiedFunctionAtGridPoint(size_t basisI, size_t pointJ) {
    const base::GridPoint& gpBasis = gridStorage[basisI];
    const base::GridPoint& gpPoint = gridStorage[pointJ];
    double result = 1.0;

    for (size_t t = 0; t < gridStorage.getDimension(); t++) {
      const double result1d = modWaveletBasis->eval(
          gpBasis.getLevel(t), gpBasis.getIndex(t), gridStorage.getCoordinate(gpPoint, t));

      if (result1d == 0.0) {
        return 0.0;
      }

      result *= result1d;
    }

    return result;
  }
};
}  // namespace optimization
}  // namespace sgpp

#endif /* SGPP_OPTIMIZATION_SLE_SYSTEM_HIERARCHISATIONSLE_HPP */
