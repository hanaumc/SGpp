// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#ifndef GRIDSTORAGE_HPP
#define GRIDSTORAGE_HPP

#include <sgpp/base/grid/LevelIndexTypes.hpp>
#include <sgpp/base/grid/storage/hashmap/HashGridPoint.hpp>
#include <sgpp/base/grid/storage/hashmap/HashGridStorage.hpp>
#include <sgpp/base/grid/storage/hashmap/HashGridIterator.hpp>

#include <sgpp/globaldef.hpp>

namespace sgpp {
namespace base {

/**
 * Main typedef for GridPoint
 */
// typedef HashGridPoint<unsigned int, unsigned int> GridPoint;
typedef HashGridPoint GridPoint;

/**
 * Main typedef for GridStorage
 */
typedef HashGridStorage GridStorage;

}  // namespace base
}  // namespace sgpp

#endif /* GRIDSTORAGE_HPP */
