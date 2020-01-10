//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _       _
//                        | |     | |
//                    __ _| |_ ___| | __ _ _ __   __ _
//                   / _` | __/ __| |/ _` | '_ \ / _` |
//                  | (_| | || (__| | (_| | | | | (_| |
//                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
//                    __/ |                       __/ |
//                   |___/                       |___/
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#pragma once

#include <tuple>
#include <vector>

#include "atlas/mesh.h"
#include "atlas/mesh/HybridElements.h"

using Point = std::tuple<double, double>;
enum class Orientation { Horizontal = 0, Vertical };

// quick class that maps a structured Atlas mesh to cartesian coordinates in [0,1]
//  can probably be optimized by using the information implied by the Atlas numbering schemes
class AtlasToCartesian {
private:
  std::vector<Point> nodeToCart;

public:
  Point cellMidpoint(const atlas::Mesh& mesh, int cellIdx) const;

  Orientation edgeOrientation(const atlas::Mesh& mesh, int edgeIdx) const;

  Point nodeLocation(int nodeIdx) const { return nodeToCart[nodeIdx]; }

  std::tuple<Point, Point> cartesianEdge(const atlas::Mesh& mesh, int edgeIdx) const;

  Point edgeMidpoint(const atlas::Mesh& mesh, int edgeIdx) const;

  AtlasToCartesian(const atlas::Mesh& mesh);
};