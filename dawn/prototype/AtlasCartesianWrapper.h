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
using Vector = std::tuple<double, double>;
enum class Orientation { Horizontal = 0, Diagonal = 1, Vertical = 2 };
// quick class that maps a structured Atlas mesh to cartesian coordinates in [0,1]
//  can probably be optimized by using the information implied by the Atlas numbering schemes
class AtlasToCartesian {
private:
  std::vector<Point> nodeToCart;
  std::vector<Point> nodeToCartUnskewed;

public:
  Orientation edgeOrientation(const atlas::Mesh& mesh, int edgeIdx) const;
  Point cellMidpoint(const atlas::Mesh& mesh, int cellIdx) const;
  Point cellCircumcenter(const atlas::Mesh& mesh, int cellIdx) const;
  double cellArea(const atlas::Mesh& mesh, int cellIdx) const;

  double edgeLength(const atlas::Mesh& mesh, int edgeIdx) const;
  double dualEdgeLength(const atlas::Mesh& mesh, int edgeIdx) const;
  double tangentOrientation(const atlas::Mesh& mesh, int edgeIdx) const;
  Vector primalNormal(const atlas::Mesh& mesh, int edgeIdx) const;
  std::tuple<Point, Point> cartesianEdge(const atlas::Mesh& mesh, int edgeIdx) const;
  Point edgeMidpoint(const atlas::Mesh& mesh, int edgeIdx) const;

  Point nodeLocation(int nodeIdx) const { return nodeToCart[nodeIdx]; }
  double dualCellArea(const atlas::Mesh& mesh, int nodeIdx) const;

  explicit AtlasToCartesian(const atlas::Mesh& mesh, double scale = 1, bool skewTrafo = false);
};