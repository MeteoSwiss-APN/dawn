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

#include "AtlasCartesianWrapper.h"

Point AtlasToCartesian::cellMidpoint(const atlas::Mesh& mesh, int cellIdx) const {
  const atlas::mesh::HybridElements::Connectivity& cellNodeConnectivity =
      mesh.cells().node_connectivity();

  const int missingVal = cellNodeConnectivity.missing_value();
  int numNbh = cellNodeConnectivity.cols(cellIdx);
  double midX = 0;
  double midY = 0;
  for(int nbh = 0; nbh < numNbh; nbh++) {
    int nbhIdx = cellNodeConnectivity(cellIdx, nbh);
    assert(nbhIdx != missingVal);
    auto [nbhX, nbhY] = nodeLocation(nbhIdx);
    midX += nbhX;
    midY += nbhY;
  }
  midX /= numNbh;
  midY /= numNbh;
  return {midX, midY};
}

Orientation AtlasToCartesian::edgeOrientation(const atlas::Mesh& mesh, int edgeIdx) const {
  auto [lo, hi] = cartesianEdge(mesh, edgeIdx);
  auto [loX, loY] = lo;
  auto [hiX, hiY] = hi;
  double dx = fabs(loX - hiX);
  double dy = fabs(loY - hiY);
  double tol = 1e3 * std::numeric_limits<double>::epsilon();
  assert(!(dx < tol && dy < tol));
  if(dx < tol) {
    return Orientation::Vertical;
  } else {
    return Orientation::Horizontal;
  }
}

std::tuple<Point, Point> AtlasToCartesian::cartesianEdge(const atlas::Mesh& mesh,
                                                         int edgeIdx) const {
  const atlas::mesh::HybridElements::Connectivity& edgeNodeConnectivity =
      mesh.edges().node_connectivity();
  const int missingVal = edgeNodeConnectivity.missing_value();

  int numNbh = edgeNodeConnectivity.cols(edgeIdx);
  assert(numNbh == 2);

  int nbhLo = edgeNodeConnectivity(edgeIdx, 0);
  int nbhHi = edgeNodeConnectivity(edgeIdx, 1);

  assert((nbhLo != missingVal && nbhHi != missingVal));

  return {nodeToCart[nbhLo], nodeToCart[nbhHi]};
}

Point AtlasToCartesian::edgeMidpoint(const atlas::Mesh& mesh, int edgeIdx) const {
  auto [from, to] = cartesianEdge(mesh, edgeIdx);
  auto [fromX, fromY] = from;
  auto [toX, toY] = to;
  return {0.5 * (fromX + toX), 0.5 * (fromY + toY)};
}

AtlasToCartesian::AtlasToCartesian(const atlas::Mesh& mesh) : nodeToCart(mesh.nodes().size()) {
  double lonLo = std::numeric_limits<double>::max();
  double lonHi = -std::numeric_limits<double>::max();
  double latLo = std::numeric_limits<double>::max();
  double latHi = -std::numeric_limits<double>::max();

  auto lonlat = atlas::array::make_view<double, 2>(mesh.nodes().lonlat());

  for(int cellIdx = 0; cellIdx < mesh.nodes().size(); cellIdx++) {
    lonLo = fmin(lonlat(cellIdx, 0), lonLo);
    lonHi = fmax(lonlat(cellIdx, 0), lonHi);
    latLo = fmin(lonlat(cellIdx, 1), latLo);
    latHi = fmax(lonlat(cellIdx, 1), latHi);
  }

  for(int cellIdx = 0; cellIdx < mesh.nodes().size(); cellIdx++) {
    double lon = lonlat(cellIdx, 0);
    double lat = lonlat(cellIdx, 1);
    double cartx = (lon - lonLo) / (lonHi - lonLo);
    double carty = (lat - latLo) / (latHi - latLo);
    nodeToCart[cellIdx] = {cartx, carty};
  }
}