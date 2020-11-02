//===--------------------------------------------------------------------------------*- C++ -*-===//
//                          _
//                         | |
//                       __| | __ ___      ___ ___
//                      / _` |/ _` \ \ /\ / / '_  |
//                     | (_| | (_| |\ V  V /| | | |
//                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "IcoChainSizes.h"

#include "dawn/AST/LocationType.h"
#include "dawn/Support/HashCombine.h"

#include <assert.h>
#include <unordered_set>

namespace dawn {

using connection_t = std::tuple<ast::LocationType, ast::LocationType>;

using grid_location_t = std::tuple<int, int, int>;

/******************************************************************************
 *  We encode the grid as follows:
 *
 * \|{-1, 1}                 \|{0, 1}                  \|
 * -*-------------------------*-------------------------*-
 *  |\     {-1, 1, 0}         |\     {0, 1, 0}          |\
 *  | \                       | \                       |
 *  |  \                      |  \                      |
 *  |   \       {-1, 0, 1}    |   \       {0, 0, 1}     |
 *  |    \                    |    \                    |
 *  |     \                   |     \                   |
 *  |      \                  |      \                  |
 *  |       \                 |       \                 |
 *  |        \                |        \                |
 *  |         \               |         \               |
 *  |          \{-1, 0, 1}    |          \{0, 0, 1}     |
 *  |           \             |           \             |
 *  |            \            |            \            |
 *  |             \           |             \           |
 *  |{-1, 0, 2}    \          |{0, 0, 2}     \          |
 *  |               \         |               \         |
 *  |                \        |                \        |
 *  |                 \       |                 \       |
 *  |                  \      |                  \      |
 *  |                   \     |                   \     |
 *  |                    \    |                    \    |
 *  |   {-1, 0, 0}        \   |   {0, 0, 0}         \   |
 *  |                      \  |                      \  |
 *  |                       \ |                       \ |
 * \|{-1, 0}                 \|{0, 0}                  \|
 * -*-------------------------*-------------------------*-
 *  |\     {-1, 0, 0}         |\     {0, 0, 0}          |\
 *  | \                       | \                       |
 *  |  \                      |  \                      |
 *  |   \       {-1, -1, 1}   |   \       {0, -1, 1}    |
 *  |    \                    |    \                    |
 *  |     \                   |     \                   |
 *  |      \                  |      \                  |
 *  |       \                 |       \                 |
 *  |        \                |        \                |
 *  |         \               |         \               |
 *  |          \{-1, -1, 1}   |          \{0, -1, 1}    |
 *  |           \             |           \             |
 *  |            \            |            \            |
 *  |             \           |             \           |
 *  |{-1, -1, 2}   \          |{0, -1, 2}    \          |
 *  |               \         |               \         |
 *  |                \        |                \        |
 *  |                 \       |                 \       |
 *  |                  \      |                  \      |
 *  |                   \     |                   \     |
 *  |                    \    |                    \    |
 *  |   {-1, -1, 0}       \   |   {0, -1, 0}        \   |
 *  |                      \  |                      \  |
 *  |                       \ |                       \ |
 * \|{-1, -1}                \|{0, -1}                 \|
 * -*-------------------------*-------------------------*-
 *  |\     {-1, -1, 0}        |\     {0, -1, 0}         |\
 *
 *
 * Which is described by this general pattern:
 *
 *  |\
 *  | \
 *  |  \
 *  |   \       {x, y, 1}
 *  |    \
 *  |     \
 *  |      \
 *  |       \
 *  |        \
 *  |         \
 *  |          \{x, y, 1}
 *  |           \
 *  |            \
 *  |             \
 *  |{x, y, 2}     \
 *  |               \
 *  |                \
 *  |                 \
 *  |                  \
 *  |                   \
 *  |                    \
 *  |   {x, y, 0}         \
 *  |                      \
 *  |                       \
 *  |{x, y}                  \
 *  *-------------------------
 *         {x, y, 0}
 *
 * Note: Each location type uses a separate _id-space_.
 * {x, y, 0} can both mean an edge or cell. It's up to the user to ensure
 * they know what location type is meant.
 * To make the implementation easier, vertices are encoded also
 * as `std::tuple<int, int, int>` ({x, y, 0}).
 */

static std::array<grid_location_t, 6> get_vertex_to_edge(const grid_location_t& vertex) {
  const auto [x, y, _] = vertex;
  return {{
      {x, y, 0},
      {x, y, 2},
      {x - 1, y, 0},
      {x - 1, y, 1},
      {x, y - 1, 1},
      {x, y - 1, 2},
  }};
}

static std::array<grid_location_t, 6> get_vertex_to_cell(const grid_location_t& vertex) {
  const auto [x, y, _] = vertex;
  return {{
      {x, y, 0},
      {x - 1, y, 0},
      {x - 1, y, 1},
      {x, y - 1, 0},
      {x, y - 1, 1},
      {x - 1, y - 1, 1},
  }};
}

static std::array<grid_location_t, 2> get_edge_to_vertex(const grid_location_t& edge) {
  const auto [x, y, e] = edge;
  if(e == 0) {
    return {{{x, y, 0}, {x + 1, y, 0}}};
  } else if(e == 1) {
    return {{{x + 1, y, 0}, {x, y + 1, 0}}};
  } else if(e == 2) {
    return {{{x, y, 0}, {x, y + 1, 0}}};
  } else {
    throw std::runtime_error("Invalid edge type");
  }
}

static std::array<grid_location_t, 2> get_edge_to_cell(const grid_location_t& edge) {
  const auto [x, y, e] = edge;
  if(e == 0) {
    return {{{x, y, 0}, {x, y - 1, 1}}};
  } else if(e == 1) {
    return {{{x, y, 0}, {x, y, 1}}};
  } else if(e == 2) {
    return {{{x, y, 0}, {x - 1, y, 1}}};
  } else {
    throw std::runtime_error("Invalid edge type");
  }
}

static std::array<grid_location_t, 3> get_cell_to_vertex(const grid_location_t& cell) {
  const auto [x, y, c] = cell;
  if(c == 0) {
    return {{{x, y, 0}, {x + 1, y, 0}, {x, y + 1, 0}}};
  } else if(c == 1) {
    return {{{x + 1, y + 1, 0}, {x + 1, y, 0}, {x, y + 1, 0}}};
  } else {
    throw std::runtime_error("Invalid cell type");
  }
}

static std::array<grid_location_t, 3> get_cell_to_edge(const grid_location_t& cell) {
  const auto [x, y, c] = cell;
  if(c == 0) {
    return {{{x, y, 0}, {x, y, 1}, {x, y, 2}}};
  } else if(c == 1) {
    return {{{x, y, 1}, {x + 1, y, 2}, {x, y + 1, 0}}};
  } else {
    throw std::runtime_error("Invalid cell type");
  }
}

int ICOChainSize(const ast::NeighborChain& chain) {

  assert(1 < chain.size());

  auto previous_location_type = chain[0];
  std::unordered_set<grid_location_t> previous_locations{{0, 0, 0}};

  for(ast::NeighborChain::size_type i = 1; i < chain.size(); ++i) {

    const auto current_location_type = chain[i];
    std::unordered_set<grid_location_t> current_locations;

    assert(previous_location_type != current_location_type);
    const connection_t connection{previous_location_type, current_location_type};

    // Vertices -> Edges
    if(connection == connection_t{ast::LocationType::Vertices, ast::LocationType::Edges}) {
      for(const auto& previous_location : previous_locations) {
        const auto neighbors = get_vertex_to_edge(previous_location);
        current_locations.insert(neighbors.cbegin(), neighbors.cend());
      }
      // Vertices -> Cells
    } else if(connection == connection_t{ast::LocationType::Vertices, ast::LocationType::Cells}) {
      for(const auto& previous_location : previous_locations) {
        const auto neighbors = get_vertex_to_cell(previous_location);
        current_locations.insert(neighbors.cbegin(), neighbors.cend());
      }
      // Edges -> Vertices
    } else if(connection == connection_t{ast::LocationType::Edges, ast::LocationType::Vertices}) {
      for(const auto& previous_location : previous_locations) {
        const auto neighbors = get_edge_to_vertex(previous_location);
        current_locations.insert(neighbors.cbegin(), neighbors.cend());
      }
      // Edges -> Cells
    } else if(connection == connection_t{ast::LocationType::Edges, ast::LocationType::Cells}) {
      for(const auto& previous_location : previous_locations) {
        const auto neighbors = get_edge_to_cell(previous_location);
        current_locations.insert(neighbors.cbegin(), neighbors.cend());
      }
      // Cells -> Vertices
    } else if(connection == connection_t{ast::LocationType::Cells, ast::LocationType::Vertices}) {
      for(const auto& previous_location : previous_locations) {
        const auto neighbors = get_cell_to_vertex(previous_location);
        current_locations.insert(neighbors.cbegin(), neighbors.cend());
      }
      // Cells -> Edges
    } else if(connection == connection_t{ast::LocationType::Cells, ast::LocationType::Edges}) {
      for(const auto& previous_location : previous_locations) {
        const auto neighbors = get_cell_to_edge(previous_location);
        current_locations.insert(neighbors.cbegin(), neighbors.cend());
      }
    } else {
      assert(false);
    }

    previous_locations = std::move(current_locations);
    previous_location_type = current_location_type;
  }

  if(chain.front() == chain.back()) {
    // we don't include the starting location
    return previous_locations.size() - 1;
  } else {
    return previous_locations.size();
  }
}
} // namespace dawn