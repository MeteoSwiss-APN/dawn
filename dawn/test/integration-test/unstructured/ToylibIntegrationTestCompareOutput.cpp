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

//===------------------------------------------------------------------------------------------===//
//
//  NOTE: in code comments explaining the individual tests can be found in
//  AtlasIntegrationTestCompareOutput.cpp
//
//===------------------------------------------------------------------------------------------===//

#include "UnstructuredVerifier.h"
#include "dawn/Support/Logger.h"
#include "interface/toylib_interface.hpp"
#include "toylib/toylib.hpp"

#include <gtest/gtest.h>

namespace {
template <typename ValT, typename iteratorT, template <typename> class FieldT>
void InitData(const iteratorT& iter, FieldT<ValT>& field, size_t kSize, ValT val) {
  for(size_t level = 0; level < kSize; level++) {
    for(const auto& e : iter) {
      field(e, level) = val;
    }
  }
}
template <typename ValT, typename iteratorT, template <typename> class FieldT>
void InitSparseData(const iteratorT& iter, FieldT<ValT>& field, size_t sparseSize, size_t kSize,
                    ValT val) {
  for(size_t level = 0; level < kSize; level++) {
    for(size_t sparse = 0; sparse < sparseSize; sparse++) {
      for(const auto& e : iter) {
        field(e, sparse, level) = val;
      }
    }
  }
}
std::tuple<double, double> cellMidpoint(const toylib::Face& f) {
  double x = f.vertices()[0]->x() + f.vertices()[1]->x() + f.vertices()[2]->x();
  double y = f.vertices()[0]->y() + f.vertices()[1]->y() + f.vertices()[2]->y();
  return {x / 3., y / 3.};
}
} // namespace

namespace {
#include <generated_copyCell.hpp>
TEST(ToylibIntegrationTestCompareOutput, CopyCell) {
  toylib::Grid mesh(32, 32);
  size_t nb_levels = 1;
  toylib::FaceData<double> out(mesh, nb_levels);
  toylib::FaceData<double> in(mesh, nb_levels);

  InitData(mesh.faces(), in, nb_levels, 1.0);
  InitData(mesh.faces(), out, nb_levels, -1.0);

  dawn_generated::cxxnaiveico::copyCell<toylibInterface::toylibTag>(
      mesh, static_cast<int>(nb_levels), in, out)
      .run();

  for(const auto& f : mesh.faces())
    ASSERT_EQ(out(f, 0), 1.0);
}

#include <generated_copyEdge.hpp>
TEST(ToylibIntegrationTestCompareOutput, CopyEdge) {
  toylib::Grid mesh(32, 32);
  size_t nb_levels = 1;
  toylib::EdgeData<double> out(mesh, nb_levels);
  toylib::EdgeData<double> in(mesh, nb_levels);

  InitData(mesh.edges(), in, nb_levels, 1.0);
  InitData(mesh.edges(), out, nb_levels, -1.0);

  dawn_generated::cxxnaiveico::copyEdge<toylibInterface::toylibTag>(
      mesh, static_cast<int>(nb_levels), in, out)
      .run();

  for(const auto& e : mesh.edges())
    ASSERT_EQ(out(e, 0), 1.0);
}

#include <generated_verticalSum.hpp>
TEST(ToylibIntegrationTestCompareOutput, VerticalCopy) {
  toylib::Grid mesh(32, 32);
  size_t nb_levels = 5;
  toylib::FaceData<double> out(mesh, nb_levels);
  toylib::FaceData<double> in(mesh, nb_levels);

  double initValue = 10.;
  InitData(mesh.faces(), in, nb_levels, initValue);
  InitData(mesh.faces(), out, nb_levels, -1.0);

  dawn_generated::cxxnaiveico::verticalSum<toylibInterface::toylibTag>(
      mesh, static_cast<int>(nb_levels), in, out)
      .run();

  for(size_t level = 1; level < nb_levels - 1; level++) {
    for(const auto& f : mesh.faces()) {
      ASSERT_EQ(out(f, level), 2 * initValue);
    }
  }
}

#include <generated_accumulateEdgeToCell.hpp>
TEST(ToylibIntegrationTestCompareOutput, Accumulate) {
  toylib::Grid mesh(32, 32);
  size_t nb_levels = 1;
  toylib::EdgeData<double> in(mesh, nb_levels);
  toylib::FaceData<double> out(mesh, nb_levels);

  InitData(mesh.edges(), in, nb_levels, 1.0);
  InitData(mesh.faces(), out, nb_levels, -1.0);

  dawn_generated::cxxnaiveico::accumulateEdgeToCell<toylibInterface::toylibTag>(
      mesh, static_cast<int>(nb_levels), in, out)
      .run();

  for(const auto& f : mesh.faces())
    ASSERT_EQ(out(f, 0), 3.0);
}

#include <generated_diffusion.hpp>
#include <reference_diffusion.hpp>
TEST(ToylibIntegrationTestCompareOutput, Diffusion) {
  toylib::Grid mesh(32, 32, false, 1., 1.);
  size_t nb_levels = 1;

  toylib::FaceData<double> in_ref(mesh, nb_levels);
  toylib::FaceData<double> out_ref(mesh, nb_levels);
  toylib::FaceData<double> in_gen(mesh, nb_levels);
  toylib::FaceData<double> out_gen(mesh, nb_levels);

  for(const auto& cell : mesh.faces()) {
    auto [x, y] = cellMidpoint(cell);
    bool inX = x > 0.375 && x < 0.625;
    bool inY = y > 0.375 && y < 0.625;
    in_ref(cell, 0) = (inX && inY) ? 1 : 0;
    in_gen(cell, 0) = (inX && inY) ? 1 : 0;
  }

  for(int i = 0; i < 5; ++i) {
    // Run the stencils
    dawn_generated::cxxnaiveico::reference_diffusion<toylibInterface::toylibTag>(
        mesh, static_cast<int>(nb_levels), in_ref, out_ref)
        .run();
    dawn_generated::cxxnaiveico::diffusion<toylibInterface::toylibTag>(
        mesh, static_cast<int>(nb_levels), in_gen, out_gen)
        .run();

    // Swap in and out
    using std::swap;
    swap(in_ref, out_ref);
    swap(in_gen, out_gen);
  }

  {
    UnstructuredVerifier v;
    EXPECT_TRUE(v.compareToylibField(mesh.faces(), in_ref, in_gen, nb_levels))
        << "while comparing output (on cells)";
  }
}

#include <generated_gradient.hpp>
#include <reference_gradient.hpp>
TEST(ToylibIntegrationTestCompareOutput, Gradient) {
  const int numCell = 10;
  auto mesh =
      toylib::Grid(numCell, numCell, false, M_PI,
                   M_PI); // periodic seems bugged in toylib, but boundaries are simply cut off.
                          // since our boundaries are 0 this seems not too bad of a compromise

  // boundary edges in toylib have only one entry in their cell neighbors list; the cell inside the
  // domain. Hence the domain can be said to be "cut off" there. For the gradient test case we have
  // zero boundaries. So this kinda works out.

  // In atlas we have two entries. Instead of deleting one entry the domain is wrapped
  // around. But again, we have zero boundaries, so a boundary edge "sees" another zero on the other
  // end of the domain.

  const int nb_levels = 1;

  toylib::FaceData<double> ref_cells(mesh, nb_levels);
  toylib::EdgeData<double> ref_edges(mesh, nb_levels);
  toylib::FaceData<double> gen_cells(mesh, nb_levels);
  toylib::EdgeData<double> gen_edges(mesh, nb_levels);

  for(const auto& f : mesh.faces()) {
    auto [x, y] = cellMidpoint(f);
    double val = sin(x) * sin(y);
    ref_cells(f, 0) = val;
    gen_cells(f, 0) = val;
  }

  dawn_generated::cxxnaiveico::gradient<toylibInterface::toylibTag>(mesh, nb_levels, gen_cells,
                                                                    gen_edges)
      .run();
  dawn_generated::cxxnaiveico::reference_gradient<toylibInterface::toylibTag>(mesh, nb_levels,
                                                                              ref_cells, ref_edges)
      .run();

  {
    UnstructuredVerifier v;
    EXPECT_TRUE(v.compareToylibField(mesh.faces(), ref_cells, gen_cells, nb_levels))
        << "while comparing output (on cells)";
  }
}

#include <generated_diamond.hpp>
#include <reference_diamond.hpp>
TEST(ToylibIntegrationTestCompareOutput, Diamond) {
  const int numCell = 10;
  auto mesh = toylib::Grid(numCell, numCell, false, M_PI, M_PI, true);
  const int nb_levels = 1;

  toylib::VertexData<double> in(mesh, nb_levels);
  toylib::EdgeData<double> ref_out(mesh, nb_levels);
  toylib::EdgeData<double> gen_out(mesh, nb_levels);

  for(const auto& v : mesh.vertices()) {
    double val = sin(v.x()) * sin(v.y());
    in(v, 0) = val;
  }

  dawn_generated::cxxnaiveico::diamond<toylibInterface::toylibTag>(mesh, nb_levels, gen_out, in)
      .run();
  dawn_generated::cxxnaiveico::reference_diamond<toylibInterface::toylibTag>(mesh, nb_levels,
                                                                             ref_out, in)
      .run();

  {
    UnstructuredVerifier v;
    EXPECT_TRUE(v.compareToylibField(mesh.all_edges(), ref_out, gen_out, nb_levels))
        << "while comparing output (on cells)";
  }
}

#include <generated_diamondWeights.hpp>
#include <reference_diamondWeights.hpp>
TEST(ToylibIntegrationTestCompareOutput, DiamondWeights) {
  const int numCell = 10;
  auto mesh = toylib::Grid(numCell, numCell, false, M_PI, M_PI, true);
  const int nb_levels = 1;

  toylib::EdgeData<double> ref_out(mesh, nb_levels);
  toylib::EdgeData<double> gen_out(mesh, nb_levels);
  toylib::EdgeData<double> inv_edge_length(mesh, nb_levels);
  toylib::EdgeData<double> inv_vert_length(mesh, nb_levels);
  toylib::VertexData<double> in(mesh, nb_levels);

  for(const auto& v : mesh.vertices()) {
    double val = sin(v.x()) * sin(v.y());
    in(v, 0) = val;
  }
  for(const auto& e : mesh.edges()) {
    double dx = e.get().vertex(0).x() - e.get().vertex(1).x();
    double dy = e.get().vertex(0).y() - e.get().vertex(1).y();
    double edgeLength = sqrt(dx * dx + dy * dy);
    inv_edge_length(e, 0) = 1. / edgeLength;
    inv_vert_length(e, 0) =
        1. / (0.5 * sqrt(3.) * edgeLength * 2); // twice the height of equialt triangle
  }

  dawn_generated::cxxnaiveico::diamondWeights<toylibInterface::toylibTag>(
      mesh, nb_levels, gen_out, inv_edge_length, inv_vert_length, in)
      .run();
  dawn_generated::cxxnaiveico::reference_diamondWeights<toylibInterface::toylibTag>(
      mesh, nb_levels, ref_out, inv_edge_length, inv_vert_length, in)
      .run();

  {
    UnstructuredVerifier v;
    EXPECT_TRUE(v.compareToylibField(mesh.all_edges(), ref_out, gen_out, nb_levels))
        << "while comparing output (on cells)";
  }
}

#include <generated_intp.hpp>
#include <reference_intp.hpp>
TEST(ToylibIntegrationTestCompareOutput, Intp) {
  const int numCell = 10;
  auto mesh = toylib::Grid(numCell, numCell, false, M_PI, M_PI, true);
  const int nb_levels = 1;

  toylib::FaceData<double> ref_in(mesh, nb_levels);
  toylib::FaceData<double> gen_in(mesh, nb_levels);
  toylib::FaceData<double> ref_out(mesh, nb_levels);
  toylib::FaceData<double> gen_out(mesh, nb_levels);

  for(const auto& f : mesh.faces()) {
    auto [x, y] = cellMidpoint(f);
    double val = sin(x) * sin(y);
    ref_in(f, 0) = val;
    gen_in(f, 0) = val;
  }

  dawn_generated::cxxnaiveico::intp<toylibInterface::toylibTag>(mesh, nb_levels, gen_in, gen_out)
      .run();
  dawn_generated::cxxnaiveico::reference_intp<toylibInterface::toylibTag>(mesh, nb_levels, ref_in,
                                                                          ref_out)
      .run();

  {
    UnstructuredVerifier v;
    EXPECT_TRUE(v.compareToylibField(mesh.faces(), ref_out, gen_out, nb_levels))
        << "while comparing output (on cells)";
  }
}

#include <generated_verticalSolver.hpp>
TEST(ToylibIntegrationTestCompareOutput, verticalSolver) {
  const int numCell = 5;
  auto mesh = toylib::Grid(numCell, numCell);
  const int nb_levels = 5;

  toylib::FaceData<double> a(mesh, nb_levels);
  toylib::FaceData<double> b(mesh, nb_levels);
  toylib::FaceData<double> c(mesh, nb_levels);
  toylib::FaceData<double> d(mesh, nb_levels);

  for(const auto& f : mesh.faces()) {
    for(int k = 0; k < nb_levels; k++) {
      a(f, k) = k + 1;
      b(f, k) = k + 1;
      c(f, k) = k + 2;
    }

    d(f, 0) = 5;
    d(f, 1) = 15;
    d(f, 2) = 31;
    d(f, 3) = 53;
    d(f, 4) = 45;
  }

  dawn_generated::cxxnaiveico::tridiagonalSolve<toylibInterface::toylibTag>(mesh, nb_levels, a, b,
                                                                            c, d)
      .run();

  for(const auto& f : mesh.faces()) {
    for(int k = 0; k < nb_levels; k++) {
      EXPECT_TRUE(abs(d(f, k) - (k + 1)) < 1e3 * std::numeric_limits<double>::epsilon());
    }
  }
}

#include <generated_NestedSimple.hpp>
TEST(ToylibIntegrationTestCompareOutput, nestedSimple) {
  auto mesh = toylib::Grid(10, 10);
  const int nb_levels = 1;

  toylib::FaceData<double> cells(mesh, nb_levels);
  toylib::EdgeData<double> edges(mesh, nb_levels);
  toylib::VertexData<double> nodes(mesh, nb_levels);

  InitData(mesh.vertices(), nodes, nb_levels, 1.);

  dawn_generated::cxxnaiveico::nestedSimple<toylibInterface::toylibTag>(mesh, nb_levels, cells,
                                                                        edges, nodes)
      .run();
  // each vertex stores value 1                 1
  // vertices are reduced onto edges            2
  // each face reduces its edges (3 per face)   6
  for(const auto& f : mesh.faces()) {
    EXPECT_TRUE(fabs(cells(f, 0) - 6.) < 1e3 * std::numeric_limits<double>::epsilon());
  }
}

#include <generated_NestedWithField.hpp>
TEST(ToylibIntegrationTestCompareOutput, nestedWithField) {
  auto mesh = toylib::Grid(10, 10);
  const int nb_levels = 1;

  toylib::FaceData<double> cells(mesh, nb_levels);
  toylib::EdgeData<double> edges(mesh, nb_levels);
  toylib::VertexData<double> nodes(mesh, nb_levels);

  InitData(mesh.vertices(), nodes, nb_levels, 1.);
  InitData(mesh.edges(), edges, nb_levels, 200.);

  dawn_generated::cxxnaiveico::nestedWithField<toylibInterface::toylibTag>(mesh, nb_levels, cells,
                                                                           edges, nodes)
      .run();
  // each vertex stores value 1                 1
  // vertices are reduced onto edges            2
  // each edge stores 200                     202
  // each face reduces its edges (3 per face) 606
  for(const auto& f : mesh.faces()) {
    EXPECT_TRUE(fabs(cells(f, 0) - 606) < 1e3 * std::numeric_limits<double>::epsilon());
  }
}

#include <generated_sparseDimension.hpp>
TEST(ToylibIntegrationTestCompareOutput, sparseDimensions) {
  auto mesh = toylib::Grid(10, 10);
  const int edgesPerCell = 3;
  const int nb_levels = 1;

  toylib::FaceData<double> cells(mesh, nb_levels);
  toylib::EdgeData<double> edges(mesh, nb_levels);
  toylib::SparseFaceData<double> sparseDim(mesh, edgesPerCell, nb_levels);

  InitSparseData(mesh.faces(), sparseDim, edgesPerCell, nb_levels, 200.);
  InitData(mesh.edges(), edges, nb_levels, 1.);

  dawn_generated::cxxnaiveico::sparseDimension<toylibInterface::toylibTag>(mesh, nb_levels, cells,
                                                                           edges, sparseDim)
      .run();
  // each edge stores 1                                         1
  // this is multiplied by the sparse dim storing 200         200
  // this is reduced by sum onto the cells at 3 eges p cell   600
  for(const auto& f : mesh.faces()) {
    EXPECT_TRUE(fabs(cells(f, 0) - 600) < 1e3 * std::numeric_limits<double>::epsilon());
  }
}

#include <generated_NestedSparse.hpp>
TEST(ToylibIntegrationTestCompareOutput, nestedReduceSparseDimensions) {
  auto mesh = toylib::Grid(10, 10);
  const int edgesPerCell = 3;
  const int verticesPerEdge = 2;
  const int nb_levels = 1;

  toylib::FaceData<double> cells(mesh, nb_levels);
  toylib::EdgeData<double> edges(mesh, nb_levels);
  toylib::VertexData<double> vertices(mesh, nb_levels);

  toylib::SparseFaceData<double> sparseDim_ce(mesh, edgesPerCell, nb_levels);
  toylib::SparseEdgeData<double> sparseDim_ev(mesh, verticesPerEdge, nb_levels);

  InitSparseData(mesh.faces(), sparseDim_ce, edgesPerCell, nb_levels, 200.);
  InitSparseData(mesh.edges(), sparseDim_ev, verticesPerEdge, nb_levels, 300.);
  InitData(mesh.edges(), edges, nb_levels, 1.);
  InitData(mesh.vertices(), vertices, nb_levels, 2.);

  dawn_generated::cxxnaiveico::nestedWithSparse<toylibInterface::toylibTag>(
      mesh, nb_levels, cells, edges, vertices, sparseDim_ce, sparseDim_ev)
      .run();

  // each vertex stores 2                                                            2
  // this is multiplied by the sparse dim storing 300                              300
  // this is reduced by sum onto edges at 2 verts p edge                          1200
  // each edge stores 1                                                              1
  // this is multiplied by the reduction times the sparse dim storing 200          200
  // this is reduced by sum onto the cells at 3 eges p cell                       4200
  for(const auto& f : mesh.faces()) {
    EXPECT_TRUE(fabs(cells(f, 0) - 4200) < 1e3 * std::numeric_limits<double>::epsilon());
  }
}

#include <generated_SparseAssignment0.hpp>
TEST(ToylibIntegrationTestCompareOutput, SparseAssignment0) {
  auto mesh = toylib::Grid(10, 10);
  const int diamondSize = 4;
  const int nb_levels = 1;

  toylib::SparseEdgeData<double> vn_f(mesh, diamondSize, nb_levels);
  toylib::VertexData<double> uVert_f(mesh, nb_levels);
  toylib::VertexData<double> vVert_f(mesh, nb_levels);
  toylib::VertexData<double> nx_f(mesh, nb_levels);
  toylib::VertexData<double> ny_f(mesh, nb_levels);

  InitData(mesh.vertices(), uVert_f, nb_levels, 1.);
  InitData(mesh.vertices(), vVert_f, nb_levels, 2.);
  InitData(mesh.vertices(), nx_f, nb_levels, 3.);
  InitData(mesh.vertices(), ny_f, nb_levels, 4.);
  // dot product: vn(e,:) = u*nx + v*ny = 1*3 + 2*4 = 11

  dawn_generated::cxxnaiveico::sparseAssignment0<toylibInterface::toylibTag>(
      mesh, nb_levels, vn_f, uVert_f, vVert_f, nx_f, ny_f)
      .run();

  for(size_t level = 0; level < nb_levels; level++) {
    for(const auto& e : mesh.edges()) {
      int curDiamondSize = (e.get().faces().size() == 2) ? 4 : 3;
      for(size_t sparse = 0; sparse < curDiamondSize; sparse++) {
        EXPECT_TRUE(fabs(vn_f(e, sparse, level) - 11.) <
                    1e3 * std::numeric_limits<double>::epsilon());
      }
    }
  }
}

#include <generated_SparseAssignment1.hpp>
TEST(ToylibIntegrationTestCompareOutput, SparseAssignment1) {
  auto mesh = toylib::Grid(10, 10);
  const int diamondSize = 4;
  const int nb_levels = 1;

  toylib::SparseEdgeData<double> vn_f(mesh, diamondSize, nb_levels);
  toylib::SparseEdgeData<double> nx_f(mesh, diamondSize, nb_levels);
  toylib::SparseEdgeData<double> ny_f(mesh, diamondSize, nb_levels);
  toylib::VertexData<double> uVert_f(mesh, nb_levels);
  toylib::VertexData<double> vVert_f(mesh, nb_levels);

  InitSparseData(mesh.edges(), nx_f, diamondSize, nb_levels, 1.);
  InitSparseData(mesh.edges(), ny_f, diamondSize, nb_levels, 2.);
  InitData(mesh.vertices(), uVert_f, nb_levels, 3.);
  InitData(mesh.vertices(), vVert_f, nb_levels, 4.);
  // dot product: vn(e,:) = u*nx + v*ny = 1*3 + 2*4 = 11

  dawn_generated::cxxnaiveico::sparseAssignment1<toylibInterface::toylibTag>(
      mesh, nb_levels, vn_f, uVert_f, vVert_f, nx_f, ny_f)
      .run();

  for(size_t level = 0; level < nb_levels; level++) {
    for(const auto& e : mesh.edges()) {
      int curDiamondSize = (e.get().faces().size() == 2) ? 4 : 3;
      for(size_t sparse = 0; sparse < curDiamondSize; sparse++) {
        EXPECT_TRUE(fabs(vn_f(e, sparse, level) - 11.) <
                    1e3 * std::numeric_limits<double>::epsilon());
      }
    }
  }
}

#include <generated_SparseAssignment2.hpp>
TEST(ToylibIntegrationTestCompareOutput, SparseAssignment2) {
  auto mesh = toylib::Grid(10, 10);
  const int diamondSize = 4;
  const int nb_levels = 1;

  toylib::SparseEdgeData<double> sparse_f(mesh, diamondSize, nb_levels);
  toylib::EdgeData<double> e_f(mesh, nb_levels);
  toylib::VertexData<double> v_f(mesh, nb_levels);

  InitData(mesh.edges(), e_f, nb_levels, 1.);
  InitData(mesh.vertices(), v_f, nb_levels, 2.);

  // loop(e->c->v) {
  //  sparse_f = -4. * e_f(false) + v_f(true)
  // }
  dawn_generated::cxxnaiveico::sparseAssignment2<toylibInterface::toylibTag>(mesh, nb_levels,
                                                                             sparse_f, e_f, v_f)
      .run();

  for(size_t level = 0; level < nb_levels; level++) {
    for(const auto& e : mesh.edges()) {
      int curDiamondSize = (e.get().faces().size() == 2) ? 4 : 3;
      for(size_t sparse = 0; sparse < curDiamondSize; sparse++) {
        EXPECT_TRUE(fabs(sparse_f(e, sparse, level) - (-2.)) <
                    1e3 * std::numeric_limits<double>::epsilon());
      }
    }
  }
}

#include <generated_SparseAssignment3.hpp>
TEST(ToylibIntegrationTestCompareOutput, SparseAssignment3) {
  auto mesh = toylib::Grid(10, 10);
  const int intpSize = 9;
  const int nb_levels = 1;

  toylib::SparseFaceData<double> sparse_f(mesh, intpSize, nb_levels);
  toylib::FaceData<double> A_f(mesh, nb_levels);
  toylib::FaceData<double> B_f(mesh, nb_levels);

  InitSparseData(mesh.faces(), sparse_f, intpSize, nb_levels, 1.);
  InitData(mesh.faces(), A_f, nb_levels, 1.);
  for(const auto f : mesh.faces()) {
    B_f(f, 0) = f.id();
  }

  dawn_generated::cxxnaiveico::sparseAssignment3<toylibInterface::toylibTag>(mesh, nb_levels,
                                                                             sparse_f, A_f, B_f)
      .run();

  for(size_t level = 0; level < nb_levels; level++) {
    for(const auto& f : mesh.faces()) {
      size_t curIntpSize =
          toylibInterface::getNeighbors(toylibInterface::toylibTag{}, mesh,
                                        {dawn::LocationType::Cells, dawn::LocationType::Edges,
                                         dawn::LocationType::Cells, dawn::LocationType::Edges,
                                         dawn::LocationType::Cells},
                                        &f)
              .size();
      for(size_t sparse = 0; sparse < curIntpSize; sparse++) {
        EXPECT_TRUE(fabs(sparse_f(f, sparse, 0) - (1 - f.id())) <
                    1e3 * std::numeric_limits<double>::epsilon());
      }
    }
  }
}

#include <generated_SparseAssignment4.hpp>
TEST(ToylibIntegrationTestCompareOutput, SparseAssignment4) {
  auto mesh = toylib::Grid(10, 10);
  const int edgesPerCell = 3;
  const int nb_levels = 1;

  toylib::SparseFaceData<double> sparse_f(mesh, edgesPerCell, nb_levels);
  toylib::VertexData<double> e_f(mesh, nb_levels);

  InitData(mesh.vertices(), e_f, nb_levels, 1.);
  dawn_generated::cxxnaiveico::sparseAssignment4<toylibInterface::toylibTag>(mesh, nb_levels,
                                                                             sparse_f, e_f)
      .run();
  // reduce value 1 from vertices to edge (=2), assign to sparse dim

  for(size_t level = 0; level < nb_levels; level++) {
    for(const auto& f : mesh.faces()) {
      for(size_t sparse = 0; sparse < edgesPerCell; sparse++) {
        EXPECT_TRUE(fabs(sparse_f(f, sparse, 0) - 2.) <
                    1e3 * std::numeric_limits<double>::epsilon());
      }
    }
  }
}

#include <generated_SparseAssignment5.hpp>
TEST(ToylibIntegrationTestCompareOutput, SparseAssignment5) {
  auto mesh = toylib::Grid(10, 10);
  const int edgesPerCell = 3;
  const int nb_levels = 1;

  toylib::SparseFaceData<double> sparse_f(mesh, edgesPerCell, nb_levels);
  toylib::VertexData<double> v_f(mesh, nb_levels);
  toylib::FaceData<double> c_f(mesh, nb_levels);

  InitData(mesh.vertices(), v_f, nb_levels, 3.);
  InitData(mesh.faces(), c_f, nb_levels, 2.);
  dawn_generated::cxxnaiveico::sparseAssignment5<toylibInterface::toylibTag>(mesh, nb_levels,
                                                                             sparse_f, v_f, c_f)
      .run();

  // reduce value 2 from cells to vertex (=12) and multiply by vertex field(=36) reduce this onto
  // edges (two vertices per edge = 72)
  for(size_t level = 0; level < nb_levels; level++) {
    for(const auto& f : mesh.faces()) {
      for(size_t sparse = 0; sparse < edgesPerCell; sparse++) {

        auto e = f.edges()[sparse];
        double sol = e->vertices()[0]->faces().size() * 6 + e->vertices()[1]->faces().size() * 6;

        EXPECT_TRUE(fabs(sparse_f(f, sparse, 0) - sol) <
                    1e3 * std::numeric_limits<double>::epsilon());
      }
    }
  }
}

#include <generated_sparseDimensionTwice.hpp>
TEST(ToylibIntegrationTestCompareOutput, sparseDimensionsTwice) {
  auto mesh = toylib::Grid(10, 10);
  const int edgesPerCell = 3;
  const int nb_levels = 1;

  toylib::FaceData<double> cells(mesh, nb_levels);
  toylib::EdgeData<double> edges(mesh, nb_levels);
  toylib::SparseFaceData<double> sparseDim(mesh, edgesPerCell, nb_levels);

  InitSparseData(mesh.faces(), sparseDim, edgesPerCell, nb_levels, 200.);
  InitData(mesh.edges(), edges, nb_levels, 1.);

  dawn_generated::cxxnaiveico::sparseDimensionTwice<toylibInterface::toylibTag>(
      mesh, nb_levels, cells, edges, sparseDim)
      .run();
  // each edge stores 1                                         1
  // this is multiplied by the sparse dim storing 200         200
  // this is reduced by sum onto the cells at 3 eges p cell   600
  for(const auto& f : mesh.faces()) {
    EXPECT_TRUE(fabs(cells(f, 0) - 600) < 1e3 * std::numeric_limits<double>::epsilon());
  }
}

} // namespace
