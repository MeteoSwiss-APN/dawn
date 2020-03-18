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
std::tuple<double, double> edgeMidpoint(const toylib::Edge& e) {
  double x = e.vertices()[0]->x() + e.vertices()[1]->x();
  double y = e.vertices()[0]->y() + e.vertices()[1]->y();
  return {x / 2., y / 2.};
}

void debugDumpMesh(const toylib::Grid& mesh, const std::string prefix) {
  {
    char buf[256];
    sprintf(buf, "%sT.txt", prefix.c_str());
    FILE* fp = fopen(buf, "w+");
    for(const auto& cellIt : mesh.faces()) {
      int nodeIdx0 = cellIt.vertices()[0]->id();
      int nodeIdx1 = cellIt.vertices()[1]->id();
      int nodeIdx2 = cellIt.vertices()[2]->id();
      fprintf(fp, "%d %d %d\n", nodeIdx0, nodeIdx1, nodeIdx2);
    }
    fclose(fp);
  }

  {
    char buf[256];
    sprintf(buf, "%sP.txt", prefix.c_str());
    FILE* fp = fopen(buf, "w+");
    for(const auto& nodeIt : mesh.vertices()) {
      fprintf(fp, "%f %f \n", nodeIt.x(), nodeIt.y());
    }
    fclose(fp);
  }
}

} // namespace

namespace {
TEST(ToylibIntegrationTestCompareOutput, NbhTestTEMP) {
  int w = 10;
  toylib::Grid mesh(w, w, false, 1., 1., true);
  debugDumpMesh(mesh, "nbhToy");

  {
    std::vector<dawn::LocationType> chain{dawn::LocationType::Edges, dawn::LocationType::Cells,
                                          dawn::LocationType::Vertices};
    int testIdx = w * w / 2 + w / 2;
    toylib::Edge e = mesh.edges()[testIdx];
    auto [ex, ey] = edgeMidpoint(e);
    printf("%f %f\n", ex, ey);
    printf("-----\n");
    std::vector<const toylib::ToylibElement*> diamond =
        toylibInterface::getNeighbors(mesh, chain, &e);
    for(const auto it : diamond) {
      const toylib::Vertex* v = static_cast<const toylib::Vertex*>(it);
      printf("%f %f\n", v->x(), v->y());
    }
    printf("\n");
  }

  // {
  //   std::vector<dawn::LocationType> chain{
  //       dawn::LocationType::Vertices,
  //       dawn::LocationType::Cells,
  //       dawn::LocationType::Edges,
  //       dawn::LocationType::Cells,
  //   };
  //   int testIdx = nx * ny / 2 + ny / 2;
  //   printf("%f %f\n", xy(testIdx, atlas::LON), xy(testIdx, atlas::LAT));
  //   printf("-----\n");
  //   std::vector<int> star = atlasInterface::getNeighbors(mesh, chain, testIdx);
  //   for(const auto it : star) {
  //     auto [x, y] = cellMidpoint(mesh, it);
  //     printf("%f %f\n", x, y);
  //   }
  //   printf("\n");
  // }

  // {
  //   std::vector<dawn::LocationType> chain{
  //       dawn::LocationType::Vertices,
  //       dawn::LocationType::Cells,
  //       dawn::LocationType::Edges,
  //   };
  //   int testIdx = (nx + 3) * ny / 2 + ny / 2 + 4;
  //   printf("%f %f\n", xy(testIdx, atlas::LON), xy(testIdx, atlas::LAT));
  //   printf("-----\n");
  //   std::vector<int> fan = atlasInterface::getNeighbors(mesh, chain, testIdx);
  //   for(const auto it : fan) {
  //     auto [x, y] = edgeMidpoint(mesh, it);
  //     printf("%f %f\n", x, y);
  //   }
  //   printf("\n");
  // }

  // {
  //   std::vector<dawn::LocationType> chain{dawn::LocationType::Cells, dawn::LocationType::Edges,
  //                                         dawn::LocationType::Cells, dawn::LocationType::Edges,
  //                                         dawn::LocationType::Cells};
  //   int testIdx = 4 * nx;
  //   auto [x, y] = cellMidpoint(mesh, testIdx);
  //   printf("%f %f\n", x, y);
  //   printf("-----\n");
  //   std::vector<int> intp = atlasInterface::getNeighbors(mesh, chain, testIdx);
  //   for(const auto it : intp) {
  //     if(it == -1) {
  //       continue;
  //     }
  //     auto [x, y] = cellMidpoint(mesh, it);
  //     printf("%f %f\n", x, y);
  //   }
  //   printf("\n");
  // }
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
    AtlasVerifier v;
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

  dawn_generated::cxxnaiveico::reference_gradient<toylibInterface::toylibTag>(mesh, nb_levels,
                                                                              ref_cells, ref_edges)
      .run();
  dawn_generated::cxxnaiveico::gradient<toylibInterface::toylibTag>(mesh, nb_levels, gen_cells,
                                                                    gen_edges)
      .run();

  {
    AtlasVerifier v;
    EXPECT_TRUE(v.compareToylibField(mesh.faces(), ref_cells, gen_cells, nb_levels))
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
  // each face reduces its edges (4 per face)   6
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
  // each face reduces its edges (4 per face) 606
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
  // this is reduced by sum onto the cells at 4 eges p cell   600
  for(const auto& f : mesh.faces()) {
    EXPECT_TRUE(fabs(cells(f, 0) - 600) < 1e3 * std::numeric_limits<double>::epsilon());
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
  // this is reduced by sum onto the cells at 4 eges p cell   600
  for(const auto& f : mesh.faces()) {
    EXPECT_TRUE(fabs(cells(f, 0) - 600) < 1e3 * std::numeric_limits<double>::epsilon());
  }
}

} // namespace