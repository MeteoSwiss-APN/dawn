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

#include <gtest/gtest.h>

#include "interface/toylib_interface.hpp"
#include "toylib/toylib.hpp"

namespace {

// compare two (partial neighborhoods)
bool nbhsValidAndEqual(std::vector<const toylib::ToylibElement*> a,
                       std::vector<const toylib::ToylibElement*> b) {
  // need to have same length
  if(!(a.size() == b.size())) {
    return false;
  }

  // same elements, but order may be different
  std::sort(a.begin(), a.end(),
            [](const toylib::ToylibElement* left, const toylib::ToylibElement* right) {
              return left->id() < right->id();
            });
  std::sort(b.begin(), b.end(),
            [](const toylib::ToylibElement* left, const toylib::ToylibElement* right) {
              return left->id() < right->id();
            });

  if(!(a == b)) {
    return false;
  }

  // unique elements only
  auto isUnique = [](std::vector<const toylib::ToylibElement*> in) -> bool {
    auto it = std::unique(in.begin(), in.end());
    return (it == in.end());
  };

  return (isUnique(a) && isUnique(b));
}

TEST(TestToylibInterface, Diamond) {
  int w = 10;
  toylib::Grid mesh(w, w, false, 1., 1., true);
  std::vector<dawn::LocationType> chain{dawn::LocationType::Edges, dawn::LocationType::Cells,
                                        dawn::LocationType::Vertices};
  int testIdx = w * w / 2 + w / 2;
  toylib::Edge e = mesh.edges()[testIdx];
  std::vector<const toylib::ToylibElement*> diamond =
      toylibInterface::getNeighbors(mesh, chain, &e);
  ASSERT_TRUE(diamond.size() == 4);

  std::vector<const toylib::ToylibElement*> diamondLo{diamond[0], diamond[1]};
  std::vector<const toylib::ToylibElement*> diamondHi{diamond[2], diamond[3]};

  std::vector<const toylib::ToylibElement*> diamondLoRef{e.vertices()[0], e.vertices()[1]};
  std::vector<const toylib::ToylibElement*> diamondHiRef;
  for(auto f : e.faces()) {
    for(auto v : f->vertices()) {
      if(find(diamondLoRef.begin(), diamondLoRef.end(), v) == diamondLoRef.end() &&
         find(diamondHiRef.begin(), diamondHiRef.end(), v) == diamondHiRef.end()) {
        diamondHiRef.push_back(v);
      }
    }
  }
  ASSERT_TRUE(nbhsValidAndEqual(diamondLo, diamondLoRef));
  ASSERT_TRUE(nbhsValidAndEqual(diamondHi, diamondHiRef));
}

TEST(TestToylibInterface, Star) {
  int w = 10;
  toylib::Grid mesh(w, w, false, 1., 1., true);
  std::vector<dawn::LocationType> chain{
      dawn::LocationType::Vertices,
      dawn::LocationType::Cells,
      dawn::LocationType::Edges,
      dawn::LocationType::Cells,
  };
  int testIdx = w * w / 2 + w;
  toylib::Vertex v = mesh.vertices()[testIdx];
  std::vector<const toylib::ToylibElement*> star = toylibInterface::getNeighbors(mesh, chain, &v);
  ASSERT_TRUE(star.size() == 12);

  std::vector<const toylib::ToylibElement*> starLo{star.begin(), star.begin() + 6};
  std::vector<const toylib::ToylibElement*> starHi{star.begin() + 6, star.end()};

  std::vector<const toylib::ToylibElement*> starLoRef;
  for(auto f : v.faces()) {
    starLoRef.push_back(f);
  }
  std::vector<const toylib::ToylibElement*> starHiRef;
  for(auto f : v.faces()) {
    for(auto innerF : f->faces()) {
      if(find(starLoRef.begin(), starLoRef.end(), innerF) == starLoRef.end() &&
         find(starHiRef.begin(), starHiRef.end(), innerF) == starHiRef.end()) {
        starHiRef.push_back(innerF);
      }
    }
  }

  ASSERT_TRUE(nbhsValidAndEqual(starLo, starLoRef));
  ASSERT_TRUE(nbhsValidAndEqual(starHi, starHiRef));
}

TEST(TestToylibInterface, Fan) {
  int w = 10;
  toylib::Grid mesh(w, w, false, 1., 1., true);
  std::vector<dawn::LocationType> chain{
      dawn::LocationType::Vertices,
      dawn::LocationType::Cells,
      dawn::LocationType::Edges,
  };
  int testIdx = (w + 3) * w / 2 + w / 2 + 4;
  toylib::Vertex v = mesh.vertices()[testIdx];
  std::vector<const toylib::ToylibElement*> fan = toylibInterface::getNeighbors(mesh, chain, &v);
  ASSERT_TRUE(fan.size() == 12);

  std::vector<const toylib::ToylibElement*> fanLo{fan.begin(), fan.begin() + 6};
  std::vector<const toylib::ToylibElement*> fanHi{fan.begin() + 6, fan.end()};

  std::vector<const toylib::ToylibElement*> fanLoRef;
  for(auto e : v.edges()) {
    fanLoRef.push_back(e);
  }
  std::vector<const toylib::ToylibElement*> fanHiRef;
  for(auto f : v.faces()) {
    for(auto e : f->edges()) {
      if(find(fanLoRef.begin(), fanLoRef.end(), e) == fanLoRef.end() &&
         find(fanHiRef.begin(), fanHiRef.end(), e) == fanHiRef.end()) {
        fanHiRef.push_back(e);
      }
    }
  }

  ASSERT_TRUE(nbhsValidAndEqual(fanLo, fanLoRef));
  ASSERT_TRUE(nbhsValidAndEqual(fanHi, fanHiRef));
}

TEST(TestToylibInterface, Intp) {
  int w = 10;
  toylib::Grid mesh(w, w, false, 1., 1., true);
  std::vector<dawn::LocationType> chain{dawn::LocationType::Cells, dawn::LocationType::Edges,
                                        dawn::LocationType::Cells, dawn::LocationType::Edges,
                                        dawn::LocationType::Cells};
  int testIdx = 4 * w + w / 2;

  toylib::Face c = mesh.faces()[testIdx];
  std::vector<const toylib::ToylibElement*> intp = toylibInterface::getNeighbors(mesh, chain, &c);
  assert(intp.size() == 9);

  std::vector<const toylib::ToylibElement*> intpLo{intp.begin(), intp.begin() + 3};
  std::vector<const toylib::ToylibElement*> intpHi{intp.begin() + 3, intp.end()};

  std::vector<const toylib::ToylibElement*> intpLoRef;
  for(auto f : c.faces()) {
    intpLoRef.push_back(f);
  }
  std::vector<const toylib::ToylibElement*> intpHiRef;
  for(auto f : c.faces()) {
    for(auto innerF : f->faces()) {
      if(find(intpLoRef.begin(), intpLoRef.end(), innerF) == intpLoRef.end() &&
         find(intpHiRef.begin(), intpHiRef.end(), innerF) == intpHiRef.end()) {
        if(innerF->id() != c.id()) {
          intpHiRef.push_back(innerF);
        }
      }
    }
  }

  ASSERT_TRUE(nbhsValidAndEqual(intpLo, intpLoRef));
  ASSERT_TRUE(nbhsValidAndEqual(intpHi, intpHiRef));
}

} // namespace