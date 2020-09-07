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

#pragma once

#include "defs.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include "unstructured_interface.hpp"

#include <assert.h>
#include <map>
#include <vector>

#define DEVICE_MISSING_VALUE -1

#define gpuErrchk(ans)                                                                             \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if(code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if(abort)
      exit(code);
  }
}

namespace dawn {

struct GlobalGpuTriMesh {
  int NumEdges;
  int NumCells;
  int NumVertices;
  std::map<std::vector<dawn::LocationType>, int*> NeighborTables;
};

// Tag for no library (raw pointers)
// TODO this is a temporary HACK to keep the templated interface and have a constructor from raw
// pointers (ICON). Needs refactoring.
struct NoLibTag {};
dawn::GlobalGpuTriMesh meshType(NoLibTag);
int indexType(NoLibTag);
template <typename T>
::dawn::float_type cellFieldType(NoLibTag);
template <typename T>
::dawn::float_type edgeFieldType(NoLibTag);
template <typename T>
::dawn::float_type vertexFieldType(NoLibTag);
template <typename T>
::dawn::float_type sparseCellFieldType(NoLibTag);
template <typename T>
::dawn::float_type sparseEdgeFieldType(NoLibTag);
template <typename T>
::dawn::float_type sparseVertexFieldType(NoLibTag);
template <typename T>
::dawn::float_type verticalFieldType(NoLibTag);
// ENDTODO

inline void reshape(const dawn::float_type* input, dawn::float_type* output, int kSize,
                    int numElements, int sparseSize) {
  // In: edges, klevels, sparse
  // Out: klevels, sparse, edges

  for(int elIdx = 0; elIdx < numElements; elIdx++)
    for(int kLevel = 0; kLevel < kSize; kLevel++)
      for(int sparseIdx = 0; sparseIdx < sparseSize; sparseIdx++) {
        output[kLevel * numElements * sparseSize + sparseIdx * numElements + elIdx] =
            input[elIdx * kSize * sparseSize + kLevel * sparseSize + sparseIdx];
      }
}

inline void reshape(const dawn::float_type* input, dawn::float_type* output, int kSize,
                    int numElements) {
  // In: edges, klevels
  // Out: klevels, edges

  for(int elIdx = 0; elIdx < numElements; elIdx++)
    for(int kLevel = 0; kLevel < kSize; kLevel++) {
      output[kLevel * numElements + elIdx] = input[elIdx * kSize + kLevel];
    }
}

inline void reshape_back(const dawn::float_type* input, dawn::float_type* output, int kSize,
                         int numElements) {
  // In: klevels, edges
  // Out: edges, klevels

  for(int elIdx = 0; elIdx < numElements; elIdx++)
    for(int kLevel = 0; kLevel < kSize; kLevel++) {
      output[elIdx * kSize + kLevel] = input[kLevel * numElements + elIdx];
    }
}
inline void reshape_back(const dawn::float_type* input, dawn::float_type* output, int kSize,
                         int numElements, int sparseSize) {
  // In: klevels, sparse, edges
  // Out: edges, klevels, sparse
  for(int elIdx = 0; elIdx < numElements; elIdx++)
    for(int kLevel = 0; kLevel < kSize; kLevel++)
      for(int sparseIdx = 0; sparseIdx < sparseSize; sparseIdx++) {
        output[elIdx * kSize * sparseSize + kLevel * sparseSize + sparseIdx] =
            input[kLevel * numElements * sparseSize + sparseIdx * numElements + elIdx];
      }
}

template <class FieldT>
void initField(const FieldT& field, dawn::float_type** cudaStorage, int kSize) {
  gpuErrchk(cudaMalloc((void**)cudaStorage, sizeof(dawn::float_type) * field.numElements()));
  gpuErrchk(cudaMemcpy(*cudaStorage, field.data(), sizeof(dawn::float_type) * field.numElements(),
                       cudaMemcpyHostToDevice));
}
template <class FieldT>
void initField(const FieldT& field, dawn::float_type** cudaStorage, int denseSize, int kSize,
               bool doReshape) {
  gpuErrchk(cudaMalloc((void**)cudaStorage, sizeof(dawn::float_type) * field.numElements()));
  if(doReshape) {
    dawn::float_type* reshaped = new dawn::float_type[field.numElements()];
    reshape(field.data(), reshaped, kSize, denseSize);
    gpuErrchk(cudaMemcpy(*cudaStorage, reshaped, sizeof(dawn::float_type) * field.numElements(),
                         cudaMemcpyHostToDevice));
    delete[] reshaped;
  } else {
    gpuErrchk(cudaMemcpy(*cudaStorage, field.data(), sizeof(dawn::float_type) * field.numElements(),
                         cudaMemcpyHostToDevice));
  }
}
template <class SparseFieldT>
void initSparseField(const SparseFieldT& field, dawn::float_type** cudaStorage, int denseSize,
                     int sparseSize, int kSize, bool doReshape) {
  gpuErrchk(cudaMalloc((void**)cudaStorage, sizeof(dawn::float_type) * field.numElements()));
  if(doReshape) {
    dawn::float_type* reshaped = new dawn::float_type[field.numElements()];
    reshape(field.data(), reshaped, kSize, denseSize, sparseSize);
    gpuErrchk(cudaMemcpy(*cudaStorage, reshaped, sizeof(dawn::float_type) * field.numElements(),
                         cudaMemcpyHostToDevice));
    delete[] reshaped;
  } else {
    gpuErrchk(cudaMemcpy(*cudaStorage, field.data(), sizeof(dawn::float_type) * field.numElements(),
                         cudaMemcpyHostToDevice));
  }
}

inline void initField(::dawn::float_type* field, dawn::float_type** cudaStorage, int kSize) {
  gpuErrchk(cudaMalloc((void**)cudaStorage, sizeof(dawn::float_type) * kSize));
  gpuErrchk(
      cudaMemcpy(*cudaStorage, field, sizeof(dawn::float_type) * kSize, cudaMemcpyHostToDevice));
}
inline void initField(::dawn::float_type* field, dawn::float_type** cudaStorage, int denseSize,
                      int kSize, bool doReshape) {
  const int numElements = denseSize * kSize;
  gpuErrchk(cudaMalloc((void**)cudaStorage, sizeof(dawn::float_type) * numElements));
  if(doReshape) {
    dawn::float_type* reshaped = new dawn::float_type[numElements];
    reshape(field, reshaped, kSize, denseSize);
    gpuErrchk(cudaMemcpy(*cudaStorage, reshaped, sizeof(dawn::float_type) * numElements,
                         cudaMemcpyHostToDevice));
    delete[] reshaped;
  } else {
    gpuErrchk(cudaMemcpy(*cudaStorage, field, sizeof(dawn::float_type) * numElements,
                         cudaMemcpyHostToDevice));
  }
}
inline void initSparseField(::dawn::float_type*& field, dawn::float_type** cudaStorage,
                            int denseSize, int sparseSize, int kSize, bool doReshape) {
  const int numElements = denseSize * sparseSize * kSize;
  gpuErrchk(cudaMalloc((void**)cudaStorage, sizeof(dawn::float_type) * numElements));
  if(doReshape) {
    dawn::float_type* reshaped = new dawn::float_type[numElements];
    reshape(field, reshaped, kSize, denseSize, sparseSize);
    gpuErrchk(cudaMemcpy(*cudaStorage, reshaped, sizeof(dawn::float_type) * numElements,
                         cudaMemcpyHostToDevice));
    delete[] reshaped;
  } else {
    gpuErrchk(cudaMemcpy(*cudaStorage, field, sizeof(dawn::float_type) * numElements,
                         cudaMemcpyHostToDevice));
  }
}

template <typename LibTag>
void generateNbhTable(dawn::mesh_t<LibTag> const& mesh, std::vector<dawn::LocationType> chain,
                      int numElements, int numNbhPerElement, int* target) {
  std::vector<dawn::nbh_table_index_t<LibTag>> elems;
  switch(chain.front()) {
  case dawn::LocationType::Cells: {
    for(auto cell : getCells(LibTag{}, mesh)) {
      elems.push_back(cell);
    }
    break;
  }
  case dawn::LocationType::Edges: {
    for(auto edge : getEdges(LibTag{}, mesh)) {
      elems.push_back(edge);
    }
    break;
  }
  case dawn::LocationType::Vertices: {
    for(auto vertex : getVertices(LibTag{}, mesh)) {
      elems.push_back(vertex);
    }
    break;
  }
  }

  assert(elems.size() == numElements);

  std::vector<int> hostTable;
  for(int elem : elems) {
    auto neighbors = getNeighbors(LibTag{}, mesh, chain, elem);
    for(int nbhIdx = 0; nbhIdx < numNbhPerElement; nbhIdx++) {
      if(nbhIdx < neighbors.size()) {
        hostTable.push_back(neighbors[nbhIdx]);
      } else {
        hostTable.push_back(DEVICE_MISSING_VALUE);
      }
    }
  }

  assert(hostTable.size() == numElements * numNbhPerElement);
  gpuErrchk(cudaMemcpy(target, hostTable.data(), sizeof(int) * numElements * numNbhPerElement,
                       cudaMemcpyHostToDevice));
}
} // namespace dawn