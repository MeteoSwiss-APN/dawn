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

void reshape(const dawn::float_type* input, dawn::float_type* output, int kSize, int numEdges,
             int sparseSize) {
  // In: edges, klevels, sparse
  // Out: klevels, sparse, edges

  for(int edgeIdx = 0; edgeIdx < numEdges; edgeIdx++)
    for(int kLevel = 0; kLevel < kSize; kLevel++)
      for(int sparseIdx = 0; sparseIdx < sparseSize; sparseIdx++) {
        output[kLevel * numEdges * sparseSize + sparseIdx * numEdges + edgeIdx] =
            input[edgeIdx * kSize * sparseSize + kLevel * sparseSize + sparseIdx];
      }
}

void reshape(const dawn::float_type* input, dawn::float_type* output, int kSize, int numEdges) {
  // In: edges, klevels
  // Out: klevels, edges

  for(int edgeIdx = 0; edgeIdx < numEdges; edgeIdx++)
    for(int kLevel = 0; kLevel < kSize; kLevel++) {
      output[kLevel * numEdges + edgeIdx] = input[edgeIdx * kSize + kLevel];
    }
}

void reshape_back(const dawn::float_type* input, dawn::float_type* output, int kSize,
                  int numEdges) {
  // In: klevels, edges
  // Out: edges, klevels

  for(int edgeIdx = 0; edgeIdx < numEdges; edgeIdx++)
    for(int kLevel = 0; kLevel < kSize; kLevel++) {
      output[edgeIdx * kSize + kLevel] = input[kLevel * numEdges + edgeIdx];
    }
}
void reshape_back(const dawn::float_type* input, dawn::float_type* output, int kSize, int numEdges,
                  int sparseSize) {
  // In: klevels, sparse, edges
  // Out: edges, klevels, sparse
  for(int edgeIdx = 0; edgeIdx < numEdges; edgeIdx++)
    for(int kLevel = 0; kLevel < kSize; kLevel++)
      for(int sparseIdx = 0; sparseIdx < sparseSize; sparseIdx++) {
        output[edgeIdx * kSize * sparseSize + kLevel * sparseSize + sparseIdx] =
            input[kLevel * numEdges * sparseSize + sparseIdx * numEdges + edgeIdx];
      }
}

template <class FieldT>
void initField(const FieldT& field, dawn::float_type** cudaStorage, int denseSize, int kSize) {
  dawn::float_type* reshaped = new dawn::float_type[field.numElements()];
  reshape(field.data(), reshaped, kSize, denseSize);
  gpuErrchk(cudaMalloc((void**)cudaStorage, sizeof(dawn::float_type) * field.numElements()));
  gpuErrchk(cudaMemcpy(*cudaStorage, reshaped, sizeof(dawn::float_type) * field.numElements(),
                       cudaMemcpyHostToDevice));
  delete[] reshaped;
}
template <class SparseFieldT>
void initSparseField(const SparseFieldT& field, dawn::float_type** cudaStorage, int denseSize,
                     int sparseSize, int kSize) {
  dawn::float_type* reshaped = new dawn::float_type[field.numElements()];
  reshape(field.data(), reshaped, kSize, denseSize, sparseSize);
  gpuErrchk(cudaMalloc((void**)cudaStorage, sizeof(dawn::float_type) * field.numElements()));
  gpuErrchk(cudaMemcpy(*cudaStorage, reshaped, sizeof(dawn::float_type) * field.numElements(),
                       cudaMemcpyHostToDevice));
  delete[] reshaped;
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