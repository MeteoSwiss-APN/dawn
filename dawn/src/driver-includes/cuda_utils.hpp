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

#define gpuErrchk(ans)                                                                             \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if(code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if(abort)
      exit(code);
  }
}

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