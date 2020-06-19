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

#include "dawn/IIR/Cache.h"
#include "dawn/IIR/Stage.h"
#include "dawn/Support/Array.h"
#include "dawn/Support/IndexRange.h"
#include <map>
#include <string>

namespace dawn {
namespace iir {
class MultiStage;
class StencilInstantiation;
class Stencil;
class StencilMetaInformation;
class Field;
} // namespace iir
namespace codegen {
namespace cuda {

class CodeGeneratorHelper {
public:
  enum class FunctionArgType { FT_Caller, FT_Callee };

  static std::string generateStrideName(int dim, Array3i fieldDims);
  static std::string indexIteratorName(Array3i dims);
  static void generateFieldAccessDeref(std::stringstream& ss,
                                       const std::unique_ptr<iir::MultiStage>& ms,
                                       const iir::StencilMetaInformation& metadata,
                                       const int accessID,
                                       const std::unordered_map<int, Array3i> fieldIndexMap,
                                       ast::Offsets const& offset);
  ///
  /// @brief produces a string of (i,j,k) accesses for the C++ generated naive code,
  /// from an array of offseted accesses
  ///
  static std::array<std::string, 3> ijkfyOffset(const ast::Offsets& offset, bool isTemporary,
                                                const Array3i iteratorDims);

  /// @brief determines wheter an accessID will perform an access to main memory
  static bool hasAccessIDMemAccess(const int accessID,
                                   const std::unique_ptr<iir::Stencil>& stencil);

  /// @brief return true if the ms can be solved in parallel (in the vertical dimension)
  static bool solveKLoopInParallel(const std::unique_ptr<iir::MultiStage>& ms);

  /// @brief computes the partition of all the intervals used within a multi-stage
  static std::vector<iir::Interval>
  computePartitionOfIntervals(const std::unique_ptr<iir::MultiStage>& ms);

  /// @brief determines whether for code generation, using temporaries will be required.
  /// Even if the stencil contains temporaries, in some cases, like when they are local cached, they
  /// are not required for code generation. Also in the case of no redundant computations,
  /// temporaries will become normal fields
  static bool useTemporaries(const std::unique_ptr<iir::Stencil>& stencil,
                             const iir::StencilMetaInformation& metadata);

  /// @brief computes the maximum extent required by all temporaries, which will be used for proper
  /// allocation
  static iir::Extents computeTempMaxWriteExtent(iir::Stencil const& stencil);

  static std::vector<std::string>
  generateStrideArguments(const IndexRange<const std::map<int, iir::Field>>& nonTempFields,
                          const IndexRange<const std::map<int, iir::Field>>& tempFields,
                          const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                          const std::unique_ptr<iir::MultiStage>& ms,
                          CodeGeneratorHelper::FunctionArgType funArg);

  /// @brief compose the cuda kernel name of a stencil instantiation
  static std::string
  buildCudaKernelName(const std::shared_ptr<iir::StencilInstantiation>& instantiation,
                      const std::unique_ptr<iir::MultiStage>& ms);

  /// @brief compose the cuda kernel name of a stencil instantiation
  static std::string
  buildCudaKernelName(const std::shared_ptr<iir::StencilInstantiation>& instantiation,
                      const std::unique_ptr<iir::MultiStage>& ms,
                      const std::unique_ptr<iir::Stage>& stage);
};

} // namespace cuda
} // namespace codegen
} // namespace dawn
