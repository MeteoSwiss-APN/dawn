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

#ifndef GTCLANG_UNITTEST_IRSPLITTER_H
#define GTCLANG_UNITTEST_IRSPLITTER_H

#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/Pass.h"
#include "dawn/Optimizer/PassInlining.h"
#include "dawn/Optimizer/PassFieldVersioning.h"
#include "dawn/Optimizer/PassMultiStageSplitter.h"
#include "dawn/Optimizer/PassStageSplitter.h"
#include "dawn/Optimizer/PassTemporaryType.h"
#include "dawn/Optimizer/PassFixVersionedInputFields.h"
#include "dawn/Optimizer/PassComputeStageExtents.h"
#include "dawn/Optimizer/PassSetSyncStage.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/DiagnosticsEngine.h"
#include "dawn/Support/NonCopyable.h"
#include "gtclang/Unittest/GTClang.h"

#include <string>
#include <vector>

namespace gtclang {

/// @brief Emulate invocation of GTClang from command-line
/// @ingroup unittest
class IRSplitter : dawn::NonCopyable {
public:
  /// @brief Run GTClang with given flags
  ///
  /// @return a pair of a shared pointer to the SIR and a boolean `true` on success, `false`
  /// otherwise
  void split(const std::string& dslFile);
};

} // namespace gtclang

#endif
