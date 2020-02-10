##===------------------------------------------------------------------------------*- CMake -*-===##
##                          _
##                         | |
##                       __| | __ ___      ___ ___
##                      / _` |/ _` \ \ /\ / / '_  |
##                     | (_| | (_| |\ V  V /| | | |
##                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
##
##
##  This file is distributed under the MIT License (MIT).
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

add_library(DawnOptimizer
  CreateVersionAndRename.cpp
  CreateVersionAndRename.h
  OptimizerContext.cpp
  OptimizerContext.h
  Pass.h
  PassComputeStageExtents.cpp
  PassComputeStageExtents.h
  PassDataLocalityMetric.cpp
  PassDataLocalityMetric.h
  PassFieldVersioning.cpp
  PassFieldVersioning.h
  PassFixVersionedInputFields.cpp
  PassFixVersionedInputFields.h
  PassInlining.cpp
  PassInlining.h
  PassIntervalPartitioner.cpp
  PassIntervalPartitioner.h
  PassLocalVarType.h
  PassLocalVarType.cpp
  PassManager.cpp
  PassManager.h
  PassMultiStageSplitter.cpp
  PassMultiStageSplitter.h
  PassPrintStencilGraph.cpp
  PassPrintStencilGraph.h
  PassSetBlockSize.cpp
  PassSetBlockSize.h
  PassSetBoundaryCondition.cpp
  PassSetBoundaryCondition.h
  PassSetCaches.cpp
  PassSetCaches.h
  PassSetNonTempCaches.cpp
  PassSetNonTempCaches.h
  PassSetStageGraph.cpp
  PassSetStageGraph.h
  PassSetStageName.cpp
  PassSetStageName.h
  PassSetSyncStage.cpp
  PassSetSyncStage.h
  PassSSA.cpp
  PassSSA.h
  PassStageMerger.cpp
  PassStageMerger.h
  PassStageReordering.cpp
  PassStageReordering.h
  PassStageSplitter.cpp
  PassStageSplitter.h
  PassStencilSplitter.cpp
  PassStencilSplitter.h
  PassTemporaryFirstAccess.cpp
  PassTemporaryFirstAccess.h
  PassTemporaryMerger.cpp
  PassTemporaryMerger.h
  PassTemporaryType.cpp
  PassTemporaryType.h
  PassTemporaryToStencilFunction.cpp
  PassTemporaryToStencilFunction.h
  PassValidation.cpp
  PassValidation.h
  ReadBeforeWriteConflict.cpp
  ReadBeforeWriteConflict.h
  Renaming.cpp
  Renaming.h
  ReorderStrategy.h
  ReorderStrategyGreedy.cpp
  ReorderStrategyGreedy.h
  ReorderStrategyPartitioning.cpp
  ReorderStrategyPartitioning.h
  Replacing.cpp
  Replacing.h
  StatementMapper.cpp
  StatementMapper.h
  TemporaryHandling.cpp
  TemporaryHandling.h
)

target_add_dawn_standard_props(DawnOptimizer)
target_link_libraries(DawnOptimizer PUBLIC DawnSupport DawnIIR DawnValidator)
