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

yoda_add_library(
  NAME DawnOptimizer
  SOURCES AccessComputation.h
          AccessComputation.cpp   
          AccessUtils.cpp
          AccessUtils.h
          BoundaryExtent.cpp  
          BoundaryExtent.h
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
  OBJECT
)

target_include_directories(DawnOptimizerObjects SYSTEM PUBLIC ${Boost_INCLUDE_DIR})

yoda_combine_libraries(
  NAME DawnOptimizer
  OBJECTS DawnOptimizerObjects
  DEPENDS DawnSupportStatic
  INSTALL_DESTINATION ${DAWN_INSTALL_LIB_DIR}
  VERSION ${DAWN_VERSION}
  EXPORT_GROUP DawnTargets
)

