##===------------------------------------------------------------------------------*- CMake -*-===##
##                                 ____ ____  _     
##                                / ___/ ___|| |    
##                               | |  _\___ \| |    
##                               | |_| |___) | |___ 
##                                \____|____/|_____| - Generic Stencil Language
##                  
##  This file is distributed under the MIT License (MIT). 
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

gsl_add_library(
  NAME GSLOptimizer
  SOURCES AccessComputation.h
          AccessComputation.cpp   
          Accesses.h
          Accesses.cpp  
          BoundaryExtent.h
          BoundaryExtent.cpp  
          Cache.h
          Cache.cpp
          DependencyGraph.h
          DependencyGraphAccesses.h
          DependencyGraphAccesses.cpp      
          DependencyGraphStage.h
          DependencyGraphStage.cpp
          DoMethod.h
          DoMethod.cpp
          Extents.h
          Extents.cpp
          Field.h
          Interval.h
          Interval.cpp
          LoopOrder.h     
          LoopOrder.cpp    
          MultiStage.h
          MultiStage.cpp 
          OptimizerContext.h
          OptimizerContext.cpp 
          Pass.h
          PassDataLocalityMetric.h
          PassDataLocalityMetric.cpp      
          PassFieldVersioning.h
          PassFieldVersioning.cpp
          PassPrintStencilGraph.h
          PassPrintStencilGraph.cpp
          PassInlining.h
          PassInlining.cpp
          PassMultiStageSplitter.h
          PassMultiStageSplitter.cpp
          PassManager.h 
          PassManager.cpp   
          PassSetStageName.h
          PassSetStageName.cpp
          PassSetStageGraph.h
          PassSetStageGraph.cpp
          PassSetCaches.h
          PassSetCaches.cpp
          PassSSA.h
          PassSSA.cpp
          PassStageMerger.h
          PassStageMerger.cpp
          PassStageSplitter.h
          PassStageSplitter.cpp
          PassStencilSplitter.h
          PassStencilSplitter.cpp
          PassStageReordering.h
          PassStageReordering.cpp
          PassTemporaryMerger.h
          PassTemporaryMerger.cpp
          PassTemporaryFirstAccess.h
          PassTemporaryFirstAccess.cpp
          PassTemporaryType.h
          PassTemporaryType.cpp
          ReadBeforeWriteConflict.h
          ReadBeforeWriteConflict.cpp
          ReorderStrategy.h
          ReorderStrategyGreedy.h
          ReorderStrategyGreedy.cpp
          ReorderStrategyPartitioning.h
          ReorderStrategyPartitioning.cpp
          Renaming.h
          Renaming.cpp
          Replacing.h
          Replacing.cpp
          Stage.h
          Stage.cpp
          Stencil.h
          Stencil.cpp
          StatementAccessesPair.h
          StatementAccessesPair.cpp
          StencilFunctionInstantiation.h
          StencilFunctionInstantiation.cpp
          StencilInstantiation.h
          StencilInstantiation.cpp
)
