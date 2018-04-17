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

#include "dawn/Optimizer/PassSetCaches.h"
#include "dawn/Optimizer/Cache.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/StatementAccessesPair.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/Support/Unreachable.h"
#include <iostream>
#include <set>
#include <vector>

namespace dawn {

namespace {

enum FirstAccessKind {
  FK_Unknown = 0,
  FK_WriteOnly = 1,   ///< The first access is write only (e.g `a = ...`)
  FK_ReadOnly = 2,    ///< The first access is read only (e.g `... = a`)
  FK_ReadAndWrite = 4 ///< The first access is read and write (e.g `a = a + ...`)
};

/// @brief Do we compute the first levels or do we need to access main memory?
///
/// If the first write access of the field has a read access to the same field as well, it is not
/// computed and we need to fill/flush the caches.
static FirstAccessKind getFirstAccessKind(const MultiStage& MS, int AccessID) {

  // Get the stage of the first access
  for(const auto& stagePtr : MS.getStages()) {
    Stage& stage = *stagePtr;
    if(std::find_if(stage.getFields().begin(), stage.getFields().end(), [&](const Field& field) {
         return field.getAccessID() == AccessID;
       }) != stage.getFields().end()) {

      auto getFirstAccessKindFromDoMethod = [&](const DoMethod* doMethod) -> FirstAccessKind {
        for(const auto& statementAccesssPair : doMethod->getStatementAccessesPairs()) {
          const Accesses& accesses = *statementAccesssPair->getAccesses();

          for(const auto& writeAccessIDExtentPair : accesses.getWriteAccesses()) {
            // Storage has a write access, does it have a read access as well?
            if(writeAccessIDExtentPair.first == AccessID) {
              for(const auto& readAccessIDExtentPair : accesses.getReadAccesses())
                // Yes!
                if(readAccessIDExtentPair.first == AccessID)
                  return FK_ReadAndWrite;
              // No!
              return FK_WriteOnly;
            }
          }

          for(const auto& readAccessIDExtentPair : accesses.getReadAccesses())
            // First access is read-only
            if(readAccessIDExtentPair.first == AccessID)
              return FK_ReadOnly;
        }
        // Storage not referenced in this Do-Method
        return FK_Unknown;
      };

      // We need to check all Do-Methods
      if(MS.getLoopOrder() == LoopOrderKind::LK_Parallel) {
        FirstAccessKind firstAccess = FK_Unknown;
        for(const auto& doMethodPtr : stage.getDoMethods())
          firstAccess = static_cast<FirstAccessKind>(
              firstAccess | getFirstAccessKindFromDoMethod(doMethodPtr.get()));
        return firstAccess;

      } else {
        // Forward and backward loop orders
        std::vector<DoMethod*> doMethods;
        std::transform(stage.getDoMethods().begin(), stage.getDoMethods().end(),
                       std::back_inserter(doMethods),
                       [](const std::unique_ptr<DoMethod>& D) { return D.get(); });

        if(MS.getLoopOrder() == LoopOrderKind::LK_Forward) {
          // Sort the Do-Method in ascending order (i.e lowest interval is first)
          std::sort(doMethods.begin(), doMethods.end(), [](DoMethod* D1, DoMethod* D2) {
            return D1->getInterval().lowerBound() < D2->getInterval().lowerBound();
          });
        } else {
          // Sort the Do-Method in descending order (i.e top interval is first)
          std::sort(doMethods.begin(), doMethods.end(), [](DoMethod* D1, DoMethod* D2) {
            return D1->getInterval().upperBound() > D2->getInterval().upperBound();
          });
        }

        for(DoMethod* D : doMethods) {
          FirstAccessKind firstAccess = getFirstAccessKindFromDoMethod(D);
          if(firstAccess != FK_Unknown)
            return firstAccess;
        }
      }
    }
  }
  dawn_unreachable("MultiStage does not contain the given field");
}

/// @brief Combine the policies from the first MultiStage (`MS1Policy`) and the immediately
/// following (`MS2Policy`)
///
/// MS1Policy has to be one of: {local, fill, bpfill}
/// MS2Policy has to be one of: {local, flush, epflush}
///
///                                       MS2
///
///                ||    local    ||     flush     ||   epflush    ||
///       =========++=============++===============++==============++
///        local   ||    local    |      flush     |    epflush     |
///       =========++-------------+----------------+----------------+
///  MS1   fill    ||    fill     | fill_and_flush | fill_and_flush |
///       =========++-------------+----------------+----------------+
///        bpfill  ||   pbfill    | fill_and_flush | fill_and_flush |
///       =========++-------------+----------------+----------------+
///
Cache::CacheIOPolicy combinePolicy(Cache::CacheIOPolicy MS1Policy, Cache::CacheIOPolicy MS2Policy) {
  if(MS1Policy == Cache::local) {
    if(MS2Policy == Cache::flush || MS2Policy == Cache::epflush)
      return MS2Policy; // flush, epflush
    else if(MS2Policy == Cache::local)
      return MS2Policy; // local
  } else if(MS1Policy == Cache::fill || MS1Policy == Cache::bpfill) {
    if(MS2Policy == Cache::flush || MS2Policy == Cache::epflush)
      return Cache::fill_and_flush;
    else if(MS2Policy == Cache::local)
      return MS1Policy; // fill, pbfill
  }
  dawn_unreachable("invalid policy combination");
}

} // anonymous namespace

PassSetCaches::PassSetCaches() : Pass("PassSetCaches") {}

bool PassSetCaches::run(const std::shared_ptr<StencilInstantiation>& instantiation) {
  OptimizerContext* context = instantiation->getOptimizerContext();

  for(const auto& stencilPtr : instantiation->getStencils()) {
    const Stencil& stencil = *stencilPtr;

    // Set IJ-Caches
    int msIdx = 0;
    for(const auto& multiStagePtr : stencil.getMultiStages()) {
      MultiStage& MS = *multiStagePtr;

      std::set<int> outputFields;

      auto isOutput = [](const Field& field) {
        return field.getIntend() == Field::IK_Output || field.getIntend() == Field::IK_InputOutput;
      };

      for(const auto& stagePtr : multiStagePtr->getStages()) {
        for(const Field& field : stagePtr->getFields()) {

          // Field is already cached, skip
          if(MS.isCached(field.getAccessID()))
            continue;

          // Field has vertical extents, can't be ij-cached
          if(!field.getExtents().isVerticalPointwise())
            continue;

          // Currently we only cache temporaries!
          if(!instantiation->isTemporaryField(field.getAccessID()))
            continue;

          // Cache the field
          if(field.getIntend() == Field::IK_Input && outputFields.count(field.getAccessID()) &&
             !field.getExtents().isHorizontalPointwise()) {

            Cache& cache = multiStagePtr->setCache(Cache::IJ, Cache::local, field.getAccessID());
            instantiation->insertCachedVariable(field.getAccessID());

            if(context->getOptions().ReportPassSetCaches) {
              std::cout << "\nPASS: " << getName() << ": " << instantiation->getName() << ": MS"
                        << msIdx << ": "
                        << instantiation->getOriginalNameFromAccessID(field.getAccessID()) << ":"
                        << cache.getCacheTypeAsString() << ":" << cache.getCacheIOPolicyAsString()
                        << std::endl;
            }
          }

          if(isOutput(field))
            outputFields.insert(field.getAccessID());
        }
      }
      msIdx++;
    }

    // Set K-Caches
    if(context->getOptions().UseKCaches ||
       stencil.getSIRStencil()->Attributes.has(sir::Attr::AK_UseKCaches)) {

      // Get the fields of all Multi-Stages
      std::vector<std::unordered_map<int, Field>> fields;
      std::transform(stencil.getMultiStages().begin(), stencil.getMultiStages().end(),
                     std::back_inserter(fields),
                     [](const std::shared_ptr<MultiStage>& MSPtr) { return MSPtr->getFields(); });

      int numMS = fields.size();
      for(int MSIndex = 0; MSIndex < numMS; ++MSIndex) {
        for(const auto& AccessIDFieldPair : fields[MSIndex]) {
          MultiStage& MS = *stencil.getMultiStageFromMultiStageIndex(MSIndex);
          const Field& field = AccessIDFieldPair.second;

          // Field is already cached, skip
          if(MS.isCached(field.getAccessID()))
            continue;

          // Field has horizontal extents, can't be k-cached
          if(!field.getExtents().isHorizontalPointwise())
            continue;

          // Currently we only cache temporaries!
          //          if(!instantiation->isTemporaryField(field.getAccessID()))
          //            continue;

          if(!instantiation->isTemporaryField(field.getAccessID()) &&
             (field.getIntend() == Field::IK_Output ||
              (field.getIntend() == Field::IK_Input && field.getExtents().isPointwise())))
            continue;

          // Determine if we need to fill the cache by analyzing the current multi-stage
          Cache::CacheIOPolicy policy = Cache::unknown;
          if(field.getIntend() == Field::IK_Input) {
            policy = Cache::fill;
          } else if(field.getIntend() == Field::IK_Output)
            policy = Cache::local;
          else if(field.getIntend() == Field::IK_InputOutput) {
            // Do we compute the first levels or do we need to access main memory (i.e fill the
            // fist accesses)?
            FirstAccessKind firstAccess = getFirstAccessKind(MS, field.getAccessID());
            if(firstAccess == FK_WriteOnly)
              policy = Cache::local;
            else {
              // Do we have a read in counter loop order or pointwise access?
              if(field.getExtents()
                     .getVerticalLoopOrderAccesses(MS.getLoopOrder())
                     .CounterLoopOrder ||
                 field.getExtents().isVerticalPointwise())
                policy = Cache::fill;
              else
                policy = Cache::bpfill;
            }
          }

          // Determine if we need to flush the cache by analyzing the next multi-stage
          if(MSIndex != (numMS - 1) && fields[MSIndex + 1].count(field.getAccessID())) {

            const MultiStage& nextMS = *stencil.getMultiStageFromMultiStageIndex(MSIndex + 1);
            const Field& fieldInNextMS = fields[MSIndex + 1].find(field.getAccessID())->second;

            if(fieldInNextMS.getIntend() == Field::IK_Input)
              policy = combinePolicy(policy, Cache::flush);
            else if(fieldInNextMS.getIntend() == Field::IK_InputOutput) {
              // Do we compute the first levels or do we need to access main memory (i.e flush
              // the last accesses)?
              FirstAccessKind firstAccess = getFirstAccessKind(nextMS, fieldInNextMS.getAccessID());
              if(firstAccess == FK_WriteOnly)
                policy = combinePolicy(policy, Cache::local);
              else {
                if(fieldInNextMS.getExtents()
                       .getVerticalLoopOrderAccesses(nextMS.getLoopOrder())
                       .CounterLoopOrder ||
                   fieldInNextMS.getExtents().isVerticalPointwise())
                  policy = combinePolicy(policy, Cache::flush);
                else
                  policy = combinePolicy(policy, Cache::epflush);
              }
            }
          } else {
            if(!instantiation->isTemporaryField(field.getAccessID()) &&
               field.getIntend() != Field::IK_Input)
              policy = combinePolicy(policy, Cache::flush);
          }

          auto interval = MS.computeEnclosingAccessInterval(field.getAccessID());

          // Set the cache
          DAWN_ASSERT(interval.is_initialized());
          Cache& cache = MS.setCache(Cache::K, policy, field.getAccessID(), *interval);

          if(context->getOptions().ReportPassSetCaches) {
            std::cout << "\nPASS: " << getName() << ": " << instantiation->getName() << ": MS"
                      << MSIndex << ": "
                      << instantiation->getOriginalNameFromAccessID(field.getAccessID()) << ":"
                      << cache.getCacheTypeAsString() << ":" << cache.getCacheIOPolicyAsString()
                      << std::endl;
          }
        }
      }
    }
  }

  return true;
}

} // namespace dawn
