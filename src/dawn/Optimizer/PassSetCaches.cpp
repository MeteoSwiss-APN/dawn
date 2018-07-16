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
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Support/Unreachable.h"
#include "dawn/Optimizer/IntervalAlgorithms.h"
#include <iostream>
#include <set>
#include <vector>

namespace dawn {

namespace {

enum class FirstAccessKind { FK_Read, FK_Write, FK_Mixed };

/// @brief properties of a cache candidate
struct CacheCandidate {
  Cache::CacheIOPolicy policy_;
  boost::optional<Cache::window> window_;
  Interval interval_;
};

/// @brief Combine the policies from the first MultiStage (`MS1Policy`) and the immediately
/// following (`MS2Policy`)
///
/// MS1Policy has to be one of: {local, fill, bpfill}
/// MS2Policy has to be one of: {local, flush, epflush}
///
///                                       MS2
///
///                    ||    local    ||     fill      ||   bpfill     ||
///       =========++=================++===============++==============++
///        local       ||    local    |      flush     |    epflush     |
///       =============++-------------+----------------+----------------+
///  MS1   fill (ro)   ||    fill     |      fill      |      fill      |
///       =============++-------------+----------------+----------------+
///        fill (w)    ||    fill     + fill_and_flush + fill_and_flush +
///       =============++-------------+----------------+----------------+
///        bpfill (ro) ||   bpfill    |       fill     |      fill      |
///       =============++-------------+----------------+----------------+
///        bpfill (w ) ||   bpfill    | fill_and_flush | fill_and_flush |
///       =============++-------------+----------------+----------------+
///
CacheCandidate combinePolicy(CacheCandidate const& MS1Policy, Field::IntendKind fieldIntend,
                             CacheCandidate const& MS2Policy) {
  if(MS1Policy.policy_ == Cache::local) {
    if(MS2Policy.policy_ == Cache::fill)
      return CacheCandidate{Cache::flush, boost::make_optional(Cache::window{}),
                            MS1Policy.interval_};
    if(MS2Policy.policy_ == Cache::bpfill) {
      DAWN_ASSERT(MS2Policy.window_.is_initialized());
      auto const& window = *(MS2Policy.window_);
      return CacheCandidate{Cache::epflush,
                            boost::make_optional(Cache::window{-window.m_p, -window.m_m}),
                            MS1Policy.interval_};
    }
    if(MS2Policy.policy_ == Cache::local)
      return MS2Policy;
    dawn_unreachable("Not valid policy");
  }
  if(MS1Policy.policy_ == Cache::fill || MS1Policy.policy_ == Cache::bpfill) {
    if(MS2Policy.policy_ == Cache::fill || MS2Policy.policy_ == Cache::bpfill) {
      if(fieldIntend == Field::IK_Input)
        return CacheCandidate{Cache::fill, MS1Policy.window_, MS1Policy.interval_};
      else
        return CacheCandidate{Cache::fill_and_flush, MS1Policy.window_, MS1Policy.interval_};
    }
    if(MS2Policy.policy_ == Cache::local)
      return MS1Policy;
    dawn_unreachable("Not valid policy");
  }
  dawn_unreachable("invalid policy combination");
}

/// computes a cache candidate of a field for a multistage
CacheCandidate computeCacheCandidateForMS(Field const& field, bool isTemporaryField,
                                          iir::MultiStage const& MS) {

  if(field.getIntend() == Field::IK_Input) {
    boost::optional<Interval> interval =
        MS.computeEnclosingAccessInterval(field.getAccessID(), true);
    // make sure the access interval has the same boundaries as from any interval of the mss
    interval->merge(field.getInterval());
    DAWN_ASSERT(interval.is_initialized());

    return CacheCandidate{Cache::fill, boost::optional<Cache::window>(), *interval};
  }
  if(field.getIntend() == Field::IK_Output) {
    // we do not cache output only normal fields since there is not data reuse
    DAWN_ASSERT(isTemporaryField);
    return CacheCandidate{Cache::local, boost::optional<Cache::window>(), field.getInterval()};
  }

  if(field.getIntend() == Field::IK_InputOutput) {

    boost::optional<Interval> interval =
        MS.computeEnclosingAccessInterval(field.getAccessID(), true);

    DAWN_ASSERT(interval.is_initialized());

    DAWN_ASSERT(interval->contains(field.getInterval()));

    MultiInterval multiInterval = MS.computeReadAccessInterval(field.getAccessID());
    if(multiInterval.empty())
      return CacheCandidate{Cache::local, boost::optional<Cache::window>(), field.getInterval()};

    if(multiInterval.numPartitions() > 1 ||
       multiInterval.getIntervals()[0].contains(field.getInterval()))
      return CacheCandidate{Cache::fill, boost::optional<Cache::window>(), *interval};

    Interval const& readInterval = multiInterval.getIntervals()[0];

    Cache::window window =
        computeWindowOffset(MS.getLoopOrder(), readInterval, field.getInterval());

    if(((MS.getLoopOrder() == LoopOrderKind::LK_Forward) && window.m_m <= 0 && window.m_p <= 0) ||
       ((MS.getLoopOrder() == LoopOrderKind::LK_Backward) && window.m_m >= 0 && window.m_p >= 0)) {
      return CacheCandidate{
          Cache::bpfill, boost::make_optional(Cache::window{
                             ((MS.getLoopOrder() == LoopOrderKind::LK_Forward) ? window.m_m : 0),
                             ((MS.getLoopOrder() == LoopOrderKind::LK_Forward) ? 0 : window.m_p)}),
          *interval};
    }

    return CacheCandidate{Cache::fill, boost::optional<Cache::window>(), *interval};
  }
  dawn_unreachable("Policy of Field not found");
}

} // anonymous namespace

PassSetCaches::PassSetCaches() : Pass("PassSetCaches") {}

bool PassSetCaches::run(const std::shared_ptr<iir::StencilInstantiation>& instantiation) {
  OptimizerContext* context = instantiation->getOptimizerContext();

  for(const auto& stencilPtr : instantiation->getStencils()) {
    const iir::Stencil& stencil = *stencilPtr;

    // Set IJ-Caches
    int msIdx = 0;
    for(auto multiStageIt = stencil.getMultiStages().begin();
        multiStageIt != stencil.getMultiStages().end(); ++multiStageIt) {

      iir::MultiStage& MS = *(*multiStageIt);

      std::set<int> outputFields;

      auto isOutput = [](const Field& field) {
        return field.getIntend() == Field::IK_Output || field.getIntend() == Field::IK_InputOutput;
      };

      for(auto stageIt = MS.getStages().begin(); stageIt != MS.getStages().end(); stageIt++) {
        for(const Field& field : (*stageIt)->getFields()) {

          // Field is already cached, skip
          if(MS.isCached(field.getAccessID())) {
            continue;
          }

          // Field has vertical extents, can't be ij-cached
          if(!field.getExtents().isVerticalPointwise()) {
            continue;
          }

          // Currently we only cache temporaries!
          if(!instantiation->isTemporaryField(field.getAccessID())) {
            continue;
          }

          // Cache the field
          if(field.getIntend() == Field::IK_Input && outputFields.count(field.getAccessID()) &&
             !field.getExtents().isHorizontalPointwise()) {

            Cache& cache = MS.setCache(Cache::IJ, Cache::local, field.getAccessID());
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
                     std::back_inserter(fields), [](const std::shared_ptr<iir::MultiStage>& MSPtr) {
                       return MSPtr->getFields();
                     });

      int numMS = fields.size();
      std::set<int> mssProcessedFields;
      for(int MSIndex = 0; MSIndex < numMS; ++MSIndex) {
        for(const auto& AccessIDFieldPair : fields[MSIndex]) {
          iir::MultiStage& MS = *stencil.getMultiStageFromMultiStageIndex(MSIndex);
          const Field& field = AccessIDFieldPair.second;
          bool mssProcessedField = mssProcessedFields.count(field.getAccessID());
          if(!mssProcessedField)
            mssProcessedFields.emplace(field.getAccessID());

          // Field is already cached, skip
          if(MS.isCached(field.getAccessID()))
            continue;

          // Field has horizontal extents, can't be k-cached
          if(!field.getExtents().isHorizontalPointwise())
            continue;
          // we dont know how to cache fields with out of center writes
          if(field.getWriteExtents().is_initialized() && !field.getWriteExtents()->isPointwise())
            continue;

          if(!instantiation->isTemporaryField(field.getAccessID()) &&
             (field.getIntend() == Field::IK_Output ||
              (field.getIntend() == Field::IK_Input && field.getExtents().isPointwise())))
            continue;

          // Determine if we need to fill the cache by analyzing the current multi-stage
          CacheCandidate cacheCandidate = computeCacheCandidateForMS(
              field, instantiation->isTemporaryField(field.getAccessID()), MS);

          //          if(cacheCandidate.intend_ == FirstAccessKind::FK_Mixed)
          //            continue;

          DAWN_ASSERT(
              (cacheCandidate.policy_ != Cache::fill && cacheCandidate.policy_ != Cache::bpfill) ||
              !instantiation->isTemporaryField(field.getAccessID() || mssProcessedField));

          if(!instantiation->isTemporaryField(field.getAccessID()) &&
             field.getIntend() != Field::IK_Input) {

            cacheCandidate = combinePolicy(cacheCandidate, field.getIntend(),
                                           CacheCandidate{Cache::CacheIOPolicy::fill,
                                                          boost::optional<Cache::window>(),
                                                          /* FirstAccessKind::FK_Read, */
                                                          field.getInterval()});
          } else {

            for(int MSIndex2 = MSIndex + 1; MSIndex2 < numMS; MSIndex2++) {
              if(!fields[MSIndex2].count(field.getAccessID()))
                continue;

              const iir::MultiStage& nextMS = *stencil.getMultiStageFromMultiStageIndex(MSIndex2);
              const Field& fieldInNextMS = fields[MSIndex2].find(field.getAccessID())->second;

              CacheCandidate policyMS2 = computeCacheCandidateForMS(
                  fieldInNextMS, instantiation->isTemporaryField(fieldInNextMS.getAccessID()),
                  nextMS);
              // if the interval of the two cache candidate do not overlap, there is no data
              // dependency, therefore we skip it
              if(!cacheCandidate.interval_.overlaps(policyMS2.interval_))
                continue;

              cacheCandidate = combinePolicy(cacheCandidate, field.getIntend(), policyMS2);
              break;
            }
          }

          Interval interval = field.getInterval();
          if(cacheCandidate.policy_ == Cache::CacheIOPolicy::fill) {
            auto interval_ = MS.computeEnclosingAccessInterval(field.getAccessID(), true);
            DAWN_ASSERT(interval_.is_initialized());
            interval = *interval_;
          }

          // Set the cache
          Cache& cache = MS.setCache(Cache::K, cacheCandidate.policy_, field.getAccessID(),
                                     interval, cacheCandidate.window_);

          if(context->getOptions().ReportPassSetCaches) {
            std::cout << "\nPASS: " << getName() << ": " << instantiation->getName() << ": MS"
                      << MSIndex << ": "
                      << instantiation->getOriginalNameFromAccessID(field.getAccessID()) << ":"
                      << cache.getCacheTypeAsString() << ":" << cache.getCacheIOPolicyAsString()
                      << (cache.getWindow().is_initialized()
                              ? (std::string(":") + cache.getWindow()->toString())
                              : "")
                      << std::endl;
          }
        }
      }
    }
  }

  return true;
}

} // namespace dawn
