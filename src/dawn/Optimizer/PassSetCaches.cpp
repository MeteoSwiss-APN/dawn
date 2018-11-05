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
#include "dawn/IIR/Cache.h"
#include "dawn/IIR/IntervalAlgorithms.h"
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Support/Unreachable.h"
#include <iostream>
#include <set>
#include <vector>

namespace dawn {

namespace {

enum class FirstAccessKind { FK_Read, FK_Write, FK_Mixed };

/// @brief properties of a cache candidate
struct CacheCandidate {
  iir::Cache::CacheIOPolicy policy_;
  boost::optional<iir::Cache::window> window_;
  iir::Interval interval_;
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
CacheCandidate combinePolicy(CacheCandidate const& MS1Policy, iir::Field::IntendKind fieldIntend,
                             CacheCandidate const& MS2Policy) {
  if(MS1Policy.policy_ == iir::Cache::local) {
    if(MS2Policy.policy_ == iir::Cache::fill)
      return CacheCandidate{iir::Cache::flush, boost::make_optional(iir::Cache::window{}),
                            MS1Policy.interval_};
    if(MS2Policy.policy_ == iir::Cache::bpfill) {
      DAWN_ASSERT(MS2Policy.window_.is_initialized());
      auto const& window = *(MS2Policy.window_);
      return CacheCandidate{iir::Cache::epflush,
                            boost::make_optional(iir::Cache::window{-window.m_p, -window.m_m}),
                            MS1Policy.interval_};
    }
    if(MS2Policy.policy_ == iir::Cache::local)
      return MS2Policy;
    dawn_unreachable("Not valid policy");
  }
  if(MS1Policy.policy_ == iir::Cache::fill || MS1Policy.policy_ == iir::Cache::bpfill) {
    if(MS2Policy.policy_ == iir::Cache::fill || MS2Policy.policy_ == iir::Cache::bpfill) {
      if(fieldIntend == iir::Field::IK_Input)
        return CacheCandidate{iir::Cache::fill, MS1Policy.window_, MS1Policy.interval_};
      else
        return CacheCandidate{iir::Cache::fill_and_flush, MS1Policy.window_, MS1Policy.interval_};
    }
    if(MS2Policy.policy_ == iir::Cache::local)
      return MS1Policy;
    dawn_unreachable("Not valid policy");
  }
  dawn_unreachable("invalid policy combination");
}

/// computes a cache candidate of a field for a multistage
CacheCandidate computeCacheCandidateForMS(iir::Field const& field, bool isTemporaryField,
                                          iir::MultiStage const& MS) {

  if(field.getIntend() == iir::Field::IK_Input) {
    boost::optional<iir::Interval> interval =
        MS.computeEnclosingAccessInterval(field.getAccessID(), true);
    // make sure the access interval has the same boundaries as from any interval of the mss
    interval->merge(field.getInterval());
    DAWN_ASSERT(interval.is_initialized());

    return CacheCandidate{iir::Cache::fill, boost::optional<iir::Cache::window>(), *interval};
  }
  if(field.getIntend() == iir::Field::IK_Output) {
    // we do not cache output only normal fields since there is not data reuse
    DAWN_ASSERT(isTemporaryField);
    return CacheCandidate{iir::Cache::local, boost::optional<iir::Cache::window>(),
                          field.getInterval()};
  }

  if(field.getIntend() == iir::Field::IK_InputOutput) {

    boost::optional<iir::Interval> interval =
        MS.computeEnclosingAccessInterval(field.getAccessID(), true);

    DAWN_ASSERT(interval.is_initialized());

    DAWN_ASSERT(interval->contains(field.getInterval()));

    iir::MultiInterval multiInterval = MS.computeReadAccessInterval(field.getAccessID());
    if(multiInterval.empty())
      return CacheCandidate{iir::Cache::local, boost::optional<iir::Cache::window>(),
                            field.getInterval()};

    if(multiInterval.numPartitions() > 1 ||
       multiInterval.getIntervals()[0].contains(field.getInterval()))
      return CacheCandidate{iir::Cache::fill, boost::optional<iir::Cache::window>(), *interval};

    iir::Interval const& readInterval = multiInterval.getIntervals()[0];

    iir::Cache::window window =
        computeWindowOffset(MS.getLoopOrder(), readInterval, field.getInterval());

    if(((MS.getLoopOrder() == iir::LoopOrderKind::LK_Forward) && window.m_m <= 0 &&
        window.m_p <= 0) ||
       ((MS.getLoopOrder() == iir::LoopOrderKind::LK_Backward) && window.m_m >= 0 &&
        window.m_p >= 0)) {
      return CacheCandidate{
          iir::Cache::bpfill,
          boost::make_optional(iir::Cache::window{
              ((MS.getLoopOrder() == iir::LoopOrderKind::LK_Forward) ? window.m_m : 0),
              ((MS.getLoopOrder() == iir::LoopOrderKind::LK_Forward) ? 0 : window.m_p)}),
          *interval};
    }

    return CacheCandidate{iir::Cache::fill, boost::optional<iir::Cache::window>(), *interval};
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
    for(const auto& multiStagePtr : stencil.getChildren()) {

      iir::MultiStage& MS = *(multiStagePtr);

      std::set<int> outputFields;

      auto isOutput = [](const iir::Field& field) {
        return field.getIntend() == iir::Field::IK_Output ||
               field.getIntend() == iir::Field::IK_InputOutput;
      };

      for(const auto& stage : MS.getChildren()) {
        for(const auto& fieldPair : stage->getFields()) {
          const iir::Field& field = fieldPair.second;

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
          if(field.getIntend() == iir::Field::IK_Input && outputFields.count(field.getAccessID()) &&
             !field.getExtents().isHorizontalPointwise()) {

            iir::Cache& cache = MS.setCache(iir::Cache::IJ, iir::Cache::local, field.getAccessID());
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

      std::set<int> mssProcessedFields;
      for(int MSIndex = 0; MSIndex < stencil.getChildren().size(); ++MSIndex) {
        iir::MultiStage& ms = *stencil.getMultiStageFromMultiStageIndex(MSIndex);

        const auto& fields = ms.getFields();
        for(const auto& AccessIDFieldPair : fields) {
          const iir::Field& field = AccessIDFieldPair.second;
          bool mssProcessedField = mssProcessedFields.count(field.getAccessID());
          if(!mssProcessedField)
            mssProcessedFields.emplace(field.getAccessID());

          // Field is already cached, skip
          if(ms.isCached(field.getAccessID()))
            continue;

          // Field has horizontal extents, can't be k-cached
          if(!field.getExtents().isHorizontalPointwise())
            continue;
          // we dont know how to cache fields with out of center writes
          if(field.getWriteExtents().is_initialized() && !field.getWriteExtents()->isPointwise())
            continue;

          if(!instantiation->isTemporaryField(field.getAccessID()) &&
             (field.getIntend() == iir::Field::IK_Output ||
              (field.getIntend() == iir::Field::IK_Input && field.getExtents().isPointwise())))
            continue;

          // Determine if we need to fill the cache by analyzing the current multi-stage
          CacheCandidate cacheCandidate = computeCacheCandidateForMS(
              field, instantiation->isTemporaryField(field.getAccessID()), ms);

          DAWN_ASSERT((cacheCandidate.policy_ != iir::Cache::fill &&
                       cacheCandidate.policy_ != iir::Cache::bpfill) ||
                      !instantiation->isTemporaryField(field.getAccessID() || mssProcessedField));

          if(!instantiation->isTemporaryField(field.getAccessID()) &&
             field.getIntend() != iir::Field::IK_Input) {

            cacheCandidate = combinePolicy(cacheCandidate, field.getIntend(),
                                           CacheCandidate{iir::Cache::CacheIOPolicy::fill,
                                                          boost::optional<iir::Cache::window>(),
                                                          /* FirstAccessKind::FK_Read, */
                                                          field.getInterval()});
          } else {

            for(int MSIndex2 = MSIndex + 1; MSIndex2 < stencil.getChildren().size(); MSIndex2++) {
              iir::MultiStage& nextMS = *stencil.getMultiStageFromMultiStageIndex(MSIndex2);
              const auto& nextMSfields = nextMS.getFields();

              if(!nextMSfields.count(field.getAccessID()))
                continue;

              const iir::Field& fieldInNextMS = nextMSfields.find(field.getAccessID())->second;

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

          iir::Interval interval = field.getInterval();
          auto interval_ = ms.computeEnclosingAccessInterval(field.getAccessID(), true);
          DAWN_ASSERT(interval_.is_initialized());
          auto enclosingAccessedInterval = *interval_;

          // Set the cache
          iir::Cache& cache =
              ms.setCache(iir::Cache::K, cacheCandidate.policy_, field.getAccessID(), interval,
                          enclosingAccessedInterval, cacheCandidate.window_);

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
