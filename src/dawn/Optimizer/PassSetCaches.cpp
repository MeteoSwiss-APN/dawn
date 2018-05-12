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
#include "dawn/Support/OptionalUtil.h"
#include "dawn/Optimizer/IntervalAlgorithms.h"
#include <iostream>
#include <set>
#include <vector>

namespace dawn {

namespace {

enum class FirstAccessKind { FK_Read, FK_Write, FK_Mixed };

FirstAccessKind toFirstAccess(Field::IntendKind access) {
  switch(access) {
  case Field::IK_Input:
    return FirstAccessKind::FK_Read;
    break;
  case Field::IK_InputOutput:
  case Field::IK_Output:
    return FirstAccessKind::FK_Write;
    break;
  default:
    dawn_unreachable("unknown access kind");
  }
}

/// @brief properties of a cache candidate
struct CacheCandidate {
  Cache::CacheIOPolicy policy_;
  boost::optional<Cache::window> window_;
  //  FirstAccessKind intend_;
  Interval interval_;
};

/// @brief combine the access kind of two accesses
boost::optional<FirstAccessKind> combineFirstAccess(boost::optional<FirstAccessKind> firstAccess,
                                                    boost::optional<FirstAccessKind> secondAccess) {
  if(!firstAccess.is_initialized()) {
    firstAccess = secondAccess;
  }
  if(!secondAccess.is_initialized())
    return firstAccess;

  switch(*firstAccess) {
  case FirstAccessKind::FK_Read:
    return (*secondAccess == FirstAccessKind::FK_Read)
               ? firstAccess
               : boost::make_optional(FirstAccessKind::FK_Mixed);
    break;
  case FirstAccessKind::FK_Write:
    return (*secondAccess == FirstAccessKind::FK_Write)
               ? firstAccess
               : boost::make_optional(FirstAccessKind::FK_Mixed);
    break;
  case FirstAccessKind::FK_Mixed:
    return firstAccess;
    break;
  default:
    dawn_unreachable("unknown access kind");
  }
}

bool iterationPastInterval(Interval const& iterInterval, Interval const& baseInterval,
                           LoopOrderKind loopOrder) {
  if(loopOrder == LoopOrderKind::LK_Forward) {
    return iterInterval.lowerBound() > baseInterval.upperBound();
  }
  if(loopOrder == LoopOrderKind::LK_Backward) {
    return iterInterval.upperBound() < baseInterval.lowerBound();
  }
  dawn_unreachable("loop order not supported");
}

/// @brief Do we compute the first levels or do we need to access main memory?
///
/// If the first write access of the field has a read access to the same field as well, it is not
/// computed and we need to fill/flush the caches.
// TODO need to unittest this
static boost::optional<FirstAccessKind> getFirstAccessKind(const MultiStage& MS, int AccessID) {

  // Get the stage of the first access
  for(const auto& stagePtr : MS.getStages()) {
    Stage& stage = *stagePtr;
    if(std::find_if(stage.getFields().begin(), stage.getFields().end(), [&](const Field& field) {
         return field.getAccessID() == AccessID;
       }) != stage.getFields().end()) {

      auto getFirstAccessKindFromDoMethod = [&](
          const DoMethod* doMethod) -> boost::optional<FirstAccessKind> {

        for(const auto& statementAccesssPair : doMethod->getStatementAccessesPairs()) {
          const Accesses& accesses = *statementAccesssPair->getAccesses();
          // indepdently of whether the statement has also a write access, if there is a read
          // access, it should happen in the RHS so first
          if(accesses.hasReadAccess(AccessID)) {

            //            if(accesses.getReadAccess(AccessID)
            //                   .getVerticalLoopOrderAccesses(MS.getLoopOrder())
            //                   .CounterLoopOrder ||
            //               (!firstAccess.is_initialized() || (*firstAccess ==
            //               FirstAccessKind::FK_Read)))
            return boost::make_optional(FirstAccessKind::FK_Read);
          }
          if(accesses.hasWriteAccess(AccessID))
            return boost::make_optional(FirstAccessKind::FK_Write);
          // TODO provide solution for block statments
          // that can be compile time resolved
        }
        return boost::optional<FirstAccessKind>();
      };

      boost::optional<FirstAccessKind> firstAccess;

      // We need to check all Do-Methods
      if(MS.getLoopOrder() == LoopOrderKind::LK_Parallel) {

        for(const auto& doMethodPtr : stage.getDoMethods()) {
          auto thisFirstAccess = getFirstAccessKindFromDoMethod(doMethodPtr.get());

          firstAccess = combineFirstAccess(firstAccess, thisFirstAccess);
        }
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
          auto thisFirstAccess = getFirstAccessKindFromDoMethod(D);
          if(thisFirstAccess.is_initialized())
            std::cout << "DD " << static_cast<typename std::underlying_type<FirstAccessKind>::type>(
                                      *thisFirstAccess)
                      << std::endl;
          else
            std::cout << " DD NO" << std::endl;

          if(firstAccess.is_initialized())
            std::cout << "EE " << static_cast<typename std::underlying_type<FirstAccessKind>::type>(
                                      *firstAccess)
                      << std::endl;
          else
            std::cout << " EE NO" << std::endl;

          firstAccess = combineFirstAccess(firstAccess, thisFirstAccess);
          if(firstAccess.is_initialized())
            std::cout << "FF " << static_cast<typename std::underlying_type<FirstAccessKind>::type>(
                                      *firstAccess)
                      << std::endl;
          else
            std::cout << " FF NO" << std::endl;
        }
      }
      return firstAccess;
    }
  }
  dawn_unreachable("MultiStage does not contain the given field");
}

/// @brief Combine the policies from the first MultiStage (`MS1Policy`) and the
/// immediately
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
  // TODO properly compute the window
  if(MS1Policy.policy_ == Cache::local) {
    if(MS2Policy.policy_ == Cache::fill)
      return CacheCandidate{Cache::flush,
                            boost::make_optional(Cache::window{}), /*MS1Policy.intend_, */
                            MS1Policy.interval_};                  // flush
    if(MS2Policy.policy_ == Cache::bpfill) {
      DAWN_ASSERT(MS2Policy.window_.is_initialized());
      auto const& window = *(MS2Policy.window_);
      return CacheCandidate{Cache::epflush,
                            boost::make_optional(Cache::window{-window.m_p, -window.m_m}),
                            /*MS1Policy.intend_,*/
                            MS1Policy.interval_}; // epflush
    }
    if(MS2Policy.policy_ == Cache::local)
      return MS2Policy; // local
    dawn_unreachable("Not valid policy");
  }
  if(MS1Policy.policy_ == Cache::fill || MS1Policy.policy_ == Cache::bpfill) {
    if(MS2Policy.policy_ == Cache::fill || MS2Policy.policy_ == Cache::bpfill) {
      if(fieldIntend == Field::IK_Input)
        return CacheCandidate{Cache::fill, MS1Policy.window_,
                              /*MS1Policy.intend_,*/
                              MS1Policy.interval_};
      else
        return CacheCandidate{Cache::fill_and_flush, MS1Policy.window_,
                              /*MS1Policy.intend_,*/
                              MS1Policy.interval_};
    }
    if(MS2Policy.policy_ == Cache::local)
      return MS1Policy; // fill, pbfill
    dawn_unreachable("Not valid policy");
  }
  dawn_unreachable("invalid policy combination");
}

/// computes the cache window required by bpfill and epflush based on the accessed interval of a
/// field and the interval of the iteration
Cache::window computeCacheWindow(LoopOrderKind loopOrder, Interval const& accessedInterval,
                                 Interval const& iterationInterval) {
  if(loopOrder == LoopOrderKind::LK_Forward) {
    DAWN_ASSERT(accessedInterval.upperBound() == iterationInterval.upperBound());
    DAWN_ASSERT(accessedInterval.lowerBound() <= iterationInterval.lowerBound());

    return Cache::window{accessedInterval.lowerBound() - iterationInterval.lowerBound(), 0};

  } else if(loopOrder == LoopOrderKind::LK_Backward) {
    DAWN_ASSERT(accessedInterval.lowerBound() == iterationInterval.lowerBound());
    DAWN_ASSERT(accessedInterval.upperBound() >= iterationInterval.upperBound());

    return Cache::window{0, accessedInterval.upperBound() - iterationInterval.upperBound()};
  } else
    dawn_unreachable_internal();
  return Cache::window{};
}

/// computes a cache candidate of a field for a multistage
CacheCandidate computeCacheCandidateForMS(Field const& field, bool isTemporaryField,
                                          MultiStage const& MS) {

  if(field.getIntend() == Field::IK_Input) {
    boost::optional<Interval> interval = MS.computeEnclosingAccessInterval(field.getAccessID());
    // make sure the access interval has the same boundaries as from any interval of the mss
    interval->merge(field.getInterval());
    DAWN_ASSERT(interval.is_initialized());

    return CacheCandidate{Cache::fill, boost::optional<Cache::window>(),
                          /*toFirstAccess(field.getIntend()),
         */ *interval};
  }
  if(field.getIntend() == Field::IK_Output) {
    // we do not cache output only normal fields since there is not data reuse
    DAWN_ASSERT(isTemporaryField);
    return CacheCandidate{Cache::local, boost::optional<Cache::window>(),
                          /* toFirstAccess(field.getIntend()), */
                          field.getInterval()};
  }

  if(field.getIntend() == Field::IK_InputOutput) {

    boost::optional<Interval> interval = MS.computeEnclosingAccessInterval(field.getAccessID());
    // make sure the access interval has the same boundaries as from any interval of the mss
    interval->merge(field.getInterval());

    DAWN_ASSERT(interval.is_initialized());

    std::cout << "IIII " << *interval << " " << field.getInterval() << std::endl;
    DAWN_ASSERT(interval->contains(field.getInterval()));

    MultiInterval multiInterval = MS.computeReadAccessInterval(field.getAccessID());
    std::cout << "OVERAL " << multiInterval << std::endl;
    if(multiInterval.empty())
      return CacheCandidate{Cache::local, boost::optional<Cache::window>(), /* *firstAccess, */
                            field.getInterval()};

    //      if(((MS.getLoopOrder() == LoopOrderKind::LK_Forward) &&
    //          interval->lowerBound() < field.getInterval().lowerBound()) ||
    //         ((MS.getLoopOrder() == LoopOrderKind::LK_Backward) &&
    //          interval->upperBound() > field.getInterval().upperBound())) {

    if(multiInterval.numPartitions() > 1 ||
       multiInterval.getIntervals()[0].contains(field.getInterval()))
      return CacheCandidate{Cache::fill, boost::optional<Cache::window>(), *interval};

    Interval const& readInterval = multiInterval.getIntervals()[0];

    std::cout << "PrePPP " << multiInterval << " " << MS.getLoopOrder() << " "
              << multiInterval.numPartitions() << " " << field.getInterval() << std::endl;

    Cache::window window =
        computeWindowOffset(MS.getLoopOrder(), readInterval, field.getInterval());

    if(((MS.getLoopOrder() == LoopOrderKind::LK_Forward) && window.m_m <= 0 && window.m_p <= 0) ||
       ((MS.getLoopOrder() == LoopOrderKind::LK_Backward) && window.m_m >= 0 && window.m_p >= 0)) {
      std::cout << "COMPPPP " << window.m_m << " " << window.m_p << std::endl;
      return CacheCandidate{
          Cache::bpfill, boost::make_optional(Cache::window{
                             ((MS.getLoopOrder() == LoopOrderKind::LK_Forward) ? window.m_m : 0),
                             ((MS.getLoopOrder() == LoopOrderKind::LK_Forward) ? 0 : window.m_p)}),
          *interval};
    }

    return CacheCandidate{Cache::fill, boost::optional<Cache::window>(), *interval};

    std::cout << "PPP " << multiInterval << " " << MS.getLoopOrder() << " "
              << multiInterval.numPartitions() << std::endl;
    //    // Do we compute the first levels or do we need to access main memory (i.e fill the
    //    // fist accesses)?
    //    // TODO test getFirstAccessKind
    //    boost::optional<FirstAccessKind> firstAccess = getFirstAccessKind(MS,
    //    field.getAccessID());
    //    DAWN_ASSERT(firstAccess.is_initialized());
    //    if(*firstAccess == FirstAccessKind::FK_Write) {
    //      // Example (kcache candidate a):
    //      //   k=k_end:k_start
    //      //     a = in
    //      //     out = a[k-1]
    //      boost::optional<Interval> interval =
    //      MS.computeEnclosingAccessInterval(field.getAccessID());
    //      DAWN_ASSERT(interval.is_initialized());

    //      DAWN_ASSERT(interval->contains(field.getInterval()));

    //      if(((MS.getLoopOrder() == LoopOrderKind::LK_Forward) &&
    //          interval->upperBound() > field.getInterval().upperBound()) ||
    //         ((MS.getLoopOrder() == LoopOrderKind::LK_Backward) &&
    //          interval->lowerBound() < field.getInterval().lowerBound()))
    //        return CacheCandidate{Cache::fill, boost::optional<Cache::window>(), *firstAccess,
    //                              *interval};
    //      if(((MS.getLoopOrder() == LoopOrderKind::LK_Forward) &&
    //          interval->lowerBound() < field.getInterval().lowerBound()) ||
    //         ((MS.getLoopOrder() == LoopOrderKind::LK_Backward) &&
    //          interval->upperBound() > field.getInterval().upperBound())) {
    //        std::cout << "IIOIOI " << interval->upperBound() << " " <<
    //        field.getInterval().upperBound()
    //                  << std::endl;
    //        return CacheCandidate{Cache::bpfill,
    //                              boost::make_optional(Cache::window{
    //                                  interval->lowerBound() - field.getInterval().lowerBound(),
    //                                  interval->upperBound() - field.getInterval().upperBound()}),
    //                              *firstAccess, *interval};
    //      }

    //      return CacheCandidate{Cache::local, boost::optional<Cache::window>(), *firstAccess,
    //                            *interval};
    //    }
    //    // Do we have a read in counter loop order or pointwise access?
    //    // TODO test getVerticalLoopOrderAccesses
    //    if(field.getExtents().getVerticalLoopOrderAccesses(MS.getLoopOrder()).CounterLoopOrder ||
    //       field.getExtents().isVerticalPointwise()) {

    //      // Example (kcache candidate a):
    //      //   1. a += a
    //      //   2. k=k_end:k_start
    //      //         a += a[k-1]

    //      // TODO Should not require a bpfill
    //      // If the field is a temporary, we only set the policy fo fill. A flush will be
    //      // granted if following MSS also read the temporary field
    //      boost::optional<Interval> interval =
    //      MS.computeEnclosingAccessInterval(field.getAccessID());
    //      DAWN_ASSERT(interval.is_initialized());
    //      return CacheCandidate{Cache::fill, boost::optional<Cache::window>(),
    //                            toFirstAccess(field.getIntend()), *interval};
    //    }
    //    // Example (kcache candidate a):
    //    //     k=k_end-1:k_start
    //    //        a += a[k+1]
    //    // TODO should be a temporary, otherwise will require also a flush
    //    auto accessedInterval = MS.computeEnclosingAccessInterval(field.getAccessID());
    //    DAWN_ASSERT(accessedInterval.is_initialized());
    //    auto cacheWindow = boost::make_optional(
    //        computeCacheWindow(MS.getLoopOrder(), *accessedInterval, field.getInterval()));
    //    boost::optional<Interval> interval =
    //    MS.computeEnclosingAccessInterval(field.getAccessID());
    //    DAWN_ASSERT(interval.is_initialized());
    //    std::cout << "INBPFIL " << std::endl;
    //    return CacheCandidate{Cache::bpfill, cacheWindow, toFirstAccess(field.getIntend()),
    //    *interval};
  }
  dawn_unreachable("Policy of Field not found");
}

} // anonymous namespace

PassSetCaches::PassSetCaches() : Pass("PassSetCaches") {}

bool PassSetCaches::isAccessIDReadAfter(
    const int accessID, std::list<std::shared_ptr<Stage>>::const_iterator stage,
    std::list<std::shared_ptr<MultiStage>>::const_iterator multiStage,
    const Stencil& stencil) const {

  DAWN_ASSERT(multiStage != stencil.getMultiStages().end());
  DAWN_ASSERT(stage != (*multiStage)->getStages().end());

  // TODO clean this algo
  for(std::list<std::shared_ptr<Stage>>::const_iterator stageIt = stage;
      stageIt != (*multiStage)->getStages().end(); ++stageIt) {

    for(const Field& field : (*stageIt)->getFields()) {
      if(field.getAccessID() != accessID)
        continue;
      if(field.getIntend() == Field::IK_Input) {
        return true;
      }
      if(field.getIntend() == Field::IK_Output) {
        return false;
      }
      if(field.getIntend() == Field::IK_InputOutput) {
        boost::optional<FirstAccessKind> firstAccess =
            getFirstAccessKind(*(*multiStage), field.getAccessID());
        DAWN_ASSERT(firstAccess.is_initialized());
        if(*firstAccess == FirstAccessKind::FK_Write) {
          return false;
        }

        return true;
      }
      dawn_unreachable("Unknown intend");
    }
  }

  for(std::list<std::shared_ptr<MultiStage>>::const_iterator multiStageIt = std::next(multiStage);
      multiStageIt != stencil.getMultiStages().end(); ++multiStageIt) {

    for(std::list<std::shared_ptr<Stage>>::const_iterator stageIt =
            (*multiStageIt)->getStages().begin();
        stage != (*multiStageIt)->getStages().end(); ++stageIt) {

      for(const Field& field : (*stageIt)->getFields()) {
        if(field.getAccessID() != accessID)
          continue;
        if(field.getIntend() == Field::IK_Input) {
          return true;
        }
        if(field.getIntend() == Field::IK_Output) {
          return false;
        }
        if(field.getIntend() == Field::IK_InputOutput) {
          boost::optional<FirstAccessKind> firstAccess =
              getFirstAccessKind(*(*multiStageIt), field.getAccessID());
          DAWN_ASSERT(firstAccess.is_initialized());
          if(*firstAccess == FirstAccessKind::FK_Write) {
            return false;
          }

          return true;
        }
        dawn_unreachable("Unknown intend");
      }
    }
  }

  return false;
}

bool PassSetCaches::run(const std::shared_ptr<StencilInstantiation>& instantiation) {
  OptimizerContext* context = instantiation->getOptimizerContext();

  for(const auto& stencilPtr : instantiation->getStencils()) {
    const Stencil& stencil = *stencilPtr;

    // Set IJ-Caches
    int msIdx = 0;
    for(auto multiStageIt = stencil.getMultiStages().begin();
        multiStageIt != stencil.getMultiStages().end(); ++multiStageIt) {

      MultiStage& MS = *(*multiStageIt);

      std::set<int> outputFields;

      auto isOutput = [](const Field& field) {
        return field.getIntend() == Field::IK_Output || field.getIntend() == Field::IK_InputOutput;
      };

      for(auto stageIt = MS.getStages().begin(); stageIt != MS.getStages().end(); stageIt++) {
        for(const Field& field : (*stageIt)->getFields()) {

          // Field is already cached, skip
          if(MS.isCached(field.getAccessID())) {
            std::cout << "already cache " << instantiation->getNameFromAccessID(field.getAccessID())
                      << std::endl;
            continue;
          }

          // Field has vertical extents, can't be ij-cached
          if(!field.getExtents().isVerticalPointwise()) {
            std::cout << "pointwise " << instantiation->getNameFromAccessID(field.getAccessID())
                      << std::endl;

            continue;
          }

          // Currently we only cache temporaries!
          if(!instantiation->isTemporaryField(field.getAccessID())) {
            std::cout << "Only temporary "
                      << instantiation->getNameFromAccessID(field.getAccessID()) << std::endl;

            continue;
          }

          //          if(isAccessIDReadAfter(field.getAccessID(), stageIt, multiStageIt, stencil)) {
          //            std::cout << "read after " <<
          //            instantiation->getNameFromAccessID(field.getAccessID())
          //                      << std::endl;

          //            continue;
          //          }
          // Cache the field
          if(field.getIntend() == Field::IK_Input && outputFields.count(field.getAccessID()) &&
             !field.getExtents().isHorizontalPointwise()) {

            std::cout << "caching " << instantiation->getNameFromAccessID(field.getAccessID())
                      << std::endl;

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
                     std::back_inserter(fields),
                     [](const std::shared_ptr<MultiStage>& MSPtr) { return MSPtr->getFields(); });

      boost::optional<Cache::window> cacheWindow;
      int numMS = fields.size();
      std::set<int> mssProcessedFields;
      for(int MSIndex = 0; MSIndex < numMS; ++MSIndex) {
        for(const auto& AccessIDFieldPair : fields[MSIndex]) {
          MultiStage& MS = *stencil.getMultiStageFromMultiStageIndex(MSIndex);
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
          //          std::cout << " JJ " << MSIndex << " "
          //                    << instantiation->getNameFromAccessID(field.getAccessID()) << " "
          //                    << field.getWriteExtents() << std::endl;
          // we dont know how to cache fields with out of center writes
          if(field.getWriteExtents().is_initialized() && !field.getWriteExtents()->isPointwise())
            continue;

          //          std::cout << " JJ1 " <<
          //          instantiation->getNameFromAccessID(field.getAccessID())
          //                    << field.getExtents().isPointwise() << std::endl;

          if(!instantiation->isTemporaryField(field.getAccessID()) &&
             (field.getIntend() == Field::IK_Output ||
              (field.getIntend() == Field::IK_Input && field.getExtents().isPointwise())))
            continue;

          //          std::cout << " JJ2 " << field.getExtents().isPointwise() << " "
          //                    << instantiation->getNameFromAccessID(field.getAccessID()) <<
          //                    std::endl;

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

              const MultiStage& nextMS = *stencil.getMultiStageFromMultiStageIndex(MSIndex2);
              const Field& fieldInNextMS = fields[MSIndex2].find(field.getAccessID())->second;

              CacheCandidate policyMS2 = computeCacheCandidateForMS(
                  fieldInNextMS, instantiation->isTemporaryField(fieldInNextMS.getAccessID()),
                  nextMS);
              // if the interval of the two cache candidate do not overlap, there is no data
              // dependency, therefore we skip it
              if(!cacheCandidate.interval_.overlaps(policyMS2.interval_))
                continue;

              //              if(policyMS2.intend_ == FirstAccessKind::FK_Mixed) {
              //                cacheCandidate.intend_ = FirstAccessKind::FK_Mixed;
              //                break;
              //              }
              cacheCandidate = combinePolicy(cacheCandidate, field.getIntend(), policyMS2);
              break;
            }

            //            if(cacheCandidate.intend_ == FirstAccessKind::FK_Mixed)
            //              continue;
          }

          Interval interval = field.getInterval();
          if(cacheCandidate.policy_ == Cache::CacheIOPolicy::fill) {
            auto interval_ = MS.computeEnclosingAccessInterval(field.getAccessID());
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
