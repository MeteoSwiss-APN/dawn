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
  FirstAccessKind intend_;
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
          if(accesses.hasReadAccess(AccessID))
            return boost::make_optional(FirstAccessKind::FK_Read);
          if(accesses.hasWriteAccess(AccessID))
            return boost::make_optional(FirstAccessKind::FK_Write);
          // TODO provide solution for block statments
          // that can be compile time resolved
        }
        return boost::optional<FirstAccessKind>();
      };
      boost::optional<FirstAccessKind> firstAccess{};

      // We need to check all Do-Methods
      if(MS.getLoopOrder() == LoopOrderKind::LK_Parallel) {

        for(const auto& doMethodPtr : stage.getDoMethods()) {
          std::cout << "UU " << std::endl;
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
          firstAccess = combineFirstAccess(firstAccess, thisFirstAccess);
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
  std::cout << "COMBINE POLICIES " << MS1Policy.policy_ << " " << MS2Policy.policy_ << std::endl;
  // TODO properly compute the window
  if(MS1Policy.policy_ == Cache::local) {
    if(MS2Policy.policy_ == Cache::fill)
      return CacheCandidate{Cache::flush, boost::make_optional(Cache::window{}), MS1Policy.intend_,
                            MS1Policy.interval_}; // flush
    if(MS2Policy.policy_ == Cache::bpfill) {
      DAWN_ASSERT(MS2Policy.window_.is_initialized());
      auto const& window = *(MS2Policy.window_);
      return CacheCandidate{Cache::epflush,
                            boost::make_optional(Cache::window{-window.m_p, -window.m_m}),
                            MS1Policy.intend_, MS1Policy.interval_}; // epflush
    }
    if(MS2Policy.policy_ == Cache::local)
      return MS2Policy; // local
    dawn_unreachable("Not valid policy");
  }
  if(MS1Policy.policy_ == Cache::fill || MS1Policy.policy_ == Cache::bpfill) {
    if(MS2Policy.policy_ == Cache::fill || MS2Policy.policy_ == Cache::bpfill) {
      if(fieldIntend == Field::IK_Input)
        return CacheCandidate{Cache::fill, MS1Policy.window_, MS1Policy.intend_,
                              MS1Policy.interval_};
      else
        return CacheCandidate{Cache::fill_and_flush, MS1Policy.window_, MS1Policy.intend_,
                              MS1Policy.interval_};
    }
    if(MS2Policy.policy_ == Cache::local)
      return MS1Policy; // fill, pbfill
    std::cout << "PO  " << MS1Policy.policy_ << " " << MS2Policy.policy_ << std::endl;
    dawn_unreachable("Not valid policy");
  }
  dawn_unreachable("invalid policy combination");
}

} // anonymous namespace

PassSetCaches::PassSetCaches() : Pass("PassSetCaches") {}

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

CacheCandidate computePolicyMS1(Field const& field, bool isTemporaryField, MultiStage const& MS) {

  if(field.getIntend() == Field::IK_Input) {
    std::cout << "SET FILL "
              << field.getAccessID() /* instantiation->getNameFromAccessID(field.getAccessID()) */
              << std::endl;
    boost::optional<Interval> interval = MS.computeEnclosingAccessInterval(field.getAccessID());
    DAWN_ASSERT(interval.is_initialized());

    return CacheCandidate{Cache::fill, boost::optional<Cache::window>(),
                          toFirstAccess(field.getIntend()), *interval};
  }
  if(field.getIntend() == Field::IK_Output) {
    std::cout << "SET LOCAL "
              << field.getAccessID() /*instantiation->getNameFromAccessID(field.getAccessID())*/
              << std::endl;
    // we do not cache output only normal fields since there is not data reuse
    DAWN_ASSERT(isTemporaryField);
    return CacheCandidate{Cache::local, boost::optional<Cache::window>(),
                          toFirstAccess(field.getIntend()), field.getInterval()};
  }

  if(field.getIntend() == Field::IK_InputOutput) {
    // Do we compute the first levels or do we need to access main memory (i.e fill the
    // fist accesses)?
    // TODO test getFirstAccessKind
    boost::optional<FirstAccessKind> firstAccess = getFirstAccessKind(MS, field.getAccessID());
    DAWN_ASSERT(firstAccess.is_initialized());
    std::cout << "ACH " << static_cast<int>(*firstAccess) << std::endl;
    if(*firstAccess == FirstAccessKind::FK_Write) {
      // Example (kcache candidate a):
      //   k=k_end:k_start
      //     a = in
      //     out = a[k-1]
      boost::optional<Interval> interval = MS.computeEnclosingAccessInterval(field.getAccessID());
      DAWN_ASSERT(interval.is_initialized());

      std::cout << "SET LOCALB "
                << field.getAccessID() /*instantiation->getNameFromAccessID(field.getAccessID()) */
                << field.getInterval() << " " << *interval << std::endl;

      DAWN_ASSERT(interval->contains(field.getInterval()));

      if(((MS.getLoopOrder() == LoopOrderKind::LK_Forward) &&
          interval->upperBound() > field.getInterval().upperBound()) ||
         ((MS.getLoopOrder() == LoopOrderKind::LK_Backward) &&
          interval->lowerBound() < field.getInterval().lowerBound()))
        return CacheCandidate{Cache::fill, boost::optional<Cache::window>(), *firstAccess,
                              *interval};
      if(((MS.getLoopOrder() == LoopOrderKind::LK_Forward) &&
          interval->lowerBound() < field.getInterval().lowerBound()) ||
         ((MS.getLoopOrder() == LoopOrderKind::LK_Backward) &&
          interval->upperBound() > field.getInterval().upperBound()))
        return CacheCandidate{Cache::bpfill,
                              boost::make_optional(Cache::window{
                                  interval->lowerBound() - field.getInterval().lowerBound(),
                                  interval->upperBound() - field.getInterval().upperBound()}),
                              *firstAccess, *interval};

      return CacheCandidate{Cache::local, boost::optional<Cache::window>(), *firstAccess,
                            *interval};
    }
    // Do we have a read in counter loop order or pointwise access?
    // TODO test getVerticalLoopOrderAccesses
    if(field.getExtents().getVerticalLoopOrderAccesses(MS.getLoopOrder()).CounterLoopOrder ||
       field.getExtents().isVerticalPointwise()) {

      // Example (kcache candidate a):
      //   1. a += a
      //   2. k=k_end:k_start
      //         a += a[k-1]

      // TODO Should not require a bpfill
      // If the field is a temporary, we only set the policy fo fill. A flush will be
      // granted if following MSS also read the temporary field
      std::cout << "SET FILLB "
                << field.getAccessID() /* instantiation->getNameFromAccessID(field.getAccessID()) */
                << std::endl;

      boost::optional<Interval> interval = MS.computeEnclosingAccessInterval(field.getAccessID());
      DAWN_ASSERT(interval.is_initialized());
      return CacheCandidate{Cache::fill, boost::optional<Cache::window>(),
                            toFirstAccess(field.getIntend()), *interval};
    }
    // Example (kcache candidate a):
    //     k=k_end-1:k_start
    //        a += a[k+1]
    std::cout << "SET BPFILL "
              << field.getAccessID() /*instantiation->getNameFromAccessID(field.getAccessID()) */
              << std::endl;

    // TODO should be a temporary, otherwise will require also a flush
    auto accessedInterval = MS.computeEnclosingAccessInterval(field.getAccessID());
    DAWN_ASSERT(accessedInterval.is_initialized());
    auto cacheWindow = boost::make_optional(
        computeCacheWindow(MS.getLoopOrder(), *accessedInterval, field.getInterval()));

    boost::optional<Interval> interval = MS.computeEnclosingAccessInterval(field.getAccessID());
    DAWN_ASSERT(interval.is_initialized());
    return CacheCandidate{Cache::bpfill, cacheWindow, toFirstAccess(field.getIntend()), *interval};
  }
  dawn_unreachable("Policy of Field not found");
}

bool PassSetCaches::isAccessIDReadAfter(
    const int accessID, std::list<std::shared_ptr<Stage>>::const_iterator stage,
    std::list<std::shared_ptr<MultiStage>>::const_iterator multiStage,
    const Stencil& stencil) const {

  for(std::list<std::shared_ptr<MultiStage>>::const_iterator multiStageIt = multiStage;
      multiStageIt != stencil.getMultiStages().end(); ++multiStageIt) {
    for(std::list<std::shared_ptr<Stage>>::const_iterator stageIt =
            (multiStageIt == multiStage) ? stage++ : (*multiStageIt)->getStages().begin();
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
  std::cout << "PASS" << std::endl;
  OptimizerContext* context = instantiation->getOptimizerContext();

  for(const auto& stencilPtr : instantiation->getStencils()) {
    std::cout << "FOR T " << std::endl;
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
          if(MS.isCached(field.getAccessID()))
            continue;

          // Field has vertical extents, can't be ij-cached
          if(!field.getExtents().isVerticalPointwise())
            continue;

          // Currently we only cache temporaries!
          if(!instantiation->isTemporaryField(field.getAccessID()))
            continue;

          if(isAccessIDReadAfter(field.getAccessID(), stageIt, multiStageIt, stencil))
            continue;
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

    std::cout << "BEYONG " << context->getOptions().UseKCaches
              << stencil.getSIRStencil()->Attributes.has(sir::Attr::AK_UseKCaches) << std::endl;
    // Set K-Caches
    if(context->getOptions().UseKCaches ||
       stencil.getSIRStencil()->Attributes.has(sir::Attr::AK_UseKCaches)) {

      std::cout << "KK " << std::endl;
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

          std::cout << "FOR " << field.getAccessID() << " " << MSIndex << std::endl;
          // Field is already cached, skip
          if(MS.isCached(field.getAccessID()))
            continue;
          std::cout << "FOR " << field.getAccessID() << " " << MSIndex << std::endl;

          // Field has horizontal extents, can't be k-cached
          if(!field.getExtents().isHorizontalPointwise())
            continue;

          // we dont know how to cache fields with out of center writes
          if(!field.getWriteExtents().isPointwise())
            continue;

          std::cout << "FORB " << instantiation->getNameFromAccessID(field.getAccessID()) << " "
                    << instantiation->isTemporaryField(field.getAccessID()) << " "
                    << field.getIntend() << " " << field.getExtents() << std::endl;

          // Currently we only cache temporaries!
          //          if(!instantiation->isTemporaryField(field.getAccessID()))
          //            continue;

          if(!instantiation->isTemporaryField(field.getAccessID()) &&
             (field.getIntend() == Field::IK_Output ||
              (field.getIntend() == Field::IK_Input && field.getExtents().isPointwise())))
            continue;

          std::cout << "FORC " << MSIndex << std::endl;

          // Determine if we need to fill the cache by analyzing the current multi-stage
          CacheCandidate policy =
              computePolicyMS1(field, instantiation->isTemporaryField(field.getAccessID()), MS);
          if(policy.intend_ == FirstAccessKind::FK_Mixed)
            continue;

          DAWN_ASSERT((policy.policy_ != Cache::fill && policy.policy_ != Cache::bpfill) ||
                      !instantiation->isTemporaryField(field.getAccessID() || mssProcessedField));
          std::cout << "Policy1 " << policy.policy_ << " " << policy.interval_ << std::endl;

          if(!instantiation->isTemporaryField(field.getAccessID()) &&
             field.getIntend() != Field::IK_Input) {
            std::cout << "SET TO FLUSH B "
                      << instantiation->getNameFromAccessID(field.getAccessID()) << std::endl;

            policy = combinePolicy(policy, field.getIntend(),
                                   CacheCandidate{Cache::CacheIOPolicy::fill,
                                                  boost::optional<Cache::window>(),
                                                  FirstAccessKind::FK_Read, field.getInterval()});
          } else {

            for(int MSIndex2 = MSIndex + 1; MSIndex2 < numMS; MSIndex2++) {
              if(!fields[MSIndex2].count(field.getAccessID()))
                continue;

              const MultiStage& nextMS = *stencil.getMultiStageFromMultiStageIndex(MSIndex2);
              const Field& fieldInNextMS = fields[MSIndex2].find(field.getAccessID())->second;

              CacheCandidate policyMS2 = computePolicyMS1(
                  fieldInNextMS, instantiation->isTemporaryField(fieldInNextMS.getAccessID()),
                  nextMS);
              // if the interval of the two cache candidate do not overlap, there is no data
              // dependency, therefore we skip it
              if(!policy.interval_.overlaps(policyMS2.interval_))
                continue;

              std::cout << "Policy2 " << policyMS2.policy_ << " " << policyMS2.interval_
                        << std::endl;
              if(policyMS2.intend_ == FirstAccessKind::FK_Mixed) {
                policy.intend_ = FirstAccessKind::FK_Mixed;
                break;
              }
              policy = combinePolicy(policy, field.getIntend(), policyMS2);
              break;
            }

            if(policy.intend_ == FirstAccessKind::FK_Mixed)
              continue;
          }

          // TODO remove
          //          auto interval = MS.computeEnclosingAccessInterval(field.getAccessID());
          Interval interval = field.getInterval();
          if(policy.policy_ == Cache::CacheIOPolicy::fill) {
            auto interval_ = MS.computeEnclosingAccessInterval(field.getAccessID());
            DAWN_ASSERT(interval_.is_initialized());
            interval = *interval_;
          }

          std::cout << "INSERTINGCACHE " << field.getAccessID() << std::endl;
          // Set the cache
          Cache& cache =
              MS.setCache(Cache::K, policy.policy_, field.getAccessID(), interval, policy.window_);

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
