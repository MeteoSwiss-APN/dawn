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
#include "dawn/Optimizer/CacheHelper.h"
#include "dawn/Support/Unreachable.h"
#include <iostream>
#include <set>
#include <vector>

namespace dawn {

namespace {

enum class FirstAccessKind { FK_Read, FK_Write, FK_Mixed };

boost::optional<FirstAccessKind> toFirstAccess(boost::optional<AccessKind> thisFirstAccess) {
  if(!thisFirstAccess.is_initialized())
    return boost::optional<FirstAccessKind>();
  switch(*thisFirstAccess) {
  case AccessKind::AK_Read:
    return boost::make_optional(FirstAccessKind::FK_Read);
    break;
  case AccessKind::AK_Write:
    return boost::make_optional(FirstAccessKind::FK_Write);
    break;
  default:
    dawn_unreachable("unknown access kind");
  }
}

boost::optional<FirstAccessKind> combineFirstAccess(boost::optional<FirstAccessKind> firstAccess,
                                                    boost::optional<AccessKind> thisFirstAccess) {
  if(!firstAccess.is_initialized()) {
    firstAccess = toFirstAccess(thisFirstAccess);
  }
  if(!thisFirstAccess.is_initialized())
    return firstAccess;

  switch(*firstAccess) {
  case FirstAccessKind::FK_Read:
    return (*thisFirstAccess == AccessKind::AK_Read)
               ? firstAccess
               : boost::make_optional(FirstAccessKind::FK_Mixed);
    break;
  case FirstAccessKind::FK_Write:
    return (*thisFirstAccess == AccessKind::AK_Write)
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
          const DoMethod* doMethod) -> boost::optional<AccessKind> {
        std::cout << " LOOPING DO " << std::endl;

        for(const auto& statementAccesssPair : doMethod->getStatementAccessesPairs()) {
          const Accesses& accesses = *statementAccesssPair->getAccesses();
          if(accesses.hasAccess(AccessID))
            return boost::make_optional(accesses.getFirstAccessKind(AccessID));
        }
        return boost::optional<AccessKind>();
      };
      boost::optional<FirstAccessKind> firstAccess;

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
///                ||    local    ||     fill      ||   bpfill     ||
///       =========++=============++===============++==============++
///        local   ||    local    |      flush     |    epflush     |
///       =========++-------------+----------------+----------------+
///  MS1   fill    ||    fill     | fill_and_flush | fill_and_flush |
///       =========++-------------+----------------+----------------+
///        bpfill  ||   pbfill    | fill_and_flush | fill_and_flush |
///       =========++-------------+----------------+----------------+
///
std::pair<Cache::CacheIOPolicy, boost::optional<Cache::window>>
combinePolicy(std::pair<Cache::CacheIOPolicy, boost::optional<Cache::window>> const& MS1Policy,
              std::pair<Cache::CacheIOPolicy, boost::optional<Cache::window>> const& MS2Policy) {
  std::cout << "COMBINE POLICIES " << MS1Policy.first << " " << MS2Policy.first << std::endl;
  // TODO properly compute the window
  if(MS1Policy.first == Cache::local) {
    if(MS2Policy.first == Cache::fill)
      return std::make_pair(Cache::flush, boost::make_optional(Cache::window{})); // flush
    if(MS2Policy.first == Cache::bpfill) {
      DAWN_ASSERT(MS2Policy.second.is_initialized());
      auto const& window = *(MS2Policy.second);
      return std::make_pair(
          Cache::epflush, boost::make_optional(Cache::window{-window.m_p, -window.m_m})); // epflush
    }
    if(MS2Policy.first == Cache::local)
      return MS2Policy; // local
    dawn_unreachable("Not valid policy");
  }
  if(MS1Policy.first == Cache::fill || MS1Policy.first == Cache::bpfill) {
    if(MS2Policy.first == Cache::fill || MS2Policy.first == Cache::bpfill)
      return {Cache::fill_and_flush, MS1Policy.second};
    if(MS2Policy.first == Cache::local)
      return MS1Policy; // fill, pbfill
    std::cout << "PO  " << MS1Policy.first << " " << MS2Policy.first << std::endl;
    dawn_unreachable("Not valid policy");
  }
  dawn_unreachable("invalid policy combination");
}

} // anonymous namespace

PassSetCaches::PassSetCaches() : Pass("PassSetCaches") {}

Cache::window PassSetCaches::computeCacheWindow(LoopOrderKind loopOrder,
                                                Interval const& accessedInterval,
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

std::pair<Cache::CacheIOPolicy, boost::optional<Cache::window>>
PassSetCaches::computePolicyMS1(Field const& field, bool isTemporaryField, MultiStage const& MS) {

  if(field.getIntend() == Field::IK_Input) {
    std::cout << "SET FILL "
              << field.getAccessID() /* instantiation->getNameFromAccessID(field.getAccessID()) */
              << std::endl;
    return {Cache::fill, boost::optional<Cache::window>()};
  }
  if(field.getIntend() == Field::IK_Output) {
    std::cout << "SET LOCAL "
              << field.getAccessID() /*instantiation->getNameFromAccessID(field.getAccessID())*/
              << std::endl;
    // we do not cache output only normal fields since there is not data reuse
    DAWN_ASSERT(isTemporaryField);
    return {Cache::local, boost::optional<Cache::window>()};
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
      std::cout << "SET LOCALB "
                << field.getAccessID() /*instantiation->getNameFromAccessID(field.getAccessID()) */
                << std::endl;

      return {Cache::local, boost::optional<Cache::window>()};
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

      return {Cache::fill, boost::optional<Cache::window>()};
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

    return {Cache::bpfill, cacheWindow};
  }
  dawn_unreachable("Policy of Field not found");
}

bool PassSetCaches::run(const std::shared_ptr<StencilInstantiation>& instantiation) {
  std::cout << "PASS" << std::endl;
  OptimizerContext* context = instantiation->getOptimizerContext();

  for(const auto& stencilPtr : instantiation->getStencils()) {
    std::cout << "FOR T " << std::endl;
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

          std::cout << "HERE " << std::endl;
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
      for(int MSIndex = 0; MSIndex < numMS; ++MSIndex) {
        for(const auto& AccessIDFieldPair : fields[MSIndex]) {
          MultiStage& MS = *stencil.getMultiStageFromMultiStageIndex(MSIndex);
          const Field& field = AccessIDFieldPair.second;

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
          std::pair<Cache::CacheIOPolicy, boost::optional<Cache::window>> policy =
              computePolicyMS1(field, instantiation->isTemporaryField(field.getAccessID()), MS);

          DAWN_ASSERT((policy.first != Cache::fill && policy.first != Cache::bpfill &&
                       policy.first != Cache::fill_and_flush) ||
                      !instantiation->isTemporaryField(field.getAccessID()));
          std::cout << "Policy1 " << policy.first << std::endl;

          if(!instantiation->isTemporaryField(field.getAccessID()) &&
             field.getIntend() != Field::IK_Input) {
            std::cout << "SET TO FLUSH B "
                      << instantiation->getNameFromAccessID(field.getAccessID()) << std::endl;

            policy = combinePolicy(policy, std::make_pair(Cache::CacheIOPolicy::fill,
                                                          boost::optional<Cache::window>()));
          } else {

            for(int MSIndex2 = MSIndex + 1; MSIndex2 < numMS; MSIndex2++) {
              if(!fields[MSIndex2].count(field.getAccessID()))
                continue;

              const MultiStage& nextMS = *stencil.getMultiStageFromMultiStageIndex(MSIndex2);
              const Field& fieldInNextMS = fields[MSIndex2].find(field.getAccessID())->second;

              std::pair<Cache::CacheIOPolicy, boost::optional<Cache::window>> policyMS2 =
                  computePolicyMS1(fieldInNextMS,
                                   instantiation->isTemporaryField(fieldInNextMS.getAccessID()),
                                   nextMS);

              policy = combinePolicy(policy, policyMS2);
              break;
            }
          }

          // TODO remove
          //          auto interval = MS.computeEnclosingAccessInterval(field.getAccessID());
          auto interval = field.getInterval();

          std::cout << "INSERTINGCACHE " << field.getAccessID() << std::endl;
          // Set the cache
          Cache& cache =
              MS.setCache(Cache::K, policy.first, field.getAccessID(), interval, policy.second);

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
