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
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/IIR/IIR.pb.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/PassComputeStageExtents.h"
#include "dawn/Optimizer/PassInlining.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/ASTSerializer.h"
#include <fstream>
#include <google/protobuf/util/json_util.h>
#include <list>
#include <stack>
#include <tuple>
#include <utility>

namespace dawn {
static void setAccesses(proto::iir::Accesses* protoAccesses,
                        const std::shared_ptr<iir::Accesses>& accesses) {
  auto protoReadAccesses = protoAccesses->mutable_readaccess();
  for(auto IDExtentsPair : accesses->getReadAccesses()) {
    proto::iir::Extents protoExtents;
    for(auto extent : IDExtentsPair.second.getExtents()) {
      auto protoExtent = protoExtents.add_extents();
      protoExtent->set_minus(extent.Minus);
      protoExtent->set_plus(extent.Plus);
    }
    protoReadAccesses->insert({IDExtentsPair.first, protoExtents});
  }

  auto protoWriteAccesses = protoAccesses->mutable_writeaccess();
  for(auto IDExtentsPair : accesses->getWriteAccesses()) {
    proto::iir::Extents protoExtents;
    for(auto extent : IDExtentsPair.second.getExtents()) {
      auto protoExtent = protoExtents.add_extents();
      protoExtent->set_minus(extent.Minus);
      protoExtent->set_plus(extent.Plus);
    }
    protoWriteAccesses->insert({IDExtentsPair.first, protoExtents});
  }
}

static std::shared_ptr<dawn::Statement>
makeStatement(const proto::iir::StencilDescStatement* protoStatement) {
  std::vector<sir::StencilCall*> stackTrace;
  auto stmt = makeStmt(protoStatement->stmt());

  // only create the shared pointer if necessary
  if(protoStatement->stacktrace().size()) {
    for(auto protoStencilCall : protoStatement->stacktrace()) {
      sir::StencilCall call(protoStencilCall.callee());
      setLocation(protoStencilCall.mutable_loc(), call.Loc);
      for(const auto& protoField : protoStencilCall.arguments()) {
        call.Args.emplace_back(makeField(protoField));
      }
      stackTrace.push_back(&call);
    }
    return std::make_shared<Statement>(
        stmt, std::make_shared<std::vector<sir::StencilCall*>>(stackTrace));
  }
  return std::make_shared<Statement>(stmt, nullptr);
}

static iir::Extents makeExtents(const proto::iir::Extents* protoExtents) {
  int dim1minus = protoExtents->extents()[0].minus();
  int dim1plus = protoExtents->extents()[0].plus();
  int dim2minus = protoExtents->extents()[1].minus();
  int dim2plus = protoExtents->extents()[1].plus();
  int dim3minus = protoExtents->extents()[2].minus();
  int dim3plus = protoExtents->extents()[2].plus();
  return iir::Extents(dim1minus, dim1plus, dim2minus, dim2plus, dim3minus, dim3plus);
}

static void
serializeStmtAccessPair(proto::iir::StatementAccessPair* protoStmtAccessPair,
                        const std::unique_ptr<iir::StatementAccessesPair>& stmtAccessPair) {
  // serialize the statement
  ProtoStmtBuilder builder(protoStmtAccessPair->mutable_aststmt());
  stmtAccessPair->getStatement()->ASTStmt->accept(builder);

  // check if caller / callee accesses are initialized, and if so, fill them
  if(stmtAccessPair->getCallerAccesses()) {
    setAccesses(protoStmtAccessPair->mutable_calleraccesses(), stmtAccessPair->getCallerAccesses());
  }
  if(stmtAccessPair->getCalleeAccesses()) {
    setAccesses(protoStmtAccessPair->mutable_calleeaccesses(), stmtAccessPair->getCalleeAccesses());
  }
}

static void setCache(proto::iir::Cache* protoCache, const iir::Cache& cache) {
  protoCache->set_accessid(cache.getCachedFieldAccessID());
  switch(cache.getCacheIOPolicy()) {
  case iir::Cache::bpfill:
    protoCache->set_policy(proto::iir::Cache_CachePolicy_CP_BPFill);
    break;
  case iir::Cache::epflush:
    protoCache->set_policy(proto::iir::Cache_CachePolicy_CP_EPFlush);
    break;
  case iir::Cache::fill:
    protoCache->set_policy(proto::iir::Cache_CachePolicy_CP_Fill);
    break;
  case iir::Cache::fill_and_flush:
    protoCache->set_policy(proto::iir::Cache_CachePolicy_CP_FillFlush);
    break;
  case iir::Cache::flush:
    protoCache->set_policy(proto::iir::Cache_CachePolicy_CP_Flush);
    break;
  case iir::Cache::local:
    protoCache->set_policy(proto::iir::Cache_CachePolicy_CP_Local);
    break;
  case iir::Cache::unknown:
    protoCache->set_policy(proto::iir::Cache_CachePolicy_CP_Unknown);
    break;
  default:
    dawn_unreachable("unknown cache policy");
  }
  switch(cache.getCacheType()) {
  case iir::Cache::bypass:
    protoCache->set_type(proto::iir::Cache_CacheType_CT_Bypass);
    break;
  case iir::Cache::IJ:
    protoCache->set_type(proto::iir::Cache_CacheType_CT_IJ);
    break;
  case iir::Cache::IJK:
    protoCache->set_type(proto::iir::Cache_CacheType_CT_IJK);
    break;
  case iir::Cache::K:
    protoCache->set_type(proto::iir::Cache_CacheType_CT_K);
    break;
  default:
    dawn_unreachable("unknown cache type");
  }
  if(cache.getInterval().is_initialized()) {
    auto sirInterval = cache.getInterval()->asSIRInterval();
    setInterval(protoCache->mutable_interval(), &sirInterval);
  }
  if(cache.getEnclosingAccessedInterval().is_initialized()) {
    auto sirInterval = cache.getEnclosingAccessedInterval()->asSIRInterval();
    setInterval(protoCache->mutable_enclosingaccessinterval(), &sirInterval);
  }
  if(cache.getWindow().is_initialized()) {
    protoCache->mutable_cachewindow()->set_minus(cache.getWindow()->m_m);
    protoCache->mutable_cachewindow()->set_plus(cache.getWindow()->m_p);
  }
}

static iir::Cache makeCache(const proto::iir::Cache* protoCache) {
  iir::Cache::CacheTypeKind cacheType;
  iir::Cache::CacheIOPolicy cachePolicy;
  boost::optional<iir::Interval> interval;
  boost::optional<iir::Interval> enclosingInverval;
  boost::optional<iir::Cache::window> cacheWindow;
  int ID = protoCache->accessid();
  switch(protoCache->type()) {
  case proto::iir::Cache_CacheType_CT_Bypass:
    cacheType = iir::Cache::bypass;
    break;
  case proto::iir::Cache_CacheType_CT_IJ:
    cacheType = iir::Cache::IJ;
    break;
  case proto::iir::Cache_CacheType_CT_IJK:
    cacheType = iir::Cache::IJK;
    break;
  case proto::iir::Cache_CacheType_CT_K:
    cacheType = iir::Cache::K;
    break;
  default:
    dawn_unreachable("unknow cache type");
  }
  switch(protoCache->policy()) {
  case proto::iir::Cache_CachePolicy_CP_BPFill:
    cachePolicy = iir::Cache::bpfill;
    break;
  case proto::iir::Cache_CachePolicy_CP_EPFlush:
    cachePolicy = iir::Cache::epflush;
    break;
  case proto::iir::Cache_CachePolicy_CP_Fill:
    cachePolicy = iir::Cache::fill;
    break;
  case proto::iir::Cache_CachePolicy_CP_FillFlush:
    cachePolicy = iir::Cache::fill_and_flush;
    break;
  case proto::iir::Cache_CachePolicy_CP_Flush:
    cachePolicy = iir::Cache::flush;
    break;
  case proto::iir::Cache_CachePolicy_CP_Local:
    cachePolicy = iir::Cache::local;
    break;
  case proto::iir::Cache_CachePolicy_CP_Unknown:
    cachePolicy = iir::Cache::unknown;
    break;
  default:
    dawn_unreachable("unknown cache policy");
  }
  if(protoCache->has_interval()) {
    interval = boost::make_optional(*makeInterval(protoCache->interval()));
  }
  if(protoCache->has_enclosingaccessinterval()) {
    enclosingInverval = boost::make_optional(*makeInterval(protoCache->enclosingaccessinterval()));
  }
  if(protoCache->has_cachewindow()) {
    cacheWindow = boost::make_optional(
        iir::Cache::window{protoCache->cachewindow().plus(), protoCache->cachewindow().minus()});
  }

  return iir::Cache(cacheType, cachePolicy, ID, interval, enclosingInverval, cacheWindow);
}

static void computeInitialDerivedInfo(const std::shared_ptr<iir::StencilInstantiation>& target) {
  for(auto IDtoNamePair : target->getAccessIDToNameMap()) {
    target->getNameToAccessIDMap().insert({IDtoNamePair.second, IDtoNamePair.first});
  }

  for(auto StencilCallToIDPair : target->getIDToStencilCallMap()) {
    target->getStencilCallToStencilIDMap().insert(
        {StencilCallToIDPair.second, StencilCallToIDPair.first});
  }
  //////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////
  /// WITTODO: check with carlos why this does not work for stmtaccesspairs
  //////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////
  for(const auto& leaf : iterateIIROver<iir::DoMethod>(*target->getIIR())) {
    leaf->update(iir::NodeUpdateType::level);
  }
  for(const auto& leaf : iterateIIROver<iir::Stage>(*target->getIIR())) {
    leaf->update(iir::NodeUpdateType::levelAndTreeAbove);
  }
  PassComputeStageExtents passStageExtent;
  passStageExtent.run(target);
}

void IIRSerializer::serializeMetaData(proto::iir::StencilInstantiation& target,
                                      iir::StencilMetaInformation& metaData) {
  auto protoMetaData = target.mutable_metadata();

  // Filling Field: map<int32, string> AccessIDToName = 1;
  auto& protoAccessIDtoNameMap = *protoMetaData->mutable_accessidtoname();
  for(const auto& accessIDtoNamePair : metaData.AccessIDToNameMap_) {
    protoAccessIDtoNameMap.insert({accessIDtoNamePair.first, accessIDtoNamePair.second});
  }
  // Filling Field: repeated ExprIDPair ExprToAccessID = 2;
  auto& protoExprIDtoAccessID = *protoMetaData->mutable_expridtoaccessid();
  for(const auto& exprIDToAccessIDPair : metaData.ExprIDToAccessIDMap_) {
    protoExprIDtoAccessID.insert({exprIDToAccessIDPair.first, exprIDToAccessIDPair.second});
  }
  // Filling Field: repeated StmtIDPair StmtToAccessID = 3;
  auto& protoStmtIDtoAccessID = *protoMetaData->mutable_stmtidtoaccessid();
  for(const auto& stmtIDToAccessIDPair : metaData.StmtIDToAccessIDMap_) {
    protoStmtIDtoAccessID.insert({stmtIDToAccessIDPair.first, stmtIDToAccessIDPair.second});
  }
  // Filling Field: map<int32, string> LiteralIDToName = 4;
  auto& protoLiteralIDToNameMap = *protoMetaData->mutable_literalidtoname();
  for(const auto& literalIDtoNamePair : metaData.LiteralAccessIDToNameMap_) {
    protoLiteralIDToNameMap.insert({literalIDtoNamePair.first, literalIDtoNamePair.second});
  }
  // Filling Field: repeated int32 FieldAccessIDs = 5;
  for(int fieldAccessID : metaData.FieldAccessIDSet_) {
    protoMetaData->add_fieldaccessids(fieldAccessID);
  }
  // Filling Field: repeated int32 APIFieldIDs = 6;
  for(int apifieldID : metaData.apiFieldIDs_) {
    protoMetaData->add_apifieldids(apifieldID);
  }
  // Filling Field: repeated int32 TemporaryFieldIDs = 7;
  for(int temporaryFieldID : metaData.TemporaryFieldAccessIDSet_) {
    protoMetaData->add_temporaryfieldids(temporaryFieldID);
  }
  // Filling Field: repeated int32 GlobalVariableIDs = 8;
  for(int globalVariableID : metaData.GlobalVariableAccessIDSet_) {
    protoMetaData->add_globalvariableids(globalVariableID);
  }

  // Filling Field: VariableVersions versionedFields = 9;
  auto protoVariableVersions = protoMetaData->mutable_versionedfields();
  auto& protoVariableVersionMap = *protoVariableVersions->mutable_variableversionmap();
  auto variableVersions = metaData.variableVersions_;
  for(auto& IDtoVectorOfVersionsPair : variableVersions.variableVersionsMap_) {
    proto::iir::AllVersionedFields protoFieldVersions;
    for(int id : *(IDtoVectorOfVersionsPair.second)) {
      protoFieldVersions.add_allids(id);
    }
    protoVariableVersionMap.insert({IDtoVectorOfVersionsPair.first, protoFieldVersions});
  }

  // Filling Field: repeated StencilDescStatement stencilDescStatements = 10;
  for(const auto& stencilDescStmt : metaData.stencilDescStatements_) {
    auto protoStmt = protoMetaData->add_stencildescstatements();
    ProtoStmtBuilder builder(protoStmt->mutable_stmt());
    stencilDescStmt->ASTStmt->accept(builder);
    if(stencilDescStmt->StackTrace) {
      for(auto sirStackTrace : *(stencilDescStmt->StackTrace)) {
        auto protoStackTrace = protoStmt->add_stacktrace();
        setLocation(protoStackTrace->mutable_loc(), sirStackTrace->Loc);
        protoStackTrace->set_callee(sirStackTrace->Callee);
        for(auto argument : sirStackTrace->Args) {
          auto arg = protoStackTrace->add_arguments();
          arg->set_name(argument->Name);
          setLocation(arg->mutable_loc(), argument->Loc);
          arg->set_is_temporary(argument->IsTemporary);
          for(int dim : argument->fieldDimensions) {
            arg->add_field_dimensions(dim);
          }
        }
      }
    }
  }
  // Filling Field: map<int32, dawn.proto.statements.StencilCallDeclStmt> IDToStencilCall = 11;
  auto& protoIDToStencilCallMap = *protoMetaData->mutable_idtostencilcall();
  for(auto IDToStencilCall : metaData.IDToStencilCallMap_) {
    proto::statements::Stmt protoStencilCall;
    ProtoStmtBuilder builder(&protoStencilCall);
    IDToStencilCall.second->accept(builder);
    protoIDToStencilCallMap.insert({IDToStencilCall.first, protoStencilCall});
  }

  // Filling Field:
  // map<string, dawn.proto.statements.BoundaryConditionDeclStmt> FieldnameToBoundaryCondition = 12;
  auto& protoFieldNameToBC = *protoMetaData->mutable_fieldnametoboundarycondition();
  for(auto fieldNameToBC : metaData.FieldnameToBoundaryConditionMap_) {
    proto::statements::Stmt protoStencilCall;
    ProtoStmtBuilder builder(&protoStencilCall);
    fieldNameToBC.second->accept(builder);
    protoFieldNameToBC.insert({fieldNameToBC.first, protoStencilCall});
  }

  // Filling Field: map<int32, Array3i> fieldIDtoLegalDimensions = 13;
  auto& protoInitializedDimensionsMap = *protoMetaData->mutable_fieldidtolegaldimensions();
  for(auto IDToLegalDimension : metaData.fieldIDToInitializedDimensionsMap_) {
    proto::iir::Array3i array;
    array.set_int1(IDToLegalDimension.second[0]);
    array.set_int2(IDToLegalDimension.second[1]);
    array.set_int3(IDToLegalDimension.second[2]);
    protoInitializedDimensionsMap.insert({IDToLegalDimension.first, array});
  }

  // Filling Field: map<string, GlobalValueAndType> GlobalVariableToValue = 14;
  auto& protoGlobalVariableMap = *protoMetaData->mutable_globalvariabletovalue();
  for(auto& globalToValue : metaData.globalVariableMap_) {
    proto::iir::GlobalValueAndType protoGlobalToStore;
    double value = -1;
    bool valueIsSet = false;
    switch(globalToValue.second->getType()) {
    case sir::Value::Boolean:
      if(!globalToValue.second->empty()) {
        value = globalToValue.second->getValue<bool>();
        valueIsSet = true;
      }
      protoGlobalToStore.set_type(proto::iir::GlobalValueAndType_TypeKind_Boolean);
      break;
    case sir::Value::Integer:
      std::cout << "serialize int" << std::endl;
      if(!globalToValue.second->empty()) {
        value = globalToValue.second->getValue<int>();
        valueIsSet = true;
      }
      protoGlobalToStore.set_type(proto::iir::GlobalValueAndType_TypeKind_Integer);
      break;
    case sir::Value::Double:
      std::cout << "serialize double" << std::endl;
      if(!globalToValue.second->empty()) {
        value = globalToValue.second->getValue<double>();
        valueIsSet = true;
      }
      protoGlobalToStore.set_type(proto::iir::GlobalValueAndType_TypeKind_Double);
      break;
    default:
      dawn_unreachable("non-supported global type");
    }
    if(valueIsSet) {
      protoGlobalToStore.set_value(value);
    }
    protoGlobalToStore.set_valueisset(valueIsSet);
    protoGlobalVariableMap.insert({globalToValue.first, protoGlobalToStore});
  }

  // Filling Field: dawn.proto.statements.SourceLocation stencilLocation = 15;
  auto protoStencilLoc = protoMetaData->mutable_stencillocation();
  protoStencilLoc->set_column(metaData.stencilLocation_.Column);
  protoStencilLoc->set_line(metaData.stencilLocation_.Line);

  // Filling Field: string stencilMName = 16;
  protoMetaData->set_stencilname(metaData.stencilName_);
}

void IIRSerializer::serializeIIR(proto::iir::StencilInstantiation& target,
                                 const std::unique_ptr<iir::IIR>& iir) {
  auto protoIIR = target.mutable_internalir();
  // Get all the stencils
  for(const auto& stencils : iir->getChildren()) {
    // creation of a new protobuf stencil
    auto protoStencil = protoIIR->add_stencils();
    // Information other than the children
    protoStencil->set_stencilid(stencils->getStencilID());
    auto protoAttribute = protoStencil->mutable_attr();
    if(stencils->getStencilAttributes().has(sir::Attr::AK_MergeDoMethods)) {
      protoAttribute->add_attributes(
          proto::iir::Attributes::StencilAttributes::Attributes_StencilAttributes_MergeDoMethods);
    }
    if(stencils->getStencilAttributes().has(sir::Attr::AK_MergeStages)) {
      protoAttribute->add_attributes(
          proto::iir::Attributes::StencilAttributes::Attributes_StencilAttributes_MergeStages);
    }
    if(stencils->getStencilAttributes().has(sir::Attr::AK_MergeTemporaries)) {
      protoAttribute->add_attributes(
          proto::iir::Attributes::StencilAttributes::Attributes_StencilAttributes_MergeTemporaries);
    }
    if(stencils->getStencilAttributes().has(sir::Attr::AK_NoCodeGen)) {
      protoAttribute->add_attributes(
          proto::iir::Attributes::StencilAttributes::Attributes_StencilAttributes_NoCodeGen);
    }
    if(stencils->getStencilAttributes().has(sir::Attr::AK_UseKCaches)) {
      protoAttribute->add_attributes(
          proto::iir::Attributes::StencilAttributes::Attributes_StencilAttributes_UseKCaches);
    }

    // adding it's children
    for(const auto& multistages : stencils->getChildren()) {
      // creation of a protobuf multistage
      auto protoMSS = protoStencil->add_multistages();
      // Information other than the children
      if(multistages->getLoopOrder() == dawn::iir::LoopOrderKind::LK_Forward) {
        protoMSS->set_looporder(proto::iir::MultiStage::Forward);
      } else if(multistages->getLoopOrder() == dawn::iir::LoopOrderKind::LK_Backward) {
        protoMSS->set_looporder(proto::iir::MultiStage::Backward);
      } else {
        protoMSS->set_looporder(proto::iir::MultiStage::Parallel);
      }
      protoMSS->set_multistageid(multistages->getID());
      auto& protoMSSCacheMap = *protoMSS->mutable_caches();
      for(const auto& IDCachePair : multistages->getCaches()) {
        proto::iir::Cache protoCache;
        setCache(&protoCache, IDCachePair.second);
        protoMSSCacheMap.insert({IDCachePair.first, protoCache});
      }
      // adding it's children
      for(const auto& stages : multistages->getChildren()) {
        auto protoStage = protoMSS->add_stages();
        // Information other than the children
        protoStage->set_stageid(stages->getStageID());

        // adding it's children
        for(const auto& domethod : stages->getChildren()) {
          auto protoDoMethod = protoStage->add_domethods();
          // Information other than the children
          dawn::sir::Interval interval = domethod->getInterval().asSIRInterval();
          setInterval(protoDoMethod->mutable_interval(), &interval);
          protoDoMethod->set_domethodid(domethod->getID());

          // adding it's children
          for(const auto& stmtaccesspair : domethod->getChildren()) {
            auto protoStmtAccessPair = protoDoMethod->add_stmtaccesspairs();
            serializeStmtAccessPair(protoStmtAccessPair, stmtaccesspair);
          }
        }
      }
    }
  }
}

std::string
IIRSerializer::serializeImpl(const std::shared_ptr<iir::StencilInstantiation>& instantiation,
                             dawn::IIRSerializer::SerializationKind kind) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  // Before Serialization we need to ensure there are no stencilfunctions present. This is why we
  // inline everything here.
  /////////////////////////////// WITTODO //////////////////////////////////////////////////////////
  //==------------------------------------------------------------------------------------------==//
  // After we have the merge of carlos' new inliner that distinguishes between full inlining (as
  // used for optimized code generation) and precomputation (store stencil function computation one
  // for one into temporaries), we need to make sure we use the latter as those expressions can be
  // properly flagged to be revertible and we can actually go back. An example here:
  //
  // stencil_function harm_avg{
  //     return 0.5*(u[i+1] + u[i-1]);
  // }
  // stencil_function upwind_flux{
  //     return u[j+1] - u[j]
  //}
  //
  // out = upwind_flux(harm_avg(u))
  //
  // can be represented either by [precomputation]:
  //
  // tmp_1 = 0.5*(u[i+1] + u[i-1])
  // tmp_2 = tmp_1[j+1] - tmp_1[j]
  // out = tmp_2
  //
  // or by [full inlining]
  //
  // out = 0.5*(u[i+1, j+1] + u[i-1, j+1]) - 0.5*(u[i+1] + u[i-1])
  //==------------------------------------------------------------------------------------------==//

  PassInlining inliner(PassInlining::InlineStrategyKind::IK_Precomputation);
  inliner.run(instantiation);

  using namespace dawn::proto::iir;
  proto::iir::StencilInstantiation protoStencilInstantiation;
  serializeMetaData(protoStencilInstantiation, instantiation->getMetaData());
  serializeIIR(protoStencilInstantiation, instantiation->getIIR());
  protoStencilInstantiation.set_filename(instantiation->getMetaData().fileName_);

  // Encode the message
  std::string str;
  switch(kind) {
  case dawn::IIRSerializer::SK_Json: {
    google::protobuf::util::JsonPrintOptions options;
    options.add_whitespace = true;
    options.always_print_primitive_fields = true;
    options.preserve_proto_field_names = true;
    auto status =
        google::protobuf::util::MessageToJsonString(protoStencilInstantiation, &str, options);
    if(!status.ok())
      throw std::runtime_error(dawn::format("cannot serialize IIR: %s", status.ToString()));
    break;
  }
  case dawn::IIRSerializer::SK_Byte: {
    if(!protoStencilInstantiation.SerializeToString(&str))
      throw std::runtime_error(dawn::format("cannot serialize IIR:"));
    break;
  }
  default:
    dawn_unreachable("invalid SerializationKind");
  }

  return str;
}

void IIRSerializer::deserializeMetaData(std::shared_ptr<iir::StencilInstantiation>& target,
                                        const proto::iir::StencilMetaInfo& protoMetaData) {
  auto& metadata = target->getMetaData();
  for(auto IDtoName : protoMetaData.accessidtoname()) {
    metadata.AccessIDToNameMap_.insert({IDtoName.first, IDtoName.second});
  }

  for(auto exprIDToAccessID : protoMetaData.expridtoaccessid()) {
    metadata.ExprIDToAccessIDMap_[exprIDToAccessID.first] = exprIDToAccessID.second;
  }
  for(auto stmtIDToAccessID : protoMetaData.stmtidtoaccessid()) {
    metadata.StmtIDToAccessIDMap_[stmtIDToAccessID.first] = stmtIDToAccessID.second;
  }
  for(auto literalIDToName : protoMetaData.literalidtoname()) {
    metadata.LiteralAccessIDToNameMap_[literalIDToName.first] = literalIDToName.second;
  }
  for(auto fieldaccessID : protoMetaData.fieldaccessids()) {
    metadata.FieldAccessIDSet_.insert(fieldaccessID);
  }
  for(auto ApiFieldID : protoMetaData.apifieldids()) {
    metadata.apiFieldIDs_.push_back(ApiFieldID);
  }
  for(auto temporaryFieldID : protoMetaData.temporaryfieldids()) {
    metadata.TemporaryFieldAccessIDSet_.insert(temporaryFieldID);
  }
  for(auto globalVariableID : protoMetaData.globalvariableids()) {
    metadata.GlobalVariableAccessIDSet_.insert(globalVariableID);
  }

  for(auto variableVersionMap : protoMetaData.versionedfields().variableversionmap()) {
    std::shared_ptr<std::vector<int>> versions = std::make_shared<std::vector<int>>();
    for(auto versionedID : variableVersionMap.second.allids()) {
      versions->push_back(versionedID);
      metadata.variableVersions_.versionIDs_.insert(versionedID);
      metadata.variableVersions_.versionToOriginalVersionMap_.emplace(versionedID,
                                                                      variableVersionMap.first);
    }
    metadata.variableVersions_.insert(variableVersionMap.first, versions);
  }

  //  for(auto versionID : protoMetaData.versionedfields().versionids()) {
  //    metadata.variableVersions_.versionIDs_.insert(versionID);
  //  }

  //  for(auto VersionIDOriginalIDPair : protoMetaData.versionedfields().versionidtooriginalid()) {
  //    metadata.variableVersions_.versionToOriginalVersionMap_.insert(VersionIDOriginalIDPair);
  //  }

  for(auto stencilDescStmt : protoMetaData.stencildescstatements()) {
    metadata.stencilDescStatements_.push_back(makeStatement(&stencilDescStmt));
  }

  for(auto IDToCall : protoMetaData.idtostencilcall()) {
    auto call = IDToCall.second;
    std::shared_ptr<sir::StencilCall> sirStencilCall = std::make_shared<sir::StencilCall>(
        call.stencil_call_decl_stmt().stencil_call().callee(),
        makeLocation(call.stencil_call_decl_stmt().stencil_call()));
    for(const auto& protoField : call.stencil_call_decl_stmt().stencil_call().arguments()) {
      auto field = makeField(protoField);
      sirStencilCall->Args.push_back(field);
    }
    auto stmt = std::make_shared<StencilCallDeclStmt>(sirStencilCall,
                                                      makeLocation(call.stencil_call_decl_stmt()));
    stmt->setID(call.stencil_call_decl_stmt().id());
    metadata.IDToStencilCallMap_.insert({IDToCall.first, stmt});
  }

  for(auto FieldnameToBC : protoMetaData.fieldnametoboundarycondition()) {
    std::shared_ptr<BoundaryConditionDeclStmt> bc =
        dyn_pointer_cast<BoundaryConditionDeclStmt>(makeStmt((FieldnameToBC.second)));
    metadata.FieldnameToBoundaryConditionMap_[FieldnameToBC.first] = bc;
  }

  for(auto fieldIDInitializedDims : protoMetaData.fieldidtolegaldimensions()) {
    Array3i dims{fieldIDInitializedDims.second.int1(), fieldIDInitializedDims.second.int2(),
                 fieldIDInitializedDims.second.int3()};
    metadata.fieldIDToInitializedDimensionsMap_[fieldIDInitializedDims.first] = dims;
  }

  for(auto GlobalToValue : protoMetaData.globalvariabletovalue()) {
    std::shared_ptr<sir::Value> value = std::make_shared<sir::Value>();
    switch(GlobalToValue.second.type()) {
    case proto::iir::GlobalValueAndType_TypeKind_Boolean:
      value->setType(sir::Value::Boolean);
      break;
    case proto::iir::GlobalValueAndType_TypeKind_Integer:
      value->setType(sir::Value::Integer);
      std::cout << "set to int" << std::endl;
      break;
    case proto::iir::GlobalValueAndType_TypeKind_Double:
      value->setType(sir::Value::Double);
      std::cout << "set to double" << std::endl;
      break;
    default:
      dawn_unreachable("unsupported type");
    }
    if(GlobalToValue.second.valueisset()) {
      value->setValue(GlobalToValue.second.value());
    }
    metadata.globalVariableMap_[GlobalToValue.first] = value;
  }

  metadata.stencilLocation_.Column = protoMetaData.stencillocation().column();
  metadata.stencilLocation_.Line = protoMetaData.stencillocation().line();

  metadata.stencilName_ = protoMetaData.stencilname();
}

void IIRSerializer::deserializeIIR(std::shared_ptr<iir::StencilInstantiation>& target,
                                   const proto::iir::IIR& protoIIR) {
  int stencilPos = 0;
  for(const auto& protoStencils : protoIIR.stencils()) {
    int mssPos = 0;
    sir::Attr attributes;
    target->getIIR()->insertChild(
        make_unique<iir::Stencil>(*target, attributes, protoStencils.stencilid()),
        target->getIIR());
    const auto& IIRStencil = target->getIIR()->getChild(stencilPos++);

    for(auto attribute : protoStencils.attr().attributes()) {
      if(attribute ==
         proto::iir::Attributes::StencilAttributes::Attributes_StencilAttributes_MergeDoMethods) {
        IIRStencil->getStencilAttributes().set(sir::Attr::AK_MergeDoMethods);
      }
      if(attribute ==
         proto::iir::Attributes::StencilAttributes::Attributes_StencilAttributes_MergeStages) {
        IIRStencil->getStencilAttributes().set(sir::Attr::AK_MergeStages);
      }
      if(attribute ==
         proto::iir::Attributes::StencilAttributes::Attributes_StencilAttributes_MergeTemporaries) {
        IIRStencil->getStencilAttributes().set(sir::Attr::AK_MergeTemporaries);
      }
      if(attribute ==
         proto::iir::Attributes::StencilAttributes::Attributes_StencilAttributes_NoCodeGen) {
        IIRStencil->getStencilAttributes().set(sir::Attr::AK_NoCodeGen);
      }
      if(attribute ==
         proto::iir::Attributes::StencilAttributes::Attributes_StencilAttributes_UseKCaches) {
        IIRStencil->getStencilAttributes().set(sir::Attr::AK_UseKCaches);
      }
    }

    for(const auto& protoMSS : protoStencils.multistages()) {
      int stagePos = 0;
      iir::LoopOrderKind looporder;
      if(protoMSS.looporder() == proto::iir::MultiStage_LoopOrder::MultiStage_LoopOrder_Backward) {
        looporder = iir::LoopOrderKind::LK_Backward;
      }
      if(protoMSS.looporder() == proto::iir::MultiStage_LoopOrder::MultiStage_LoopOrder_Forward) {
        looporder = iir::LoopOrderKind::LK_Forward;
      }
      if(protoMSS.looporder() == proto::iir::MultiStage_LoopOrder::MultiStage_LoopOrder_Parallel) {
        looporder = iir::LoopOrderKind::LK_Parallel;
      }
      (IIRStencil)->insertChild(make_unique<iir::MultiStage>(*target, looporder));

      const auto& IIRMSS = (IIRStencil)->getChild(mssPos++);
      IIRMSS->setID(protoMSS.multistageid());

      for(const auto& IDCachePair : protoMSS.caches()) {
        IIRMSS->getCaches().insert({IDCachePair.first, makeCache(&IDCachePair.second)});
      }

      for(const auto& protoStage : protoMSS.stages()) {
        int doMethodPos = 0;
        int stageID = protoStage.stageid();

        IIRMSS->insertChild(make_unique<iir::Stage>(*target, stageID));
        const auto& IIRStage = IIRMSS->getChild(stagePos++);

        for(const auto& protoDoMethod : protoStage.domethods()) {
          (IIRStage)->insertChild(
              make_unique<iir::DoMethod>(*makeInterval(protoDoMethod.interval()), *target));

          auto& IIRDoMethod = (IIRStage)->getChild(doMethodPos++);
          (IIRDoMethod)->setID(protoDoMethod.domethodid());

          for(const auto& protoStmtAccessPair : protoDoMethod.stmtaccesspairs()) {
            auto stmt = makeStmt(protoStmtAccessPair.aststmt());
            auto statement = std::make_shared<Statement>(stmt, nullptr);

            std::shared_ptr<iir::Accesses> callerAccesses = std::make_shared<iir::Accesses>();
            for(auto writeAccess : protoStmtAccessPair.calleraccesses().writeaccess()) {
              callerAccesses->addWriteExtent(writeAccess.first, makeExtents(&writeAccess.second));
            }
            for(auto readAccess : protoStmtAccessPair.calleraccesses().readaccess()) {
              callerAccesses->addReadExtent(readAccess.first, makeExtents(&readAccess.second));
            }
            auto insertee = make_unique<iir::StatementAccessesPair>(statement);
            insertee->setCallerAccesses(callerAccesses);
            //            insertee->setCalleeAccesses(calleeAccesses);
            (IIRDoMethod)->insertChild(std::move(insertee));
          }
        }
      }
    }
  }
}

void IIRSerializer::deserializeImpl(const std::string& str, IIRSerializer::SerializationKind kind,
                                    std::shared_ptr<iir::StencilInstantiation>& target) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  // Decode the string
  proto::iir::StencilInstantiation protoStencilInstantiation;
  switch(kind) {
  case dawn::IIRSerializer::SK_Json: {
    auto status = google::protobuf::util::JsonStringToMessage(str, &protoStencilInstantiation);
    if(!status.ok())
      throw std::runtime_error(
          dawn::format("cannot deserialize StencilInstantiation: %s", status.ToString()));
    break;
  }
  case dawn::IIRSerializer::SK_Byte: {
    if(!protoStencilInstantiation.ParseFromString(str))
      throw std::runtime_error(dawn::format("cannot deserialize StencilInstantiation: %s"));
    break;
  }
  default:
    dawn_unreachable("invalid SerializationKind");
  }

  std::shared_ptr<iir::StencilInstantiation> instantiation =
      std::make_shared<iir::StencilInstantiation>(target->getOptimizerContext());

  deserializeMetaData(instantiation, (protoStencilInstantiation.metadata()));
  deserializeIIR(instantiation, (protoStencilInstantiation.internalir()));
  instantiation->getMetaData().fileName_ = protoStencilInstantiation.filename();
  computeInitialDerivedInfo(instantiation);

  target = instantiation;
}

std::shared_ptr<iir::StencilInstantiation>
IIRSerializer::deserialize(const std::string& file, OptimizerContext* context,
                           IIRSerializer::SerializationKind kind) {
  std::ifstream ifs(file);
  if(!ifs.is_open())
    throw std::runtime_error(
        dawn::format("cannot deserialize IIR: failed to open file \"%s\"", file));

  std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  std::shared_ptr<iir::StencilInstantiation> returnvalue =
      std::make_shared<iir::StencilInstantiation>(context);
  deserializeImpl(str, kind, returnvalue);
  return returnvalue;
}

std::shared_ptr<iir::StencilInstantiation>
IIRSerializer::deserializeFromString(const std::string& str, OptimizerContext* context,
                                     IIRSerializer::SerializationKind kind) {
  std::shared_ptr<iir::StencilInstantiation> returnvalue =
      std::make_shared<iir::StencilInstantiation>(context);
  deserializeImpl(str, kind, returnvalue);
  return returnvalue;
}

void dawn::IIRSerializer::serialize(const std::string& file,
                                    const std::shared_ptr<iir::StencilInstantiation> instantiation,
                                    dawn::IIRSerializer::SerializationKind kind) {
  std::ofstream ofs(file);
  if(!ofs.is_open())
    throw std::runtime_error(format("cannot serialize SIR: failed to open file \"%s\"", file));

  auto str = serializeImpl(instantiation, kind);
  std::copy(str.begin(), str.end(), std::ostreambuf_iterator<char>(ofs));
}

std::string dawn::IIRSerializer::serializeToString(
    const std::shared_ptr<iir::StencilInstantiation> instantiation,
    dawn::IIRSerializer::SerializationKind kind) {
  return serializeImpl(instantiation, kind);
}

} // namespace dawn
