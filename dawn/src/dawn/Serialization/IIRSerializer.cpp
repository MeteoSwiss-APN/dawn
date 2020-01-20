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
//===----------------------------------------TypeKind--------------------------------------------------===//
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/AST/ASTStmt.h"
#include "dawn/IIR/ASTStmt.h"
#include "dawn/IIR/ASTVisitor.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/ASTSerializer.h"
#include "dawn/Support/Assert.h"
#include <fstream>
#include <google/protobuf/util/json_util.h>
#include <memory>
#include <optional>

namespace dawn {
static void setCache(proto::iir::Cache* protoCache, const iir::Cache& cache) {
  protoCache->set_accessid(cache.getCachedFieldAccessID());
  switch(cache.getIOPolicy()) {
  case iir::Cache::IOPolicy::bpfill:
    protoCache->set_policy(proto::iir::Cache_CachePolicy_CP_BPFill);
    break;
  case iir::Cache::IOPolicy::epflush:
    protoCache->set_policy(proto::iir::Cache_CachePolicy_CP_EPFlush);
    break;
  case iir::Cache::IOPolicy::fill:
    protoCache->set_policy(proto::iir::Cache_CachePolicy_CP_Fill);
    break;
  case iir::Cache::IOPolicy::fill_and_flush:
    protoCache->set_policy(proto::iir::Cache_CachePolicy_CP_FillFlush);
    break;
  case iir::Cache::IOPolicy::flush:
    protoCache->set_policy(proto::iir::Cache_CachePolicy_CP_Flush);
    break;
  case iir::Cache::IOPolicy::local:
    protoCache->set_policy(proto::iir::Cache_CachePolicy_CP_Local);
    break;
  case iir::Cache::IOPolicy::unknown:
    protoCache->set_policy(proto::iir::Cache_CachePolicy_CP_Unknown);
    break;
  default:
    dawn_unreachable("unknown cache policy");
  }
  switch(cache.getType()) {
  case iir::Cache::CacheType::bypass:
    protoCache->set_type(proto::iir::Cache_CacheType_CT_Bypass);
    break;
  case iir::Cache::CacheType::IJ:
    protoCache->set_type(proto::iir::Cache_CacheType_CT_IJ);
    break;
  case iir::Cache::CacheType::IJK:
    protoCache->set_type(proto::iir::Cache_CacheType_CT_IJK);
    break;
  case iir::Cache::CacheType::K:
    protoCache->set_type(proto::iir::Cache_CacheType_CT_K);
    break;
  default:
    dawn_unreachable("unknown cache type");
  }
  if(cache.getInterval()) {
    auto sirInterval = cache.getInterval()->asSIRInterval();
    setInterval(protoCache->mutable_interval(), &sirInterval);
  }
  if(cache.getEnclosingAccessedInterval()) {
    auto sirInterval = cache.getEnclosingAccessedInterval()->asSIRInterval();
    setInterval(protoCache->mutable_enclosingaccessinterval(), &sirInterval);
  }
  if(cache.getWindow()) {
    protoCache->mutable_cachewindow()->set_minus(cache.getWindow()->m_m);
    protoCache->mutable_cachewindow()->set_plus(cache.getWindow()->m_p);
  }
}

static iir::Cache makeCache(const proto::iir::Cache* protoCache) {
  iir::Cache::CacheType cacheType;
  iir::Cache::IOPolicy cachePolicy;
  std::optional<iir::Interval> interval;
  std::optional<iir::Interval> enclosingInverval;
  std::optional<iir::Cache::window> cacheWindow;
  int ID = protoCache->accessid();
  switch(protoCache->type()) {
  case proto::iir::Cache_CacheType_CT_Bypass:
    cacheType = iir::Cache::CacheType::bypass;
    break;
  case proto::iir::Cache_CacheType_CT_IJ:
    cacheType = iir::Cache::CacheType::IJ;
    break;
  case proto::iir::Cache_CacheType_CT_IJK:
    cacheType = iir::Cache::CacheType::IJK;
    break;
  case proto::iir::Cache_CacheType_CT_K:
    cacheType = iir::Cache::CacheType::K;
    break;
  default:
    dawn_unreachable("unknow cache type");
  }
  switch(protoCache->policy()) {
  case proto::iir::Cache_CachePolicy_CP_BPFill:
    cachePolicy = iir::Cache::IOPolicy::bpfill;
    break;
  case proto::iir::Cache_CachePolicy_CP_EPFlush:
    cachePolicy = iir::Cache::IOPolicy::epflush;
    break;
  case proto::iir::Cache_CachePolicy_CP_Fill:
    cachePolicy = iir::Cache::IOPolicy::fill;
    break;
  case proto::iir::Cache_CachePolicy_CP_FillFlush:
    cachePolicy = iir::Cache::IOPolicy::fill_and_flush;
    break;
  case proto::iir::Cache_CachePolicy_CP_Flush:
    cachePolicy = iir::Cache::IOPolicy::flush;
    break;
  case proto::iir::Cache_CachePolicy_CP_Local:
    cachePolicy = iir::Cache::IOPolicy::local;
    break;
  case proto::iir::Cache_CachePolicy_CP_Unknown:
    cachePolicy = iir::Cache::IOPolicy::unknown;
    break;
  default:
    dawn_unreachable("unknown cache policy");
  }
  if(protoCache->has_interval()) {
    interval = std::make_optional(*makeInterval(protoCache->interval()));
  }
  if(protoCache->has_enclosingaccessinterval()) {
    enclosingInverval = std::make_optional(*makeInterval(protoCache->enclosingaccessinterval()));
  }
  if(protoCache->has_cachewindow()) {
    cacheWindow = std::make_optional(
        iir::Cache::window{protoCache->cachewindow().plus(), protoCache->cachewindow().minus()});
  }

  return iir::Cache(cacheType, cachePolicy, ID, interval, enclosingInverval, cacheWindow);
}

static void computeInitialDerivedInfo(const std::shared_ptr<iir::StencilInstantiation>& target) {
  for(const auto& leaf : iterateIIROver<iir::DoMethod>(*target->getIIR())) {
    leaf->update(iir::NodeUpdateType::levelAndTreeAbove);
  }
}

void IIRSerializer::serializeMetaData(proto::iir::StencilInstantiation& target,
                                      iir::StencilMetaInformation& metaData) {
  auto protoMetaData = target.mutable_metadata();

  // Filling Field: map<int32, string> AccessIDToName = 1;
  auto& protoAccessIDtoNameMap = *protoMetaData->mutable_accessidtoname();
  for(const auto& accessIDtoNamePair : metaData.getAccessIDToNameMap()) {
    protoAccessIDtoNameMap.insert({accessIDtoNamePair.first, accessIDtoNamePair.second});
  }
  // Filling Field: repeated AccessIDType = 2;
  auto& protoAccessIDType = *protoMetaData->mutable_accessidtotype();
  for(const auto& accessIDTypePair : metaData.fieldAccessMetadata_.accessIDType_) {
    protoAccessIDType.insert({accessIDTypePair.first, (int)accessIDTypePair.second});
  }
  // Filling Field: map<int32, string> LiteralIDToName = 3;
  auto& protoLiteralIDToNameMap = *protoMetaData->mutable_literalidtoname();
  for(const auto& literalIDtoNamePair : metaData.fieldAccessMetadata_.LiteralAccessIDToNameMap_) {
    protoLiteralIDToNameMap.insert({literalIDtoNamePair.first, literalIDtoNamePair.second});
  }
  // Filling Field: repeated int32 FieldAccessIDs = 4;
  for(int fieldAccessID : metaData.fieldAccessMetadata_.FieldAccessIDSet_) {
    protoMetaData->add_fieldaccessids(fieldAccessID);
  }
  // Filling Field: repeated int32 APIFieldIDs = 5;
  for(int apifieldID : metaData.fieldAccessMetadata_.apiFieldIDs_) {
    protoMetaData->add_apifieldids(apifieldID);
  }
  // Filling Field: repeated int32 TemporaryFieldIDs = 6;
  for(int temporaryFieldID : metaData.fieldAccessMetadata_.TemporaryFieldAccessIDSet_) {
    protoMetaData->add_temporaryfieldids(temporaryFieldID);
  }
  // Filling Field: repeated int32 GlobalVariableIDs = 7;
  for(int globalVariableID : metaData.fieldAccessMetadata_.GlobalVariableAccessIDSet_) {
    protoMetaData->add_globalvariableids(globalVariableID);
  }

  // Filling Field: VariableVersions versionedFields = 8;
  auto protoVariableVersions = protoMetaData->mutable_versionedfields();
  auto& protoVariableVersionMap = *protoVariableVersions->mutable_variableversionmap();
  auto variableVersions = metaData.fieldAccessMetadata_.variableVersions_;
  for(const auto& IDtoVectorOfVersionsPair : variableVersions.getvariableVersionsMap()) {
    proto::iir::AllVersionedFields protoFieldVersions;
    for(int id : *(IDtoVectorOfVersionsPair.second)) {
      protoFieldVersions.add_allids(id);
    }
    protoVariableVersionMap.insert({IDtoVectorOfVersionsPair.first, protoFieldVersions});
  }

  // Filling Field:
  // map<string, dawn.proto.statements.BoundaryConditionDeclStmt> FieldnameToBoundaryCondition = 9;
  auto& protoFieldNameToBC = *protoMetaData->mutable_fieldnametoboundarycondition();
  for(auto fieldNameToBC : metaData.fieldnameToBoundaryConditionMap_) {
    proto::statements::Stmt protoStencilCall;
    ProtoStmtBuilder builder(&protoStencilCall, ast::StmtData::IIR_DATA_TYPE);
    fieldNameToBC.second->accept(builder);
    protoFieldNameToBC.insert({fieldNameToBC.first, protoStencilCall});
  }

  // Filling Field: map<int32, Array3i> fieldIDtoLegalDimensions = 10;
  auto& protoInitializedDimensionsMap = *protoMetaData->mutable_fieldidtolegaldimensions();
  for(auto IDToLegalDimension : metaData.fieldIDToInitializedDimensionsMap_) {
    auto const& cartDimensions =
        dawn::sir::dimension_cast<dawn::sir::CartesianFieldDimension const&>(
            IDToLegalDimension.second);
    proto::iir::Array3i array;
    array.set_int1(cartDimensions.I());
    array.set_int2(cartDimensions.J());
    array.set_int3(cartDimensions.K());
    protoInitializedDimensionsMap.insert({IDToLegalDimension.first, array});
  }

  // Filling Field: map<int32, dawn.proto.statements.StencilCallDeclStmt> IDToStencilCall = 11;
  auto& protoIDToStencilCallMap = *protoMetaData->mutable_idtostencilcall();
  for(auto IDToStencilCall : metaData.getStencilIDToStencilCallMap().getDirectMap()) {
    proto::statements::Stmt protoStencilCall;
    ProtoStmtBuilder builder(&protoStencilCall, ast::StmtData::IIR_DATA_TYPE);
    IDToStencilCall.second->accept(builder);
    protoIDToStencilCallMap.insert({IDToStencilCall.first, protoStencilCall});
  }

  // Filling Field: map<int32, Extents> boundaryCallToExtent = 12;
  auto& protoBoundaryCallToExtent = *protoMetaData->mutable_boundarycalltoextent();
  for(auto boundaryCallToExtent : metaData.boundaryConditionToExtentsMap_)
    protoBoundaryCallToExtent.insert(
        {boundaryCallToExtent.first->getID(), makeProtoExtents(boundaryCallToExtent.second)});

  // Filling Field: dawn.proto.statements.SourceLocation stencilLocation = 13;
  for(auto allocatedFieldID : metaData.fieldAccessMetadata_.AllocatedFieldAccessIDSet_) {
    protoMetaData->add_allocatedfieldids(allocatedFieldID);
  }

  // Filling Field: dawn.proto.statements.SourceLocation stencilLocation = 14;
  auto protoStencilLoc = protoMetaData->mutable_stencillocation();
  protoStencilLoc->set_column(metaData.stencilLocation_.Column);
  protoStencilLoc->set_line(metaData.stencilLocation_.Line);

  // Filling Field: string stencilMName = 15;
  protoMetaData->set_stencilname(metaData.stencilName_);
}

void IIRSerializer::serializeIIR(proto::iir::StencilInstantiation& target,
                                 const std::unique_ptr<iir::IIR>& iir,
                                 std::set<std::string> const& usedBC) {
  auto protoIIR = target.mutable_internalir();

  switch(iir->getGridType()) {
  case ast::GridType::Cartesian:
    protoIIR->set_gridtype(proto::enums::GridType::Cartesian);
    break;
  case ast::GridType::Triangular:
    protoIIR->set_gridtype(proto::enums::GridType::Triangular);
    break;
  default:
    dawn_unreachable("invalid grid type");
  }

  auto& protoGlobalVariableMap = *protoIIR->mutable_globalvariabletovalue();
  for(auto& globalToValue : iir->getGlobalVariableMap()) {
    proto::iir::GlobalValueAndType protoGlobalToStore;
    bool valueIsSet = false;

    switch(globalToValue.second.getType()) {
    case sir::Value::Kind::Boolean:
      if(globalToValue.second.has_value()) {
        protoGlobalToStore.set_value(globalToValue.second.getValue<bool>());
        valueIsSet = true;
      }
      protoGlobalToStore.set_type(proto::iir::GlobalValueAndType_TypeKind_Boolean);
      break;
    case sir::Value::Kind::Integer:
      if(globalToValue.second.has_value()) {
        protoGlobalToStore.set_value(globalToValue.second.getValue<int>());
        valueIsSet = true;
      }
      protoGlobalToStore.set_type(proto::iir::GlobalValueAndType_TypeKind_Integer);
      break;
    case sir::Value::Kind::Double:
      if(globalToValue.second.has_value()) {
        protoGlobalToStore.set_value(globalToValue.second.getValue<double>());
        valueIsSet = true;
      }
      protoGlobalToStore.set_type(proto::iir::GlobalValueAndType_TypeKind_Double);
      break;
    default:
      dawn_unreachable("non-supported global type");
    }

    protoGlobalToStore.set_valueisset(valueIsSet);
    protoGlobalVariableMap.insert({globalToValue.first, protoGlobalToStore});
  }

  // Get all the stencils
  for(const auto& stencils : iir->getChildren()) {
    // creation of a new protobuf stencil
    auto protoStencil = protoIIR->add_stencils();
    // Information other than the children
    protoStencil->set_stencilid(stencils->getStencilID());
    auto protoAttribute = protoStencil->mutable_attr();
    if(stencils->getStencilAttributes().has(sir::Attr::Kind::MergeDoMethods)) {
      protoAttribute->add_attributes(
          proto::iir::Attributes::StencilAttributes::Attributes_StencilAttributes_MergeDoMethods);
    }
    if(stencils->getStencilAttributes().has(sir::Attr::Kind::MergeStages)) {
      protoAttribute->add_attributes(
          proto::iir::Attributes::StencilAttributes::Attributes_StencilAttributes_MergeStages);
    }
    if(stencils->getStencilAttributes().has(sir::Attr::Kind::MergeTemporaries)) {
      protoAttribute->add_attributes(
          proto::iir::Attributes::StencilAttributes::Attributes_StencilAttributes_MergeTemporaries);
    }
    if(stencils->getStencilAttributes().has(sir::Attr::Kind::NoCodeGen)) {
      protoAttribute->add_attributes(
          proto::iir::Attributes::StencilAttributes::Attributes_StencilAttributes_NoCodeGen);
    }
    if(stencils->getStencilAttributes().has(sir::Attr::Kind::UseKCaches)) {
      protoAttribute->add_attributes(
          proto::iir::Attributes::StencilAttributes::Attributes_StencilAttributes_UseKCaches);
    }

    // adding it's children
    for(const auto& multistages : stencils->getChildren()) {
      // creation of a protobuf multistage
      auto protoMSS = protoStencil->add_multistages();
      // Information other than the children
      if(multistages->getLoopOrder() == dawn::iir::LoopOrderKind::Forward) {
        protoMSS->set_looporder(proto::iir::MultiStage::Forward);
      } else if(multistages->getLoopOrder() == dawn::iir::LoopOrderKind::Backward) {
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

          auto protoStmt = protoDoMethod->mutable_ast();
          ProtoStmtBuilder builder(protoStmt, ast::StmtData::IIR_DATA_TYPE);
          auto ptr = std::make_shared<ast::BlockStmt>(
              domethod->getAST()); // TODO takes a copy to allow using shared_from_this()
          ptr->accept(builder);
        }
      }
    }
  }

  // Filling Field: repeated StencilDescStatement stencilDescStatements = 10;
  for(const auto& stencilDescStmt : iir->getControlFlowDescriptor().getStatements()) {
    auto protoStmt = protoIIR->add_controlflowstatements();
    ProtoStmtBuilder builder(protoStmt, ast::StmtData::IIR_DATA_TYPE);
    stencilDescStmt->accept(builder);
    if(stencilDescStmt->getData<iir::IIRStmtData>().StackTrace)
      DAWN_ASSERT_MSG(stencilDescStmt->getData<iir::IIRStmtData>().StackTrace->empty(),
                      "there should be no stack trace if inlining worked");
  }
  for(const auto& sf : iir->getStencilFunctions()) {
    if(usedBC.count(sf->Name) > 0) {
      auto protoBC = protoIIR->add_boundaryconditions();
      protoBC->set_name(sf->Name);
      for(auto& arg : sf->Args) {
        DAWN_ASSERT(arg->Kind == sir::StencilFunctionArg::ArgumentKind::Field);
        protoBC->add_args(arg->Name);
      }

      DAWN_ASSERT(sf->Asts.size() == 1);
      ProtoStmtBuilder builder(protoBC->mutable_aststmt(), ast::StmtData::IIR_DATA_TYPE);
      sf->Asts[0]->accept(builder);
    }
  }
}

std::string
IIRSerializer::serializeImpl(const std::shared_ptr<iir::StencilInstantiation>& instantiation,
                             Format kind) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
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

  using namespace dawn::proto::iir;
  proto::iir::StencilInstantiation protoStencilInstantiation;
  serializeMetaData(protoStencilInstantiation, instantiation->getMetaData());
  auto& fieldNameToBCMap = instantiation->getMetaData().getFieldNameToBCMap();
  std::set<std::string> usedBC;
  std::transform(
      fieldNameToBCMap.begin(), fieldNameToBCMap.end(), std::inserter(usedBC, usedBC.end()),
      [](std::pair<std::string, std::shared_ptr<iir::BoundaryConditionDeclStmt>> const& bc) {
        return bc.second->getFunctor();
      });
  serializeIIR(protoStencilInstantiation, instantiation->getIIR(), usedBC);
  protoStencilInstantiation.set_filename(instantiation->getMetaData().fileName_);

  // Encode the message
  std::string str;
  switch(kind) {
  case Format::Json: {
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
  case Format::Byte: {
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
    metadata.addAccessIDNamePair(IDtoName.first, IDtoName.second);
  }

  for(auto accessIDTypePair : protoMetaData.accessidtotype()) {
    metadata.fieldAccessMetadata_.accessIDType_.emplace(
        accessIDTypePair.first, (iir::FieldAccessType)accessIDTypePair.second);
  }

  for(auto literalIDToName : protoMetaData.literalidtoname()) {
    metadata.fieldAccessMetadata_.LiteralAccessIDToNameMap_[literalIDToName.first] =
        literalIDToName.second;
  }
  for(auto fieldaccessID : protoMetaData.fieldaccessids()) {
    metadata.fieldAccessMetadata_.FieldAccessIDSet_.insert(fieldaccessID);
  }
  for(auto ApiFieldID : protoMetaData.apifieldids()) {
    metadata.fieldAccessMetadata_.apiFieldIDs_.push_back(ApiFieldID);
  }
  for(auto temporaryFieldID : protoMetaData.temporaryfieldids()) {
    metadata.fieldAccessMetadata_.TemporaryFieldAccessIDSet_.insert(temporaryFieldID);
  }
  for(auto globalVariableID : protoMetaData.globalvariableids()) {
    metadata.fieldAccessMetadata_.GlobalVariableAccessIDSet_.insert(globalVariableID);
  }
  for(auto allocatedFieldID : protoMetaData.allocatedfieldids()) {
    metadata.fieldAccessMetadata_.AllocatedFieldAccessIDSet_.insert(allocatedFieldID);
  }

  for(auto variableVersionMap : protoMetaData.versionedfields().variableversionmap()) {
    for(auto versionedID : variableVersionMap.second.allids()) {
      metadata.addFieldVersionIDPair(variableVersionMap.first, versionedID);
    }
  }

  struct DeclStmtFinder : public iir::ASTVisitorForwarding {
    void visit(const std::shared_ptr<iir::StencilCallDeclStmt>& stmt) override {
      stencilCallDecl.insert(std::make_pair(stmt->getID(), stmt));
      ASTVisitorForwarding::visit(stmt);
    }
    void visit(const std::shared_ptr<iir::BoundaryConditionDeclStmt>& stmt) override {
      boundaryConditionDecl.insert(std::make_pair(stmt->getID(), stmt));
      ASTVisitorForwarding::visit(stmt);
    }
    std::map<int, std::shared_ptr<iir::StencilCallDeclStmt>> stencilCallDecl;
    std::map<int, std::shared_ptr<iir::BoundaryConditionDeclStmt>> boundaryConditionDecl;
  };
  DeclStmtFinder declStmtFinder;
  for(auto& stmt : target->getIIR()->getControlFlowDescriptor().getStatements())
    stmt->accept(declStmtFinder);

  for(auto IDToCall : protoMetaData.idtostencilcall()) {
    auto call = IDToCall.second;
    std::shared_ptr<ast::StencilCall> astStencilCall = std::make_shared<ast::StencilCall>(
        call.stencil_call_decl_stmt().stencil_call().callee(),
        makeLocation(call.stencil_call_decl_stmt().stencil_call()));
    for(const auto& protoFieldName : call.stencil_call_decl_stmt().stencil_call().arguments()) {
      astStencilCall->Args.push_back(protoFieldName);
    }

    auto stmt = declStmtFinder.stencilCallDecl[call.stencil_call_decl_stmt().id()];
    stmt->setID(call.stencil_call_decl_stmt().id());
    metadata.addStencilCallStmt(stmt, IDToCall.first);
  }

  for(auto FieldnameToBC : protoMetaData.fieldnametoboundarycondition()) {
    auto foundDecl = declStmtFinder.boundaryConditionDecl.find(
        FieldnameToBC.second.boundary_condition_decl_stmt().id());

    metadata.fieldnameToBoundaryConditionMap_[FieldnameToBC.first] =
        foundDecl != declStmtFinder.boundaryConditionDecl.end()
            ? foundDecl->second
            : dyn_pointer_cast<iir::BoundaryConditionDeclStmt>(
                  makeStmt(FieldnameToBC.second, ast::StmtData::IIR_DATA_TYPE));
  }

  for(auto fieldIDInitializedDims : protoMetaData.fieldidtolegaldimensions()) {
    metadata.fieldIDToInitializedDimensionsMap_.emplace(
        fieldIDInitializedDims.first,
        sir::FieldDimension(ast::cartesian, {fieldIDInitializedDims.second.int1() == 1,
                                             fieldIDInitializedDims.second.int2() == 1,
                                             fieldIDInitializedDims.second.int3() == 1}));
  }

  for(auto boundaryCallToExtent : protoMetaData.boundarycalltoextent())
    metadata.boundaryConditionToExtentsMap_.insert(
        std::make_pair(declStmtFinder.boundaryConditionDecl.at(boundaryCallToExtent.first),
                       makeExtents(&boundaryCallToExtent.second)));

  metadata.stencilLocation_.Column = protoMetaData.stencillocation().column();
  metadata.stencilLocation_.Line = protoMetaData.stencillocation().line();

  metadata.stencilName_ = protoMetaData.stencilname();
}

void IIRSerializer::deserializeIIR(std::shared_ptr<iir::StencilInstantiation>& target,
                                   const proto::iir::IIR& protoIIR) {
  for(auto GlobalToValue : protoIIR.globalvariabletovalue()) {
    std::shared_ptr<sir::Global> value;
    switch(GlobalToValue.second.type()) {
    case proto::iir::GlobalValueAndType_TypeKind_Boolean:
      if(GlobalToValue.second.valueisset()) {
        value = std::make_shared<sir::Global>(sir::Value::Kind::Boolean);
      } else {
        value = std::make_shared<sir::Global>(GlobalToValue.second.value());
      }
      break;
    case proto::iir::GlobalValueAndType_TypeKind_Integer:
      if(GlobalToValue.second.valueisset()) {
        value = std::make_shared<sir::Global>(sir::Value::Kind::Integer);
      } else {
        // the explicit cast is needed since in this case GlobalToValue.second.value()
        // may hold a double constant because of trailing dot in the IIR (e.g. 12.)
        value = std::make_shared<sir::Global>((int)GlobalToValue.second.value());
      }
      break;
    case proto::iir::GlobalValueAndType_TypeKind_Double:
      if(GlobalToValue.second.valueisset()) {
        value = std::make_shared<sir::Global>(sir::Value::Kind::Double);
      } else {
        value = std::make_shared<sir::Global>((double)GlobalToValue.second.value());
      }
      break;
    default:
      dawn_unreachable("unsupported type");
    }

    target->getIIR()->insertGlobalVariable(std::string(GlobalToValue.first),
                                           std::move(*value.get()));
  }

  int stencilPos = 0;
  for(const auto& protoStencils : protoIIR.stencils()) {
    int mssPos = 0;
    sir::Attr attributes;

    target->getIIR()->insertChild(std::make_unique<iir::Stencil>(target->getMetaData(), attributes,
                                                                 protoStencils.stencilid()),
                                  target->getIIR());
    const auto& IIRStencil = target->getIIR()->getChild(stencilPos++);

    for(auto attribute : protoStencils.attr().attributes()) {
      if(attribute ==
         proto::iir::Attributes::StencilAttributes::Attributes_StencilAttributes_MergeDoMethods) {
        IIRStencil->getStencilAttributes().set(sir::Attr::Kind::MergeDoMethods);
      }
      if(attribute ==
         proto::iir::Attributes::StencilAttributes::Attributes_StencilAttributes_MergeStages) {
        IIRStencil->getStencilAttributes().set(sir::Attr::Kind::MergeStages);
      }
      if(attribute ==
         proto::iir::Attributes::StencilAttributes::Attributes_StencilAttributes_MergeTemporaries) {
        IIRStencil->getStencilAttributes().set(sir::Attr::Kind::MergeTemporaries);
      }
      if(attribute ==
         proto::iir::Attributes::StencilAttributes::Attributes_StencilAttributes_NoCodeGen) {
        IIRStencil->getStencilAttributes().set(sir::Attr::Kind::NoCodeGen);
      }
      if(attribute ==
         proto::iir::Attributes::StencilAttributes::Attributes_StencilAttributes_UseKCaches) {
        IIRStencil->getStencilAttributes().set(sir::Attr::Kind::UseKCaches);
      }
    }

    for(const auto& protoMSS : protoStencils.multistages()) {
      int stagePos = 0;
      iir::LoopOrderKind looporder;
      if(protoMSS.looporder() == proto::iir::MultiStage_LoopOrder::MultiStage_LoopOrder_Backward) {
        looporder = iir::LoopOrderKind::Backward;
      }
      if(protoMSS.looporder() == proto::iir::MultiStage_LoopOrder::MultiStage_LoopOrder_Forward) {
        looporder = iir::LoopOrderKind::Forward;
      }
      if(protoMSS.looporder() == proto::iir::MultiStage_LoopOrder::MultiStage_LoopOrder_Parallel) {
        looporder = iir::LoopOrderKind::Parallel;
      }
      (IIRStencil)
          ->insertChild(std::make_unique<iir::MultiStage>(target->getMetaData(), looporder));

      const auto& IIRMSS = (IIRStencil)->getChild(mssPos++);
      IIRMSS->setID(protoMSS.multistageid());

      for(const auto& IDCachePair : protoMSS.caches()) {
        IIRMSS->getCaches().insert({IDCachePair.first, makeCache(&IDCachePair.second)});
      }

      for(const auto& protoStage : protoMSS.stages()) {
        int doMethodPos = 0;
        int stageID = protoStage.stageid();

        IIRMSS->insertChild(std::make_unique<iir::Stage>(target->getMetaData(), stageID));
        const auto& IIRStage = IIRMSS->getChild(stagePos++);

        for(const auto& protoDoMethod : protoStage.domethods()) {
          IIRStage->insertChild(std::make_unique<iir::DoMethod>(
              *makeInterval(protoDoMethod.interval()), target->getMetaData()));

          auto& IIRDoMethod = (IIRStage)->getChild(doMethodPos++);
          IIRDoMethod->setID(protoDoMethod.domethodid());

          auto ast = std::dynamic_pointer_cast<iir::BlockStmt>(
              makeStmt(protoDoMethod.ast(), ast::StmtData::IIR_DATA_TYPE));
          DAWN_ASSERT(ast);
          IIRDoMethod->setAST(std::move(*ast));
        }
      }
    }
  }
  for(auto& controlFlowStmt : protoIIR.controlflowstatements()) {
    target->getIIR()->getControlFlowDescriptor().insertStmt(
        makeStmt(controlFlowStmt, ast::StmtData::IIR_DATA_TYPE));
  }
  for(auto& boundaryCondition : protoIIR.boundaryconditions()) {
    auto stencilFunction = std::make_shared<sir::StencilFunction>();
    stencilFunction->Name = boundaryCondition.name();

    for(auto& proto_arg : boundaryCondition.args()) {
      auto new_arg = std::make_shared<sir::StencilFunctionArg>();
      new_arg->Name = proto_arg;
      new_arg->Kind = sir::StencilFunctionArg::ArgumentKind::Field;
      stencilFunction->Args.push_back(std::move(new_arg));
    }
    auto stmt = std::dynamic_pointer_cast<iir::BlockStmt>(
        makeStmt(boundaryCondition.aststmt(), ast::StmtData::IIR_DATA_TYPE));
    DAWN_ASSERT(stmt);
    stencilFunction->Asts.push_back(std::make_shared<iir::AST>(stmt));

    target->getIIR()->insertStencilFunction(stencilFunction);
  }
}

std::shared_ptr<iir::StencilInstantiation>
IIRSerializer::deserializeImpl(const std::string& str, IIRSerializer::Format kind) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  // Decode the string
  proto::iir::StencilInstantiation protoStencilInstantiation;
  switch(kind) {
  case dawn::IIRSerializer::Format::Json: {
    auto status = google::protobuf::util::JsonStringToMessage(str, &protoStencilInstantiation);
    if(!status.ok())
      throw std::runtime_error(
          dawn::format("cannot deserialize StencilInstantiation: %s", status.ToString()));
    break;
  }
  case dawn::IIRSerializer::Format::Byte: {
    if(!protoStencilInstantiation.ParseFromString(str))
      throw std::runtime_error(dawn::format("cannot deserialize StencilInstantiation: %s"));
    break;
  }
  default:
    dawn_unreachable("invalid SerializationKind");
  }

  std::shared_ptr<iir::StencilInstantiation> target;

  switch(protoStencilInstantiation.internalir().gridtype()) {
  case dawn::proto::enums::GridType::Cartesian:
    target = std::make_shared<iir::StencilInstantiation>(ast::GridType::Cartesian);
    break;
  case dawn::proto::enums::GridType::Triangular:
    target = std::make_shared<iir::StencilInstantiation>(ast::GridType::Triangular);
    break;
  default:
    dawn_unreachable("unknown grid type");
  }

  deserializeIIR(target, (protoStencilInstantiation.internalir()));
  deserializeMetaData(target, (protoStencilInstantiation.metadata()));
  target->getMetaData().fileName_ = protoStencilInstantiation.filename();
  computeInitialDerivedInfo(target);

  return target;
}

std::shared_ptr<iir::StencilInstantiation> IIRSerializer::deserialize(const std::string& file,
                                                                      IIRSerializer::Format kind) {
  std::ifstream ifs(file);
  if(!ifs.is_open())
    throw std::runtime_error(
        dawn::format("cannot deserialize IIR: failed to open file \"%s\"", file));

  std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

  return deserializeImpl(str, kind);
}

std::shared_ptr<iir::StencilInstantiation>
IIRSerializer::deserializeFromString(const std::string& str, OptimizerContext* context,
                                     IIRSerializer::Format kind) {
  return deserializeImpl(str, kind);
}

void IIRSerializer::serialize(const std::string& file,
                              const std::shared_ptr<iir::StencilInstantiation> instantiation,
                              dawn::IIRSerializer::Format kind) {
  std::ofstream ofs(file);
  if(!ofs.is_open())
    throw std::runtime_error(format("cannot serialize SIR: failed to open file \"%s\"", file));

  auto str = serializeImpl(instantiation, kind);
  std::copy(str.begin(), str.end(), std::ostreambuf_iterator<char>(ofs));
}

std::string
IIRSerializer::serializeToString(const std::shared_ptr<iir::StencilInstantiation> instantiation,
                                 dawn::IIRSerializer::Format kind) {
  return serializeImpl(instantiation, kind);
}

} // namespace dawn
