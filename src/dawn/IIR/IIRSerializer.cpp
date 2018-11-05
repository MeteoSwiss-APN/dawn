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

#include "dawn/IIR/IIRSerializer.h"
#include "dawn/IIR/IIR.pb.h"
#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/ASTSerialier.h"
#include <fstream>
#include <google/protobuf/util/json_util.h>
#include <list>
#include <stack>
#include <tuple>
#include <utility>

using namespace dawn;

// void setAccesses(proto::iir::Acesses* protoAcesses, const std::shared_ptr<iir::Accesses>&
// accesses);

// std::shared_ptr<dawn::Statement>
// makeStatement(const proto::iir::StencilDescStatement* protoStatement);

// void serializeStmtAccessPair(proto::iir::StatementAcessPair* protoStmtAccessPair,
//                             const std::unique_ptr<iir::StatementAccessesPair>& stmtAccessPair);
static void setAccesses(proto::iir::Acesses* protoAcesses,
                        const std::shared_ptr<iir::Accesses>& accesses) {
  auto protoReadAccesses = protoAcesses->mutable_readaccess();
  for(auto IDExtentsPair : accesses->getReadAccesses()) {
    proto::iir::Extents protoExtents;
    iir::Extents e = IDExtentsPair.second;
    for(auto extent : e.getExtents()) {
      auto protoExtent = protoExtents.add_extents();
      protoExtent->set_minus(extent.Minus);
      protoExtent->set_plus(extent.Plus);
    }
    protoReadAccesses->insert({IDExtentsPair.first, protoExtents});
  }

  auto protoWriteAccesses = protoAcesses->mutable_writeaccess();
  for(auto IDExtentsPair : accesses->getWriteAccesses()) {
    proto::iir::Extents protoExtents;
    iir::Extents e = IDExtentsPair.second;
    for(auto extent : e.getExtents()) {
      auto protoExtent = protoExtents.add_extents();
      protoExtent->set_minus(extent.Minus);
      protoExtent->set_plus(extent.Plus);
    }
    protoWriteAccesses->insert({IDExtentsPair.first, protoExtents});
  }
}

static std::shared_ptr<dawn::Statement>
makeStatement(const proto::iir::StencilDescStatement* protoStatement) {
  auto stmt = makeStmt(protoStatement->stmt());
  // WITTODO: add stack trace here
  return std::make_shared<Statement>(stmt, nullptr);
}

static void
serializeStmtAccessPair(proto::iir::StatementAcessPair* protoStmtAccessPair,
                        const std::unique_ptr<iir::StatementAccessesPair>& stmtAccessPair) {
  // serialize the statement
  ProtoStmtBuilder builder(protoStmtAccessPair->mutable_statement()->mutable_aststmt());
  stmtAccessPair->getStatement()->ASTStmt->accept(builder);

  // TODO: I don't think this is actually needed...
  // check if caller / callee acesses are initialized, and if so, fill them
  if(stmtAccessPair->getCallerAccesses()) {
    setAccesses(protoStmtAccessPair->mutable_calleraccesses(), stmtAccessPair->getCallerAccesses());
  }
  if(stmtAccessPair->getCalleeAccesses()) {
    setAccesses(protoStmtAccessPair->mutable_calleeaccesses(), stmtAccessPair->getCalleeAccesses());
  }
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
  for(const auto& exprToAccessIDPair : metaData.ExprToAccessIDMap_) {
    auto protoExprToAccessID = protoMetaData->add_exprtoaccessid();
    //    ProtoStmtBuilder builder(protoExprToAccessID->mutable_expr());
    //    exprToAccessIDPair.first->accept(builder);
    protoExprToAccessID->set_ids(exprToAccessIDPair.second);
  }
  // Filling Field: repeated StmtIDPair StmtToAccessID = 3;
  for(const auto& stmtToAccessIDPair : metaData.StmtToAccessIDMap_) {
    auto protoStmtToAccessID = protoMetaData->add_stmttoaccessid();
    ProtoStmtBuilder builder(protoStmtToAccessID->mutable_stmt());
    stmtToAccessIDPair.first->accept(builder);
    protoStmtToAccessID->set_ids(stmtToAccessIDPair.second);
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
  auto protoVariableVersionMap = *protoVariableVersions->mutable_veriableversionmap();
  auto protoVersionIDtoOriginalIDMap = *protoVariableVersions->mutable_versionidtooriginalid();

  auto variableVersions = metaData.variableVersions_;
  for(int versionedID : variableVersions.getVersionIDs()) {
    protoVariableVersions->add_versionids(versionedID);
  }
  for(auto& IDtoVectorOfVersionsPair : variableVersions.variableVersionsMap_) {
    proto::iir::AllVersionedFields protoFieldVersions;
    for(int id : *(IDtoVectorOfVersionsPair.second)) {
      protoFieldVersions.add_allids(id);
    }
    protoVariableVersionMap.insert({IDtoVectorOfVersionsPair.first, protoFieldVersions});
  }
  for(auto& VersionedIDToOriginalID : variableVersions.versionToOriginalVersionMap_) {
    protoVersionIDtoOriginalIDMap.insert(
        {VersionedIDToOriginalID.first, VersionedIDToOriginalID.second});
  }
  // Filling Field: repeated StencilDescStatement stencilDescStatements = 10;
  for(const auto& stencilDescStmt : metaData.stencilDescStatements_) {
    auto protoStmt = protoMetaData->add_stencildescstatements();
    ProtoStmtBuilder builder(protoStmt->mutable_stmt());
    stencilDescStmt->ASTStmt->accept(builder);
    if(stencilDescStmt->StackTrace)
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
  // Filling Field: map<int32, dawn.proto.statements.StencilCallDeclStmt> IDToStencilCall = 11;

  // Filling Field:
  // map<string, dawn.proto.statements.BoundaryConditionDeclStmt> FieldnameToBoundaryCondition = 12;

  // Filling Field: map<int32, Array3i> fieldIDtoLegalDimensions = 13;
  auto protoInitializedDimensionsMap = *protoMetaData->mutable_fieldidtolegaldimensions();
  for(auto IDToLegalDimension : metaData.fieldIDToInitializedDimensionsMap_) {
    proto::iir::Array3i array;
    array.set_int1(IDToLegalDimension.second[0]);
    array.set_int2(IDToLegalDimension.second[1]);
    array.set_int3(IDToLegalDimension.second[2]);
    protoInitializedDimensionsMap.insert({IDToLegalDimension.first, array});
  }
  // Filling Field: map<string, GlobalValueAndType> GlobalVariableToValue = 14;
  auto protoGlobalVariableMap = *protoMetaData->mutable_globalvariabletovalue();
  for(auto& globalToValue : metaData.globalVariableMap_) {
    proto::iir::GlobalValueAndType protoGlobalToStore;
    int typekind = -1;
    double value;
    bool valueIsSet = false;
    switch(globalToValue.second->getType()) {
    case sir::Value::Boolean:
      if(!globalToValue.second->empty()) {
        value = globalToValue.second->getValue<bool>();
        valueIsSet = true;
      }
      typekind = 1;
      break;
    case sir::Value::Integer:
      if(!globalToValue.second->empty()) {
        value = globalToValue.second->getValue<int>();
        valueIsSet = true;
      }
      typekind = 2;
      break;
    case sir::Value::Double:
      if(!globalToValue.second->empty()) {
        value = globalToValue.second->getValue<double>();
        valueIsSet = true;
      }
      typekind = 3;
      break;
    default:
      dawn_unreachable("non-supported global type");
    }
    protoGlobalToStore.set_typekind(typekind);
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
  // Filling Field: string fileName = 17;
  protoMetaData->set_filename(metaData.fileName_);
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
    protoAttribute->set_attrbits(stencils->getStencilAttributes().getBits());

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
      protoMSS->set_mulitstageid(multistages->getID());

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
            //            std::cout << "serializing this stmt\n"
            //                      <<
            //                      stmtaccesspair->toString(&stencils->getStencilInstantiation())
            //                      << std::endl;
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
  using namespace dawn::proto::iir;
  proto::iir::StencilInstantiation protoStencilInstantiation;
  serializeMetaData(protoStencilInstantiation, instantiation->getMetaData());
  serializeIIR(protoStencilInstantiation, instantiation->getIIR());

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
  for(auto exprToID : protoMetaData.exprtoaccessid()) {
    //    metadata.ExprToAccessIDMap_[makeExpr(exprToID.expr())] = exprToID.ids();
  }
  for(auto stmtToID : protoMetaData.stmttoaccessid()) {
    //    metadata.StmtToAccessIDMap_[makeStmt(stmtToID.stmt())] = stmtToID.ids();
  }
  for(auto literalIDToName : protoMetaData.literalidtoname()) {
    metadata.LiteralAccessIDToNameMap_[literalIDToName.first] = literalIDToName.second;
  }
  for(int i = 0; i < protoMetaData.fieldaccessids_size(); ++i) {
    metadata.FieldAccessIDSet_.insert(protoMetaData.fieldaccessids(i));
  }
  for(int i = 0; i < protoMetaData.apifieldids_size(); ++i) {
    metadata.apiFieldIDs_.push_back(protoMetaData.apifieldids(i));
  }
  for(int i = 0; i < protoMetaData.temporaryfieldids_size(); ++i) {
    metadata.TemporaryFieldAccessIDSet_.insert(protoMetaData.temporaryfieldids(i));
  }
  for(int i = 0; i < protoMetaData.globalvariableids_size(); ++i) {
    metadata.GlobalVariableAccessIDSet_.insert(protoMetaData.globalvariableids(i));
  }
  //
  // Variable Versions
  //

  // WITTODO: this does not work due to vdecl are not allowed
  //  for(auto stencilDescStmt : protoMetaData.stencildescstatements()) {
  //    metadata.stencilDescStatements_.push_back(makeStatement(&stencilDescStmt));
  //  }

  //  for(auto IDToCall : protoMetaData.idtostencilcall()) {
  //    metadata.IDToStencilCallMap_[IDToCall.first] = makeStmt((IDToCall.second));
  //  }
  //  for(auto FieldnameToBC : protoMetaData.fieldnametoboundarycondition()) {
  //    metadata.FieldnameToBoundaryConditionMap_[FieldnameToBC.first] =
  //    makeStmt((FieldnameToBC.second));
  //  }
  for(auto fieldIDInitializedDims : protoMetaData.fieldidtolegaldimensions()) {
    Array3i dims{fieldIDInitializedDims.second.int1(), fieldIDInitializedDims.second.int2(),
                 fieldIDInitializedDims.second.int3()};
    metadata.fieldIDToInitializedDimensionsMap_[fieldIDInitializedDims.first] = dims;
  }
  for(auto GlobalToValue : protoMetaData.globalvariabletovalue()) {
    std::shared_ptr<sir::Value> value = std::make_shared<sir::Value>();
    switch(GlobalToValue.second.typekind()) {
    case 1:
      value->setType(sir::Value::Boolean);
      break;
    case 2:
      value->setType(sir::Value::Integer);
      break;
    case 3:
      value->setType(sir::Value::Double);
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

  metadata.fileName_ = protoMetaData.filename();
}

void IIRSerializer::deserializeIIR(std::shared_ptr<iir::StencilInstantiation>& target,
                                   const proto::iir::IIR& protoIIR) {
  for(const auto& protoStencils : protoIIR.stencils()) {
    std::cout << "And now we deserialize the stencil" << std::endl;
    sir::Attr attributes;
    attributes.setBits(protoStencils.attr().attrbits());
    //    target->getIIR()->insertChild(
    //        make_unique<iir::Stencil>(*target, attributes, protoStencils.stencilid()));

    for(const auto& protoMSS : protoStencils.multistages()) {
      std::cout << "And now we deserialize the mss" << std::endl;
      for(const auto& protoStage : protoMSS.stages()) {
        std::cout << "And now we deserialize the stage" << std::endl;
        for(const auto& protoDoMethod : protoStage.domethods()) {
          std::cout << "And now we deserialize the domethod" << std::endl;
          for(const auto& protoStmtAccessPair : protoDoMethod.stmtaccesspairs()) {
            std::cout << "And now we deserialize the stmtaccesspair" << std::endl;
            auto stmt = makeStmt(protoStmtAccessPair.statement().aststmt());
            std::cout << stmt << std::endl;
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
      throw std::runtime_error(dawn::format("cannot deserialize StencilInstantiation: %s")); //,
    // ProtobufLogger::getInstance().getErrorMessagesAndReset()));
    break;
  }
  default:
    dawn_unreachable("invalid SerializationKind");
  }

  std::shared_ptr<iir::StencilInstantiation> instantiation =
      std::make_shared<iir::StencilInstantiation>(target->getOptimizerContext());

  deserializeMetaData(instantiation, (protoStencilInstantiation.metadata()));
  deserializeIIR(instantiation, (protoStencilInstantiation.internalir()));

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
