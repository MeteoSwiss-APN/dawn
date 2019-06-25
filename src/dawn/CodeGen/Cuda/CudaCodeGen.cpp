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

#include "dawn/CodeGen/Cuda/CudaCodeGen.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/CodeGen/CodeGenProperties.h"
#include "dawn/CodeGen/Cuda/ASTStencilBody.h"
#include "dawn/CodeGen/Cuda/ASTStencilDesc.h"
#include "dawn/CodeGen/Cuda/CacheProperties.h"
#include "dawn/CodeGen/Cuda/CodeGeneratorHelper.h"
#include "dawn/CodeGen/Cuda/MSCodeGen.hpp"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassInlining.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/StringUtil.h"
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

namespace dawn {
namespace codegen {
namespace cuda {

CudaCodeGen::CudaCodeGen(OptimizerContext* context) : CodeGen(context) {}

CudaCodeGen::~CudaCodeGen() {}

void CudaCodeGen::generateAllCudaKernels(
    std::stringstream& ssSW,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  for(const auto& ms : iterateIIROver<iir::MultiStage>(*(stencilInstantiation->getIIR()))) {
    DAWN_ASSERT(cachePropertyMap_.count(ms->getID()));

    MSCodeGen msCodeGen(ssSW, ms, stencilInstantiation, cachePropertyMap_.at(ms->getID()));
    msCodeGen.generateCudaKernelCode();
  }
}

std::string CudaCodeGen::generateStencilInstantiation(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  using namespace codegen;

  std::stringstream ssSW;

  Namespace cudaNamespace("cuda", ssSW);

  // map from MS ID to cacheProperty
  for(const auto& ms : iterateIIROver<iir::MultiStage>(*(stencilInstantiation->getIIR()))) {
    cachePropertyMap_.emplace(ms->getID(), makeCacheProperties(ms, stencilInstantiation, 2));
  }

  generateAllCudaKernels(ssSW, stencilInstantiation);

  Class StencilWrapperClass(stencilInstantiation->getName(), ssSW);
  StencilWrapperClass.changeAccessibility("public");

  CodeGenProperties codeGenProperties = computeCodeGenProperties(stencilInstantiation.get());

  // generate code for base class of all the inner stencils
  Structure sbase = StencilWrapperClass.addStruct("sbase", "", "timer_cuda");
  auto baseCtr = sbase.addConstructor();
  baseCtr.addArg("std::string name");
  baseCtr.addInit("timer_cuda(name)");
  baseCtr.commit();
  MemberFunction gettime = sbase.addMemberFunction("double", "get_time");
  gettime.addStatement("return total_time()");
  gettime.commit();
  MemberFunction sbase_run = sbase.addMemberFunction("virtual void", "run");
  sbase_run.startBody();
  sbase_run.commit();
  MemberFunction sbase_sync = sbase.addMemberFunction("virtual void", "sync_storages");
  sbase_sync.startBody();
  sbase_sync.commit();

  MemberFunction sbaseVdtor = sbase.addMemberFunction("virtual", "~sbase");
  sbaseVdtor.startBody();
  sbaseVdtor.commit();
  sbase.commit();

  const auto& globalsMap = stencilInstantiation->getIIR()->getGlobalVariableMap();

  generateBoundaryConditionFunctions(StencilWrapperClass, stencilInstantiation);

  generateStencilClasses(stencilInstantiation, StencilWrapperClass, codeGenProperties);

  generateStencilWrapperMembers(StencilWrapperClass, stencilInstantiation, codeGenProperties);

  generateStencilWrapperCtr(StencilWrapperClass, stencilInstantiation, codeGenProperties);

  if(!globalsMap.empty()) {
    generateGlobalsAPI(*stencilInstantiation, StencilWrapperClass, globalsMap, codeGenProperties);
  }

  generateStencilWrapperRun(StencilWrapperClass, stencilInstantiation, codeGenProperties);

  generateStencilWrapperSyncMethod(StencilWrapperClass, stencilInstantiation, codeGenProperties);

  generateStencilWrapperPublicMemberFunctions(StencilWrapperClass, codeGenProperties);

  StencilWrapperClass.commit();

  cudaNamespace.commit();

  return ssSW.str();
}

void CudaCodeGen::generateStencilWrapperPublicMemberFunctions(
    Class& stencilWrapperClass, const CodeGenProperties& codeGenProperties) const {

  // Generate name getter
  stencilWrapperClass.addMemberFunction("std::string", "get_name")
      .isConst(true)
      .addStatement("return std::string(s_name)");

  std::vector<std::string> stencilMembers;

  for(const auto& stencilProp :
      codeGenProperties.getAllStencilProperties(StencilContext::SC_Stencil)) {
    stencilMembers.push_back("m_" + stencilProp.first);
  }

  // Generate stencil getter
  MemberFunction stencilGetter =
      stencilWrapperClass.addMemberFunction("std::vector<sbase*>", "getStencils");
  stencilGetter.addStatement("return " +
                             RangeToString(", ", "std::vector<sbase*>({", "})")(
                                 stencilMembers, [](const std::string& member) { return member; }));
  stencilGetter.commit();

  MemberFunction clearMeters = stencilWrapperClass.addMemberFunction("void", "reset_meters");
  clearMeters.startBody();
  std::string s = RangeToString("\n", "", "")(
      stencilMembers, [](const std::string& member) { return member + "->reset();"; });
  clearMeters << s;
  clearMeters.commit();
}

void CudaCodeGen::generateStencilClasses(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    Class& stencilWrapperClass, CodeGenProperties& codeGenProperties) const {
  // Generate stencils
  const auto& stencils = stencilInstantiation->getStencils();
  const auto& metadata = stencilInstantiation->getMetaData();

  const auto& globalsMap = stencilInstantiation->getIIR()->getGlobalVariableMap();

  // generate the code for each of the stencils
  for(const auto& stencilPtr : stencils) {
    const auto& stencil = *stencilPtr;

    std::string stencilName = "stencil_" + std::to_string(stencil.getStencilID());
    auto stencilProperties =
        codeGenProperties.getStencilProperties(StencilContext::SC_Stencil, stencilName);

    if(stencil.isEmpty())
      continue;

    // fields used in the stencil
    const auto stencilFields = orderMap(stencil.getFields());

    auto nonTempFields = makeRange(
        stencilFields, std::function<bool(std::pair<int, iir::Stencil::FieldInfo> const&)>(
                           [](std::pair<int, iir::Stencil::FieldInfo> const& p) {
                             return !p.second.IsTemporary;
                           }));
    auto tempFields = makeRange(
        stencilFields,
        std::function<bool(std::pair<int, iir::Stencil::FieldInfo> const&)>(
            [](std::pair<int, iir::Stencil::FieldInfo> const& p) { return p.second.IsTemporary; }));

    Structure stencilClass = stencilWrapperClass.addStruct(stencilName, "", "sbase");
    auto& paramNameToType = stencilProperties->paramNameToType_;

    for(auto fieldIt : nonTempFields) {
      paramNameToType.emplace((*fieldIt).second.Name,
                              getStorageType(metadata.getFieldDimensionsMask((*fieldIt).first)));
    }

    for(auto fieldIt : tempFields) {
      paramNameToType.emplace((*fieldIt).second.Name, c_gtc().str() + "storage_t");
    }

    generateStencilClassMembers(stencilClass, stencil, globalsMap, nonTempFields, tempFields,
                                stencilProperties);

    stencilClass.changeAccessibility("public");

    generateStencilClassCtr(stencilClass, stencil, globalsMap, nonTempFields, tempFields,
                            stencilProperties);

    // virtual dtor
    MemberFunction stencilClassDtr = stencilClass.addDestructor();
    stencilClassDtr.startBody();
    stencilClassDtr.commit();

    // synchronize storages method
    MemberFunction syncStoragesMethod = stencilClass.addMemberFunction("void", "sync_storages", "");
    syncStoragesMethod.startBody();

    for(auto fieldIt : nonTempFields) {
      syncStoragesMethod.addStatement("m_" + (*fieldIt).second.Name + ".sync()");
    }

    syncStoragesMethod.commit();

    //
    // Run-Method
    //
    generateStencilRunMethod(stencilClass, stencil, stencilInstantiation, paramNameToType,
                             globalsMap);

    // Generate stencil getter
    stencilClass.addMemberFunction("sbase*", "get_stencil").addStatement("return this");
  }
}

void CudaCodeGen::generateStencilClassMembers(
    Structure& stencilClass, const iir::Stencil& stencil, const sir::GlobalVariableMap& globalsMap,
    IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& nonTempFields,
    IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& tempFields,
    std::shared_ptr<StencilProperties> stencilProperties) const {

  auto& paramNameToType = stencilProperties->paramNameToType_;

  stencilClass.addComment("Members");
  stencilClass.addComment("Temporary storage typedefs");
  addTempStorageTypedef(stencilClass, stencil);

  if(!globalsMap.empty()) {
    stencilClass.addMember("globals&", "m_globals");
  }

  stencilClass.addMember("const " + c_gtc() + "domain&", "m_dom");

  stencilClass.addComment("storage declarations");
  for(auto fieldIt : nonTempFields) {
    stencilClass.addMember(paramNameToType.at((*fieldIt).second.Name) + "&",
                           "m_" + (*fieldIt).second.Name);
  }

  stencilClass.addComment("temporary storage declarations");
  addTmpStorageDeclaration(stencilClass, tempFields);
}
void CudaCodeGen::generateStencilClassCtr(
    Structure& stencilClass, const iir::Stencil& stencil, const sir::GlobalVariableMap& globalsMap,
    IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& nonTempFields,
    IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& tempFields,
    std::shared_ptr<StencilProperties> stencilProperties) const {

  auto stencilClassCtr = stencilClass.addConstructor();

  auto& paramNameToType = stencilProperties->paramNameToType_;

  stencilClassCtr.addArg("const " + c_gtc() + "domain& dom_");
  if(!globalsMap.empty()) {
    stencilClassCtr.addArg("globals& globals_");
  }

  for(auto fieldIt : nonTempFields) {
    std::string fieldName = (*fieldIt).second.Name;
    stencilClassCtr.addArg(paramNameToType.at(fieldName) + "& " + fieldName + "_");
  }

  stencilClassCtr.addInit("sbase(\"" + stencilClass.getName() + "\")");
  stencilClassCtr.addInit("m_dom(dom_)");

  if(!globalsMap.empty()) {
    stencilClassCtr.addInit("m_globals(globals_)");
  }

  for(auto fieldIt : nonTempFields) {
    stencilClassCtr.addInit("m_" + (*fieldIt).second.Name + "(" + (*fieldIt).second.Name + "_)");
  }

  addTmpStorageInit(stencilClassCtr, stencil, tempFields);
  stencilClassCtr.commit();
}

void CudaCodeGen::generateStencilWrapperCtr(
    Class& stencilWrapperClass,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const CodeGenProperties& codeGenProperties) const {

  const auto& metadata = stencilInstantiation->getMetaData();
  const auto& globalsMap = stencilInstantiation->getIIR()->getGlobalVariableMap();

  // Generate stencil wrapper constructor
  auto StencilWrapperConstructor = stencilWrapperClass.addConstructor();
  StencilWrapperConstructor.addArg("const " + c_gtc() + "domain& dom");

  for(int fieldId : metadata.getAccessesOfType<iir::FieldAccessType::FAT_APIField>()) {
    StencilWrapperConstructor.addArg(getStorageType(metadata.getFieldDimensionsMask(fieldId)) +
                                     "& " + metadata.getFieldNameFromAccessID(fieldId));
  }

  const auto& stencils = stencilInstantiation->getStencils();

  // add the ctr initialization of each stencil
  for(const auto& stencilPtr : stencils) {
    iir::Stencil& stencil = *stencilPtr;
    if(stencil.isEmpty())
      continue;

    const auto stencilFields = orderMap(stencil.getFields());

    const std::string stencilName =
        codeGenProperties.getStencilName(StencilContext::SC_Stencil, stencil.getStencilID());

    std::string initCtr = "m_" + stencilName + "(new " + stencilName;

    initCtr += "(dom";
    if(!globalsMap.empty()) {
      initCtr += ",m_globals";
    }

    for(const auto& fieldInfoPair : stencilFields) {
      const auto& fieldInfo = fieldInfoPair.second;
      if(fieldInfo.IsTemporary)
        continue;
      initCtr += "," + (metadata.isAccessType(iir::FieldAccessType::FAT_InterStencilTemporary,
                                              fieldInfo.field.getAccessID())
                            ? ("m_" + fieldInfo.Name)
                            : (fieldInfo.Name));
    }
    initCtr += ") )";
    StencilWrapperConstructor.addInit(initCtr);
  }

  if(metadata.hasAccessesOfType<iir::FieldAccessType::FAT_InterStencilTemporary>()) {
    std::vector<std::string> tempFields;
    for(auto accessID :
        metadata.getAccessesOfType<iir::FieldAccessType::FAT_InterStencilTemporary>()) {
      tempFields.push_back(metadata.getFieldNameFromAccessID(accessID));
    }
    addTmpStorageInitStencilWrapperCtr(StencilWrapperConstructor, stencils, tempFields);
  }

  addBCFieldInitStencilWrapperCtr(StencilWrapperConstructor, codeGenProperties);

  StencilWrapperConstructor.commit();
}

void CudaCodeGen::generateStencilWrapperMembers(
    Class& stencilWrapperClass,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    CodeGenProperties& codeGenProperties) const {

  const auto& metadata = stencilInstantiation->getMetaData();
  const auto& globalsMap = stencilInstantiation->getIIR()->getGlobalVariableMap();

  stencilWrapperClass.addMember("static constexpr const char* s_name =",
                                Twine("\"") + stencilWrapperClass.getName() + Twine("\""));

  for(auto stencilPropertiesPair :
      codeGenProperties.stencilProperties(StencilContext::SC_Stencil)) {
    stencilWrapperClass.addMember("sbase*", "m_" + stencilPropertiesPair.second->name_);
  }

  stencilWrapperClass.changeAccessibility("public");
  stencilWrapperClass.addCopyConstructor(Class::Deleted);

  stencilWrapperClass.addComment("Members");

  //
  // Members
  //
  generateBCFieldMembers(stencilWrapperClass, stencilInstantiation, codeGenProperties);

  stencilWrapperClass.addComment("Stencil-Data");

  // Define allocated memebers if necessary
  if(metadata.hasAccessesOfType<iir::FieldAccessType::FAT_InterStencilTemporary>()) {
    stencilWrapperClass.addMember(c_gtc() + "meta_data_t", "m_meta_data");

    for(int AccessID :
        metadata.getAccessesOfType<iir::FieldAccessType::FAT_InterStencilTemporary>())
      stencilWrapperClass.addMember(
          c_gtc() + "storage_t",
          "m_" + stencilInstantiation->getMetaData().getFieldNameFromAccessID(AccessID));
  }

  if(!globalsMap.empty()) {
    stencilWrapperClass.addMember("globals", "m_globals");
  }
}

void CudaCodeGen::generateStencilWrapperRun(
    Class& stencilWrapperClass,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const CodeGenProperties& codeGenProperties) const {
  // Generate the run method by generate code for the stencil description AST
  MemberFunction RunMethod = stencilWrapperClass.addMemberFunction("void", "run", "");

  RunMethod.finishArgs();

  RunMethod.addStatement("sync_storages()");
  // generate the control flow code executing each inner stencil
  ASTStencilDesc stencilDescCGVisitor(stencilInstantiation->getMetaData(), codeGenProperties);
  stencilDescCGVisitor.setIndent(RunMethod.getIndent());
  for(const auto& statement :
      stencilInstantiation->getIIR()->getControlFlowDescriptor().getStatements()) {
    statement->ASTStmt->accept(stencilDescCGVisitor);
    RunMethod.addStatement(stencilDescCGVisitor.getCodeAndResetStream());
  }

  RunMethod.addStatement("sync_storages()");
  RunMethod.commit();
}

void CudaCodeGen::generateStencilWrapperSyncMethod(
    Class& stencilWrapperClass,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const CodeGenProperties& codeGenProperties) const {
  // Generate the run method by generate code for the stencil description AST
  MemberFunction syncMethod = stencilWrapperClass.addMemberFunction("void", "sync_storages");

  syncMethod.finishArgs();

  const auto& stencils = stencilInstantiation->getStencils();

  // add the ctr initialization of each stencil
  for(const auto& stencilPtr : stencils) {
    iir::Stencil& stencil = *stencilPtr;
    if(stencil.isEmpty())
      continue;

    const std::string stencilName =
        codeGenProperties.getStencilName(StencilContext::SC_Stencil, stencil.getStencilID());

    syncMethod.addStatement("m_" + stencilName + "->sync_storages()");
  }

  syncMethod.commit();
}

void CudaCodeGen::generateStencilRunMethod(
    Structure& stencilClass, const iir::Stencil& stencil,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const std::unordered_map<std::string, std::string>& paramNameToType,
    const sir::GlobalVariableMap& globalsMap) const {
  MemberFunction StencilRunMethod = stencilClass.addMemberFunction("virtual void", "run", "");
  const auto& metadata = stencilInstantiation->getMetaData();

  StencilRunMethod.startBody();

  StencilRunMethod.addComment("starting timers");
  StencilRunMethod.addStatement("start()");

  for(const auto& multiStagePtr : stencil.getChildren()) {
    StencilRunMethod.addStatement("{");

    const iir::MultiStage& multiStage = *multiStagePtr;
    bool solveKLoopInParallel_ = CodeGeneratorHelper::solveKLoopInParallel(multiStagePtr);

    const auto fields = orderMap(multiStage.getFields());

    auto nonTempFields = makeRange(fields, std::function<bool(std::pair<int, iir::Field> const&)>(
                                               [&](std::pair<int, iir::Field> const& p) {
                                                 return !metadata.isAccessType(
                                                     iir::FieldAccessType::FAT_StencilTemporary,
                                                     p.second.getAccessID());
                                               }));

    auto tempStencilFieldsNonLocalCached = makeRange(
        fields,
        std::function<bool(std::pair<int, iir::Field> const&)>(
            [&](std::pair<int, iir::Field> const& p) {
              const int accessID = p.first;
              if(!metadata.isAccessType(iir::FieldAccessType::FAT_StencilTemporary,
                                        p.second.getAccessID()))
                return false;
              for(const auto& ms : iterateIIROver<iir::MultiStage>(stencil)) {
                if(!ms->isCached(accessID))
                  continue;
                if(ms->getCache(accessID).getCacheIOPolicy() == iir::Cache::CacheIOPolicy::local)
                  return false;
              }

              return true;
            }));

    // create all the data views
    for(auto fieldIt : nonTempFields) {
      // TODO have the same FieldInfo in ms level so that we dont need to query
      // stencilInstantiation
      // all the time for name and IsTmpField
      const auto fieldName = metadata.getFieldNameFromAccessID((*fieldIt).second.getAccessID());
      StencilRunMethod.addStatement(c_gt() + "data_view<" + paramNameToType.at(fieldName) + "> " +
                                    fieldName + "= " + c_gt() + "make_device_view(m_" + fieldName +
                                    ")");
    }
    for(auto fieldIt : tempStencilFieldsNonLocalCached) {
      const auto fieldName = metadata.getFieldNameFromAccessID((*fieldIt).second.getAccessID());

      StencilRunMethod.addStatement(c_gt() + "data_view<tmp_storage_t> " + fieldName + "= " +
                                    c_gt() + "make_device_view(m_" + fieldName + ")");
    }

    DAWN_ASSERT(nonTempFields.size() > 0);

    iir::Extents maxExtents{0, 0, 0, 0, 0, 0};
    for(const auto& stage : iterateIIROver<iir::Stage>(*multiStagePtr)) {
      maxExtents.merge(stage->getExtents());
    }

    StencilRunMethod.addStatement(
        "const unsigned int nx = m_dom.isize() - m_dom.iminus() - m_dom.iplus()");
    StencilRunMethod.addStatement(
        "const unsigned int ny = m_dom.jsize() - m_dom.jminus() - m_dom.jplus()");
    StencilRunMethod.addStatement(
        "const unsigned int nz = m_dom.ksize() - m_dom.kminus() - m_dom.kplus()");

    const auto blockSize = stencilInstantiation->getIIR()->getBlockSize();

    unsigned int ntx = blockSize[0];
    unsigned int nty = blockSize[1];

    StencilRunMethod.addStatement(
        "dim3 threads(" + std::to_string(ntx) + "," + std::to_string(nty) + "+" +
        std::to_string(maxExtents[1].Plus - maxExtents[1].Minus +
                       (maxExtents[0].Minus < 0 ? 1 : 0) + (maxExtents[0].Plus > 0 ? 1 : 0)) +
        ",1)");

    // number of blocks required
    StencilRunMethod.addStatement("const unsigned int nbx = (nx + " + std::to_string(ntx) +
                                  " - 1) / " + std::to_string(ntx));
    StencilRunMethod.addStatement("const unsigned int nby = (ny + " + std::to_string(nty) +
                                  " - 1) / " + std::to_string(nty));
    if(solveKLoopInParallel_) {
      StencilRunMethod.addStatement("const unsigned int nbz = (m_dom.ksize()+" +
                                    std::to_string(blockSize[2]) + "-1) / " +
                                    std::to_string(blockSize[2]));
    } else {
      StencilRunMethod.addStatement("const unsigned int nbz = 1");
    }
    StencilRunMethod.addStatement("dim3 blocks(nbx, nby, nbz)");
    std::string kernelCall =
        CodeGeneratorHelper::buildCudaKernelName(stencilInstantiation, multiStagePtr) +
        "<<<blocks, threads>>>(";

    if(!globalsMap.empty()) {
      kernelCall = kernelCall + "m_globals,";
    }

    auto tempMSFieldsNonLocalCached = makeRange(
        fields, std::function<bool(std::pair<int, iir::Field> const&)>(
                    [&](std::pair<int, iir::Field> const& p) {
                      const int accessID = p.first;
                      if(!metadata.isAccessType(iir::FieldAccessType::FAT_StencilTemporary,
                                                p.second.getAccessID()))
                        return false;
                      if(!multiStage.isCached(accessID))
                        return true;
                      if(multiStage.getCache(accessID).getCacheIOPolicy() ==
                         iir::Cache::CacheIOPolicy::local)
                        return false;

                      return true;
                    }));

    // TODO enable const auto& below and/or enable use RangeToString
    std::string args;
    int idx = 0;
    for(auto field : nonTempFields) {
      const auto fieldName = metadata.getFieldNameFromAccessID((*field).second.getAccessID());

      args = args + (idx == 0 ? "" : ",") + "(" + fieldName + ".data()+" + "m_" + fieldName +
             ".get_storage_info_ptr()->index(" + fieldName + ".begin<0>(), " + fieldName +
             ".begin<1>(),0 ))";
      ++idx;
    }
    DAWN_ASSERT(nonTempFields.size() > 0);
    for(auto field : tempMSFieldsNonLocalCached) {
      // in some cases (where there are no horizontal extents) we dont use the special tmp index
      // iterator, but rather a normal 3d field index iterator. In that case we pass temporaries in
      // the same manner as normal fields
      if(!CodeGeneratorHelper::useTemporaries(multiStagePtr->getParent(), metadata)) {
        const auto fieldName = metadata.getFieldNameFromAccessID((*field).second.getAccessID());

        args = args + ", (" + fieldName + ".data()+" + "m_" + fieldName +
               ".get_storage_info_ptr()->index(" + fieldName + ".begin<0>(), " + fieldName +
               ".begin<1>()," + fieldName + ".begin<2>()," + fieldName + ".begin<3>(), 0))";
      } else {
        args = args + "," + metadata.getFieldNameFromAccessID((*field).second.getAccessID());
      }
    }

    std::vector<std::string> strides = CodeGeneratorHelper::generateStrideArguments(
        nonTempFields, tempMSFieldsNonLocalCached, stencilInstantiation, multiStagePtr,
        CodeGeneratorHelper::FunctionArgType::FT_Caller);

    DAWN_ASSERT(!strides.empty());

    kernelCall = kernelCall + "nx,ny,nz," + RangeToString(",", "", "")(strides) + "," + args + ")";

    StencilRunMethod.addStatement(kernelCall);

    StencilRunMethod.addStatement("}");
  }

  StencilRunMethod.addComment("stopping timers");
  StencilRunMethod.addStatement("pause()");

  StencilRunMethod.commit();
}

void CudaCodeGen::addTempStorageTypedef(Structure& stencilClass,
                                        iir::Stencil const& stencil) const {

  auto maxExtents = CodeGeneratorHelper::computeTempMaxWriteExtent(stencil);
  stencilClass.addTypeDef("tmp_halo_t")
      .addType("gridtools::halo< " + std::to_string(-maxExtents[0].Minus) + "," +
               std::to_string(-maxExtents[1].Minus) + ", 0, 0, " +
               std::to_string(getVerticalTmpHaloSize(stencil)) + ">");

  stencilClass.addTypeDef(tmpMetadataTypename_)
      .addType("storage_traits_t::storage_info_t< 0, 5, tmp_halo_t >");

  stencilClass.addTypeDef(tmpStorageTypename_)
      .addType("storage_traits_t::data_store_t< float_type, " + tmpMetadataTypename_ + ">");
}

void CudaCodeGen::addTmpStorageInit(
    MemberFunction& ctr, iir::Stencil const& stencil,
    IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& tempFields) const {
  auto maxExtents = CodeGeneratorHelper::computeTempMaxWriteExtent(stencil);

  const auto blockSize = stencil.getParent()->getBlockSize();

  if(!(tempFields.empty())) {
    ctr.addInit(tmpMetadataName_ + "(" + std::to_string(blockSize[0]) + "+" +
                std::to_string(-maxExtents[0].Minus + maxExtents[0].Plus) + ", " +
                std::to_string(blockSize[1]) + "+" +
                std::to_string(-maxExtents[1].Minus + maxExtents[1].Plus) + ", (dom_.isize()+ " +
                std::to_string(blockSize[0]) + " - 1) / " + std::to_string(blockSize[0]) +
                ", (dom_.jsize()+ " + std::to_string(blockSize[1]) + " - 1) / " +
                std::to_string(blockSize[1]) + ", dom_.ksize() + 2 * " +
                std::to_string(getVerticalTmpHaloSize(stencil)) + ")");
    for(auto fieldIt : tempFields) {
      ctr.addInit("m_" + (*fieldIt).second.Name + "(" + tmpMetadataName_ + ")");
    }
  }
}

std::unique_ptr<TranslationUnit> CudaCodeGen::generateCode() {
  DAWN_LOG(INFO) << "Starting code generation for GTClang ...";

  // Generate code for StencilInstantiations
  std::map<std::string, std::string> stencils;
  for(const auto& nameStencilCtxPair : context_->getStencilInstantiationMap()) {
    std::shared_ptr<iir::StencilInstantiation> origSI = nameStencilCtxPair.second;
    // TODO the clone seems to be broken
    //    std::shared_ptr<iir::StencilInstantiation> stencilInstantiation = origSI->clone();
    std::shared_ptr<iir::StencilInstantiation> stencilInstantiation = origSI;

    PassInlining inliner(true, PassInlining::InlineStrategyKind::IK_ComputationsOnTheFly);

    inliner.run(stencilInstantiation);

    std::string code = generateStencilInstantiation(stencilInstantiation);
    if(code.empty())
      return nullptr;
    stencils.emplace(nameStencilCtxPair.first, std::move(code));
  }

  std::string globals = generateGlobals(context_->getSIR(), "cuda");

  std::vector<std::string> ppDefines;
  auto makeDefine = [](std::string define, int value) {
    return "#define " + define + " " + std::to_string(value);
  };

  ppDefines.push_back(makeDefine("GRIDTOOLS_CLANG_GENERATED", 1));
  ppDefines.push_back("#define GRIDTOOLS_CLANG_BACKEND_T CUDA");
  //==============------------------------------------------------------------------------------===
  // BENCHMARKTODO: since we're importing two cpp files into the benchmark API we need to set
  // these
  // variables also in the naive code-generation in order to not break it. Once the move to
  // different TU's is completed, this is no longer necessary.
  // [https://github.com/MeteoSwiss-APN/gtclang/issues/32]
  //==============------------------------------------------------------------------------------===
  CodeGen::addMplIfdefs(ppDefines, 30, context_->getOptions().MaxHaloPoints);

  generateBCHeaders(ppDefines);

  DAWN_LOG(INFO) << "Done generating code";

  // TODO missing the BC
  return make_unique<TranslationUnit>(context_->getSIR()->Filename, std::move(ppDefines),
                                      std::move(stencils), std::move(globals));
}

} // namespace cuda
} // namespace codegen
} // namespace dawn
