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
#include "dawn/CodeGen/Cuda/MSCodeGen.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Array.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Iterator.h"
#include "dawn/Support/Logger.h"
#include "dawn/Support/StringUtil.h"
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

namespace dawn {
namespace codegen {
namespace cuda {

namespace {
std::string makeIntervalBoundExplicit(std::string dim, const iir::Interval& interval,
                                      iir::Interval::Bound bound, std::string dom) {
  if(interval.levelIsEnd(bound)) {
    return dom + "." + dim + "size() - " + dom + "." + dim + "plus()  + " +
           std::to_string(interval.offset(bound));
  }
  auto notEnd = interval.level(bound);
  if(notEnd == 0) {
    return dom + "." + dim + "minus() + " + std::to_string(interval.offset(bound));
  }
  return dom + "." + dim + "minus() + " + std::to_string(notEnd + interval.offset(bound));
}
} // namespace

std::unique_ptr<TranslationUnit>
run(const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>&
        stencilInstantiationMap,
    const Options& options) {
  const Array3i domain_size{options.DomainSizeI, options.DomainSizeJ, options.DomainSizeK};
  CudaCodeGen CG(stencilInstantiationMap, options.MaxHaloSize, options.nsms, options.MaxBlocksPerSM,
                 domain_size, options.RunWithSync);

  return CG.generateCode();
}

CudaCodeGen::CudaCodeGen(const StencilInstantiationContext& ctx, int maxHaloPoints, int nsms,
                         int maxBlocksPerSM, const Array3i& domainSize, bool runWithSync)
    : CodeGen(ctx, maxHaloPoints), codeGenOptions_{nsms, maxBlocksPerSM, domainSize, runWithSync} {}

CudaCodeGen::~CudaCodeGen() {}

void CudaCodeGen::generateAllCudaKernels(
    std::stringstream& ssSW,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  for(const auto& ms : iterateIIROver<iir::MultiStage>(*(stencilInstantiation->getIIR()))) {
    DAWN_ASSERT(cachePropertyMap_.count(ms->getID()));

    MSCodeGen msCodeGen(ssSW, ms, stencilInstantiation, cachePropertyMap_.at(ms->getID()),
                        codeGenOptions_, hasGlobalIndices(stencilInstantiation));
    msCodeGen.generateCudaKernelCode();
  }
}

std::string CudaCodeGen::generateStencilInstantiation(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  using namespace codegen;

  std::stringstream ssSW;

  Namespace dawnNamespace("dawn_generated", ssSW);
  Namespace cudaNamespace("cuda", ssSW);

  // map from MS ID to cacheProperty
  for(const auto& ms : iterateIIROver<iir::MultiStage>(*(stencilInstantiation->getIIR()))) {
    cachePropertyMap_.emplace(ms->getID(), makeCacheProperties(ms, stencilInstantiation, 2));
  }

  generateAllCudaKernels(ssSW, stencilInstantiation);

  Class stencilWrapperClass(stencilInstantiation->getName(), ssSW);
  stencilWrapperClass.changeAccessibility("public");

  CodeGenProperties codeGenProperties = computeCodeGenProperties(stencilInstantiation.get());

  // generate code for base class of all the inner stencils
  Structure sbase = stencilWrapperClass.addStruct("sbase", "", "timer_cuda");
  auto baseCtr = sbase.addConstructor();
  baseCtr.addArg("std::string name");
  baseCtr.addInit("timer_cuda(name)");
  baseCtr.commit();
  MemberFunction gettime = sbase.addMemberFunction("double", "get_time");
  gettime.addStatement("return total_time()");
  gettime.commit();

  sbase.commit();

  const auto& globalsMap = stencilInstantiation->getIIR()->getGlobalVariableMap();

  generateBoundaryConditionFunctions(stencilWrapperClass, stencilInstantiation);

  generateStencilClasses(stencilInstantiation, stencilWrapperClass, codeGenProperties);

  generateStencilWrapperMembers(stencilWrapperClass, stencilInstantiation, codeGenProperties);

  generateStencilWrapperCtr(stencilWrapperClass, stencilInstantiation, codeGenProperties);

  if(!globalsMap.empty()) {
    generateGlobalsAPI(*stencilInstantiation, stencilWrapperClass, globalsMap, codeGenProperties);
  }

  generateStencilWrapperSyncMethod(stencilWrapperClass);

  generateStencilWrapperRun(stencilWrapperClass, stencilInstantiation, codeGenProperties);

  generateStencilWrapperPublicMemberFunctions(stencilWrapperClass, codeGenProperties);

  stencilWrapperClass.commit();

  cudaNamespace.commit();
  dawnNamespace.commit();

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

  MemberFunction clearMeters = stencilWrapperClass.addMemberFunction("void", "reset_meters");
  clearMeters.startBody();
  std::string s = RangeToString("\n", "", "")(
      stencilMembers, [](const std::string& member) { return member + ".reset();"; });
  clearMeters << s;
  clearMeters.commit();

  MemberFunction totalTime = stencilWrapperClass.addMemberFunction("double", "get_total_time");
  totalTime.startBody();
  totalTime.addStatement("double res = 0");
  std::string s1 = RangeToString(";\n", "", "")(
      stencilMembers, [](const std::string& member) { return "res +=" + member + ".get_time()"; });
  totalTime.addStatement(s1);
  totalTime.addStatement("return res");
  totalTime.commit();
}

void CudaCodeGen::generateStencilClasses(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    Class& stencilWrapperClass, CodeGenProperties& codeGenProperties) {
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
    const auto stencilFields = stencil.getOrderedFields();

    auto nonTempFields =
        makeRange(stencilFields, [](std::pair<int, iir::Stencil::FieldInfo> const& p) {
          return !p.second.IsTemporary;
        });

    auto tempFields =
        makeRange(stencilFields, [](std::pair<int, iir::Stencil::FieldInfo> const& p) {
          return p.second.IsTemporary;
        });

    Structure stencilClass = stencilWrapperClass.addStruct(stencilName, "", "sbase");
    auto& paramNameToType = stencilProperties->paramNameToType_;

    for(const auto& fieldPair : nonTempFields) {
      paramNameToType.emplace(fieldPair.second.Name,
                              getStorageType(metadata.getFieldDimensions(fieldPair.first)));
    }

    for(const auto& fieldPair : tempFields) {
      paramNameToType.emplace(fieldPair.second.Name, c_dgt() + "storage_t");
    }

    iterationSpaceSet_ = hasGlobalIndices(stencil);
    generateStencilClassMembers(stencilClass, stencil, globalsMap, nonTempFields, tempFields,
                                stencilProperties);

    stencilClass.changeAccessibility("public");

    generateStencilClassCtr(stencilClass, stencil, globalsMap, nonTempFields, tempFields,
                            stencilProperties);

    // accumulated extents of API fields
    generateFieldExtentsInfo(stencilClass, nonTempFields, ast::GridType::Cartesian);

    //
    // Run-Method
    //
    generateStencilRunMethod(stencilClass, stencil, stencilProperties, stencilInstantiation,
                             paramNameToType, globalsMap);
  }
}

void CudaCodeGen::generateStencilClassMembers(
    Structure& stencilClass, const iir::Stencil& stencil, const sir::GlobalVariableMap& globalsMap,
    IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& nonTempFields,
    IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& tempFields,
    std::shared_ptr<StencilProperties> stencilProperties) const {

  stencilClass.addComment("Members");
  if(iterationSpaceSet_) {
    generateGlobalIndices(stencil, stencilClass, false);
  }

  stencilClass.addComment("Temporary storage typedefs");
  addTempStorageTypedef(stencilClass, stencil);

  if(!globalsMap.empty()) {
    stencilClass.addMember("globals&", "m_globals");
  }

  stencilClass.addMember("const " + c_dgt() + "domain", "m_dom");

  if(!tempFields.empty()) {
    stencilClass.addComment("temporary storage declarations");
    addTmpStorageDeclaration(stencilClass, tempFields);
  }
}

void CudaCodeGen::generateStencilClassCtr(
    Structure& stencilClass, const iir::Stencil& stencil, const sir::GlobalVariableMap& globalsMap,
    IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& nonTempFields,
    IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& tempFields,
    std::shared_ptr<StencilProperties> stencilProperties) const {

  auto stencilClassCtr = stencilClass.addConstructor();

  stencilClassCtr.addArg("const " + c_dgt() + "domain& dom_");
  if(!globalsMap.empty()) {
    stencilClassCtr.addArg("globals& globals_");
  }
  stencilClassCtr.addArg("int rank");
  stencilClassCtr.addArg("int xcols");
  stencilClassCtr.addArg("int ycols");

  stencilClassCtr.addInit("sbase(\"" + stencilClass.getName() + "\")");
  stencilClassCtr.addInit("m_dom(dom_)");

  if(!globalsMap.empty()) {
    stencilClassCtr.addInit("m_globals(globals_)");
  }

  std::string iterators = "ij";
  for(auto& stage : iterateIIROver<iir::Stage>(stencil)) {
    int index = 0;
    for(const auto& interval : stage->getIterationSpace()) {
      if(interval.has_value()) {
        std::string iterator = iterators.substr(index, 1);
        std::string arrName = "stage" + std::to_string(stage->getStageID()) + "Global" +
                              (char)std::toupper(iterator.at(0)) + "Indices";
        stencilClassCtr.addInit(arrName + "({" +
                                makeIntervalBoundExplicit(iterator, interval.value(),
                                                          iir::Interval::Bound::lower, "dom_") +
                                " , " +
                                makeIntervalBoundExplicit(iterator, interval.value(),
                                                          iir::Interval::Bound::upper, "dom_") +
                                "})");
      }
      index += 1;
    }
  }

  if(iterationSpaceSet_) {
    stencilClassCtr.addInit("globalOffsets({computeGlobalOffsets(rank, m_dom, xcols, ycols)})");
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
  StencilWrapperConstructor.addArg("const " + c_dgt() + "domain& dom");
  StencilWrapperConstructor.addArg("int rank = 1");
  StencilWrapperConstructor.addArg("int xcols = 1");
  StencilWrapperConstructor.addArg("int ycols = 1");

  const auto& stencils = stencilInstantiation->getStencils();

  // add the ctr initialization of each stencil
  for(const auto& stencilPtr : stencils) {
    iir::Stencil& stencil = *stencilPtr;
    if(stencil.isEmpty())
      continue;

    const auto stencilFields = stencil.getOrderedFields();

    const std::string stencilName =
        codeGenProperties.getStencilName(StencilContext::SC_Stencil, stencil.getStencilID());

    std::string initCtr = "m_" + stencilName;

    initCtr += "(dom";
    if(!globalsMap.empty()) {
      initCtr += ",m_globals";
    }
    initCtr += ", rank";
    initCtr += ", xcols";
    initCtr += ", ycols";
    initCtr += ")";
    StencilWrapperConstructor.addInit(initCtr);
  }

  if(metadata.hasAccessesOfType<iir::FieldAccessType::InterStencilTemporary>()) {
    std::vector<std::string> tempFields;
    for(auto accessID : metadata.getAccessesOfType<iir::FieldAccessType::InterStencilTemporary>()) {
      tempFields.push_back(metadata.getFieldNameFromAccessID(accessID));
    }
    addTmpStorageInitStencilWrapperCtr(StencilWrapperConstructor, stencils, tempFields);
  }

  StencilWrapperConstructor.commit();
}

void CudaCodeGen::generateStencilWrapperMembers(
    Class& stencilWrapperClass,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    CodeGenProperties& codeGenProperties) const {

  const auto& metadata = stencilInstantiation->getMetaData();
  const auto& globalsMap = stencilInstantiation->getIIR()->getGlobalVariableMap();

  stencilWrapperClass.addMember("static constexpr const char* s_name =",
                                "\"" + stencilWrapperClass.getName() + "\"");

  for(auto stencilPropertiesPair :
      codeGenProperties.stencilProperties(StencilContext::SC_Stencil)) {
    stencilWrapperClass.addMember(stencilPropertiesPair.second->name_,
                                  "m_" + stencilPropertiesPair.second->name_);
  }

  stencilWrapperClass.changeAccessibility("public");
  stencilWrapperClass.addCopyConstructor(Class::ConstructorDefaultKind::Deleted);

  stencilWrapperClass.addComment("Members");

  //
  // Members
  //
  stencilWrapperClass.addComment("Stencil-Data");

  // Define allocated memebers if necessary
  if(metadata.hasAccessesOfType<iir::FieldAccessType::InterStencilTemporary>()) {
    stencilWrapperClass.addMember(c_dgt() + "meta_data_t", "m_meta_data");

    for(int AccessID : metadata.getAccessesOfType<iir::FieldAccessType::InterStencilTemporary>())
      stencilWrapperClass.addMember(
          c_dgt() + "storage_t",
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
  const auto& metadata = stencilInstantiation->getMetaData();
  // Generate the run method by generate code for the stencil description AST
  MemberFunction RunMethod = stencilWrapperClass.addMemberFunction("void", "run", "");
  std::vector<std::string> apiFieldNames;

  for(const auto& fieldID : metadata.getAccessesOfType<iir::FieldAccessType::APIField>()) {
    std::string name = metadata.getFieldNameFromAccessID(fieldID);
    apiFieldNames.push_back(name);
  }

  for(const auto& fieldName : apiFieldNames) {
    RunMethod.addArg(codeGenProperties.getParamType(stencilInstantiation, fieldName) + " " +
                     fieldName);
  }

  RunMethod.finishArgs();

  RangeToString apiFieldArgs(",", "", "");

  bool withSync = codeGenOptions_.runWithSync;
  if(withSync) {
    RunMethod.addStatement("sync_storages(" + apiFieldArgs(apiFieldNames) + ")");
  }
  // generate the control flow code executing each inner stencil
  ASTStencilDesc stencilDescCGVisitor(stencilInstantiation, codeGenProperties);
  stencilDescCGVisitor.setIndent(RunMethod.getIndent());
  for(const auto& statement :
      stencilInstantiation->getIIR()->getControlFlowDescriptor().getStatements()) {
    statement->accept(stencilDescCGVisitor);
    RunMethod.addStatement(stencilDescCGVisitor.getCodeAndResetStream());
  }

  if(withSync) {
    RunMethod.addStatement("sync_storages(" + apiFieldArgs(apiFieldNames) + ")");
  }
  RunMethod.commit();
}

void CudaCodeGen::addCudaCopySymbol(MemberFunction& runMethod, const std::string& arrName,
                                    const std::string dataType) const {
  runMethod.addStatement("cudaMemcpyToSymbol(" + arrName + "_, " + arrName + ".data(), sizeof(" +
                         dataType + ") * " + arrName + ".size())");
}

void CudaCodeGen::generateStencilRunMethod(
    Structure& stencilClass, const iir::Stencil& stencil,
    const std::shared_ptr<StencilProperties>& stencilProperties,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const std::unordered_map<std::string, std::string>& paramNameToType,
    const sir::GlobalVariableMap& globalsMap) const {
  MemberFunction stencilRunMethod = stencilClass.addMemberFunction("void", "run", "");
  const auto& metadata = stencilInstantiation->getMetaData();

  // fields used in the stencil
  const auto stencilFields = stencil.getOrderedFields();

  auto nonTempFields =
      makeRange(stencilFields, [&](std::pair<int, iir::Stencil::FieldInfo> const& p) {
        return !p.second.IsTemporary && metadata.isAccessType(iir::FieldAccessType::Field, p.first);
      });

  for(const auto& field : nonTempFields) {
    stencilRunMethod.addArg(stencilProperties->paramNameToType_.at(field.second.Name) + " " +
                            field.second.Name + "_ds");
  }

  stencilRunMethod.startBody();

  stencilRunMethod.addComment("starting timers");
  stencilRunMethod.addStatement("start()");

  for(const auto& multiStagePtr : stencil.getChildren()) {
    const iir::MultiStage& multiStage = *multiStagePtr;
    bool solveKLoopInParallel_ = CodeGeneratorHelper::solveKLoopInParallel(multiStagePtr);

    const auto fields = multiStage.getOrderedFields();
    if(fields.empty())
      continue;

    stencilRunMethod.addStatement("{");

    auto msNonTempFields = makeRange(fields, [&](std::pair<int, iir::Field> const& p) {
      return !metadata.isAccessType(iir::FieldAccessType::StencilTemporary, p.second.getAccessID());
    });

    auto tempMSFieldsNonLocalCached = makeRange(fields, [&](std::pair<int, iir::Field> const& p) {
      const int accessID = p.first;
      if(!metadata.isAccessType(iir::FieldAccessType::StencilTemporary, p.second.getAccessID()))
        return false;
      if(!multiStage.isCached(accessID))
        return true;
      if(multiStage.getCache(accessID).getIOPolicy() == iir::Cache::IOPolicy::local)
        return false;
      return true;
    });

    // create all the data views
    for(const auto& fieldPair : msNonTempFields) {
      // TODO have the same FieldInfo in ms level so that we dont need to query
      // stencilInstantiation all the time for name and IsTmpField
      const auto fieldName = metadata.getFieldNameFromAccessID(fieldPair.second.getAccessID());
      stencilRunMethod.addStatement(c_gt() + "data_view<" + paramNameToType.at(fieldName) + "> " +
                                    fieldName + "= " + c_gt() + "make_device_view(" + fieldName +
                                    "_ds)");
    }

    for(const auto& fieldPair : tempMSFieldsNonLocalCached) {
      const auto fieldName = metadata.getFieldNameFromAccessID(fieldPair.second.getAccessID());
      stencilRunMethod.addStatement(c_gt() + "data_view<tmp_storage_t> " + fieldName + "= " +
                                    c_gt() + "make_device_view( m_" + fieldName + ")");
    }

    iir::Extents maxExtents{ast::cartesian};
    for(const auto& stage : iterateIIROver<iir::Stage>(*multiStagePtr)) {
      maxExtents.merge(stage->getExtents());
    }

    stencilRunMethod.addStatement(
        "const unsigned int nx = m_dom.isize() - m_dom.iminus() - m_dom.iplus()");
    stencilRunMethod.addStatement(
        "const unsigned int ny = m_dom.jsize() - m_dom.jminus() - m_dom.jplus()");
    stencilRunMethod.addStatement(
        "const unsigned int nz = m_dom.ksize() - m_dom.kminus() - m_dom.kplus()");

    const auto blockSize = stencilInstantiation->getIIR()->getBlockSize();

    unsigned int ntx = blockSize[0];
    unsigned int nty = blockSize[1];

    auto const& hMaxExtents =
        iir::extent_cast<iir::CartesianExtent const&>(maxExtents.horizontalExtent());

    stencilRunMethod.addStatement(
        "dim3 threads(" + std::to_string(ntx) + "," + std::to_string(nty) + "+" +
        std::to_string(hMaxExtents.jPlus() - hMaxExtents.jMinus() +
                       (hMaxExtents.iMinus() < 0 ? 1 : 0) + (hMaxExtents.iPlus() > 0 ? 1 : 0)) +
        ",1)");

    // number of blocks required
    stencilRunMethod.addStatement("const unsigned int nbx = (nx + " + std::to_string(ntx) +
                                  " - 1) / " + std::to_string(ntx));
    stencilRunMethod.addStatement("const unsigned int nby = (ny + " + std::to_string(nty) +
                                  " - 1) / " + std::to_string(nty));
    if(solveKLoopInParallel_) {
      stencilRunMethod.addStatement("const unsigned int nbz = (m_dom.ksize()+" +
                                    std::to_string(blockSize[2]) + "-1) / " +
                                    std::to_string(blockSize[2]));
    } else {
      stencilRunMethod.addStatement("const unsigned int nbz = 1");
    }

    if(iterationSpaceSet_) {
      std::string iterators = "IJ";
      for(auto& stage : iterateIIROver<iir::Stage>(stencil)) {
        for(auto [index, interval] : enumerate(stage->getIterationSpace())) {
          if(interval.has_value()) {
            std::string hostName = "stage" + std::to_string(stage->getStageID()) + "Global" +
                                   iterators.at(index) + "Indices";
            addCudaCopySymbol(stencilRunMethod, hostName, "int");
          }
        }
      }
      addCudaCopySymbol(stencilRunMethod, "globalOffsets", "unsigned");
    }

    stencilRunMethod.addStatement("dim3 blocks(nbx, nby, nbz)");
    std::string kernelCall =
        CodeGeneratorHelper::buildCudaKernelName(stencilInstantiation, multiStagePtr) +
        "<<<blocks, threads>>>(";

    if(!globalsMap.empty()) {
      kernelCall = kernelCall + "m_globals,";
    }

    // TODO enable const auto& below and/or enable use RangeToString
    std::string args;
    int idx = 0;
    for(const auto& fieldPair : msNonTempFields) {
      const auto fieldName = metadata.getFieldNameFromAccessID(fieldPair.second.getAccessID());
      if(idx > 0)
        args += ",";
      args += "(" + fieldName + ".data()+" + fieldName +
             "_ds.get_storage_info_ptr()->index(" + fieldName + ".begin<0>(), " + fieldName +
             ".begin<1>(),0 ))";
      ++idx;
    }

    if(!args.empty() && !tempMSFieldsNonLocalCached.empty())
      args += ",";

    idx = 0;
    for(const auto& fieldPair : tempMSFieldsNonLocalCached) {
      // in some cases (where there are no horizontal extents) we dont use the special tmp index
      // iterator, but rather a normal 3d field index iterator. In that case we pass temporaries in
      // the same manner as normal fields
      if(idx > 0)
        args += ",";
      if(!CodeGeneratorHelper::useTemporaries(multiStagePtr->getParent(), metadata)) {
        const auto fieldName = metadata.getFieldNameFromAccessID(fieldPair.second.getAccessID());
        args += "(" + fieldName + ".data()+ m_" + fieldName +
               ".get_storage_info_ptr()->index(" + fieldName + ".begin<0>(), " + fieldName +
               ".begin<1>()," + fieldName + ".begin<2>()," + fieldName + ".begin<3>(), 0))";
      } else {
        args += metadata.getFieldNameFromAccessID(fieldPair.second.getAccessID());
      }
      ++idx;
    }

    std::vector<std::string> strides = CodeGeneratorHelper::generateStrideArguments(
        msNonTempFields, tempMSFieldsNonLocalCached, stencilInstantiation, multiStagePtr,
        CodeGeneratorHelper::FunctionArgType::FT_Caller);

    kernelCall += "nx,ny,nz";
    if(!strides.empty())
      kernelCall += "," + RangeToString(",", "", "")(strides);
    if(!args.empty())
      kernelCall += "," + args;
    kernelCall +=  ")";

    stencilRunMethod.addStatement(kernelCall);
    stencilRunMethod.addStatement("}");
  }

  stencilRunMethod.addComment("stopping timers");
  stencilRunMethod.addStatement("pause()");

  stencilRunMethod.commit();
}

void CudaCodeGen::addTempStorageTypedef(Structure& stencilClass,
                                        iir::Stencil const& stencil) const {

  auto maxExtents = CodeGeneratorHelper::computeTempMaxWriteExtent(stencil);
  auto const& hMaxExtents =
      iir::extent_cast<iir::CartesianExtent const&>(maxExtents.horizontalExtent());

  stencilClass.addTypeDef("tmp_halo_t")
      .addType("gridtools::halo< " + std::to_string(-hMaxExtents.iMinus()) + "," +
               std::to_string(-hMaxExtents.jMinus()) + ", 0, 0, " +
               std::to_string(getVerticalTmpHaloSize(stencil)) + ">");

  stencilClass.addTypeDef(tmpMetadataTypename_)
      .addType("storage_traits_t::storage_info_t< 0, 5, tmp_halo_t >");

  stencilClass.addTypeDef(tmpStorageTypename_)
      .addType("storage_traits_t::data_store_t< ::dawn::float_type, " + tmpMetadataTypename_ + ">");
}

void CudaCodeGen::addTmpStorageInit(
    MemberFunction& ctr, iir::Stencil const& stencil,
    IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& tempFields) const {
  auto maxExtents = CodeGeneratorHelper::computeTempMaxWriteExtent(stencil);

  const auto blockSize = stencil.getParent()->getBlockSize();

  if(!(tempFields.empty())) {
    auto const& hMaxExtents =
        iir::extent_cast<iir::CartesianExtent const&>(maxExtents.horizontalExtent());
    ctr.addInit(tmpMetadataName_ + "(" + std::to_string(blockSize[0]) + "+" +
                std::to_string(-hMaxExtents.iMinus() + hMaxExtents.iPlus()) + ", " +
                std::to_string(blockSize[1]) + "+" +
                std::to_string(-hMaxExtents.jMinus() + hMaxExtents.jPlus()) + ", (dom_.isize()+ " +
                std::to_string(blockSize[0]) + " - 1) / " + std::to_string(blockSize[0]) +
                ", (dom_.jsize()+ " + std::to_string(blockSize[1]) + " - 1) / " +
                std::to_string(blockSize[1]) + ", dom_.ksize() + 2 * " +
                std::to_string(getVerticalTmpHaloSize(stencil)) + ")");
    for(const auto& fieldPair : tempFields) {
      ctr.addInit("m_" + fieldPair.second.Name + "(" + tmpMetadataName_ + ")");
    }
  }
}

std::unique_ptr<TranslationUnit> CudaCodeGen::generateCode() {
  DAWN_LOG(INFO) << "Starting code generation for GTClang ...";

  // Generate code for StencilInstantiations
  std::map<std::string, std::string> stencils;
  for(const auto& nameStencilCtxPair : context_) {
    std::shared_ptr<iir::StencilInstantiation> origSI = nameStencilCtxPair.second;
    // TODO the clone seems to be broken
    //    std::shared_ptr<iir::StencilInstantiation> stencilInstantiation = origSI->clone();
    std::shared_ptr<iir::StencilInstantiation> stencilInstantiation = origSI;

    std::string code = generateStencilInstantiation(stencilInstantiation);
    if(code.empty())
      return nullptr;
    stencils.emplace(nameStencilCtxPair.first, std::move(code));
  }

  std::string globals = generateGlobals(context_, "dawn_generated", "cuda");

  std::vector<std::string> ppDefines;
  auto makeDefine = [](std::string define, int value) {
    return "#define " + define + " " + std::to_string(value);
  };

  ppDefines.push_back(makeDefine("DAWN_GENERATED", 1));
  ppDefines.push_back("#undef DAWN_BACKEND_T");
  ppDefines.push_back("#define DAWN_BACKEND_T CUDA");
  //==============------------------------------------------------------------------------------===
  // BENCHMARKTODO: since we're importing two cpp files into the benchmark API we need to set
  // these
  // variables also in the naive code-generation in order to not break it. Once the move to
  // different TU's is completed, this is no longer necessary.
  // [https://github.com/MeteoSwiss-APN/gtclang/issues/32]
  //==============------------------------------------------------------------------------------===
  CodeGen::addMplIfdefs(ppDefines, 30);
  ppDefines.push_back("#include <driver-includes/gridtools_includes.hpp>");
  ppDefines.push_back("using namespace gridtools::dawn;");

  generateBCHeaders(ppDefines);

  DAWN_LOG(INFO) << "Done generating code";

  std::string filename = generateFileName(context_);
  // TODO missing the BC
  return std::make_unique<TranslationUnit>(filename, std::move(ppDefines), std::move(stencils),
                                           std::move(globals));
}

} // namespace cuda
} // namespace codegen
} // namespace dawn
