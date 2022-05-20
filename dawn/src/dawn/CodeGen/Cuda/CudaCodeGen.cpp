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
#include "dawn/CodeGen/Cuda/ASTStencilDesc.h"
#include "dawn/CodeGen/Cuda/CacheProperties.h"
#include "dawn/CodeGen/Cuda/CodeGeneratorHelper.h"
#include "dawn/CodeGen/Cuda/MSCodeGen.h"
#include "dawn/CodeGen/F90Util.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Array.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/FileSystem.h"
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
                 domain_size, options.OutputCHeader, options.OutputFortranInterface,
                 options.RunWithSync);

  return CG.generateCode();
}

std::vector<int> getUsedFields(const dawn::iir::Stencil& stencil,
                               std::unordered_set<dawn::iir::Field::IntendKind> intend = {
                                   dawn::iir::Field::IntendKind::Output,
                                   dawn::iir::Field::IntendKind::InputOutput,
                                   dawn::iir::Field::IntendKind::Input}) {
  const auto& APIFields = stencil.getMetadata().getAPIFields();
  const auto& stenFields = stencil.getOrderedFields();
  auto usedAPIFields =
      dawn::makeRange(APIFields, [&stenFields](int f) { return stenFields.count(f); });

  std::vector<int> res;
  for(auto fieldID : usedAPIFields) {
    auto field = stenFields.at(fieldID);
    if(intend.count(field.field.getIntend())) {
      res.push_back(fieldID);
    }
  }

  return res;
}
std::vector<std::string> getGlobalsNames(const dawn::ast::GlobalVariableMap& globalsMap) {
  std::vector<std::string> globalsNames;
  for(const auto& global : globalsMap) {
    globalsNames.push_back(global.first);
  }
  return globalsNames;
}

CudaCodeGen::CudaCodeGen(const StencilInstantiationContext& ctx, int maxHaloPoints, int nsms,
                         int maxBlocksPerSM, const Array3i& domainSize,
                         std::optional<std::string> outputCHeader,
                         std::optional<std::string> outputFortranInterface, bool runWithSync)
    : CodeGen(ctx, maxHaloPoints), codeGenOptions_{nsms,          maxBlocksPerSM,
                                                   domainSize,    runWithSync,
                                                   outputCHeader, outputFortranInterface} {}

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

void CudaCodeGen::generateAPIRunFunctions(
    std::stringstream& ssSW, const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    CodeGenProperties& codeGenProperties, bool onlyDecl) const {
  const auto& stencils = stencilInstantiation->getStencils();
  const auto& metadata = stencilInstantiation->getMetaData();

  // generate the code for each of the stencils
  for(const auto& stencilPtr : stencils) {
    const auto& stencil = *stencilPtr;        

    std::string stencilName = "stencil_" + std::to_string(stencil.getStencilID());
    auto stencilProperties =
      codeGenProperties.getStencilProperties(StencilContext::SC_Stencil, stencilName);  

    std::string fullyQualitfiedName =
        "dawn_generated::cuda::" + stencilInstantiation->getName() + "::" + stencilName;

    MemberFunction runFun("void", "run_" + stencilInstantiation->getName(), ssSW, 0, onlyDecl);

    const auto stencilFields = stencil.getOrderedFields();

    auto nonTempFields =
        makeRange(stencilFields, [](std::pair<int, iir::Stencil::FieldInfo> const& p) {
          return !p.second.IsTemporary;
        });

    for(auto field : nonTempFields) {
      runFun.addArg("double *" + field.second.Name + "_ptr");
    }
    runFun.finishArgs();

    if(!onlyDecl) {
      runFun.addStatement("static int iter = 0");
      runFun.addStatement("int ni = " + fullyQualitfiedName + "::m_dom.isize()");
      runFun.addStatement("int nj = " + fullyQualitfiedName + "::m_dom.jsize()");
      runFun.addStatement("int nk = " + fullyQualitfiedName + "::m_dom.ksize()");

      runFun.addStatement("meta_data_t meta_data_ijk({ni, nj, nk}, {1, ni, ni*nj})");
      runFun.addStatement("meta_data_ij_t meta_data_ij({ni, nj, 1}, {1, ni, 0})");
      runFun.addStatement("meta_data_k_t meta_data_k({1, 1, nk}, {1, 0, 0})");
      
      for(auto field : nonTempFields) {
        runFun.addStatement(stencilProperties->paramNameToType_.at(field.second.Name) + " " + field.second.Name +
                            "(meta_data_" + getStorageType(field.second.field.getFieldDimensions(), "", "") + ", " + field.second.Name + "_ptr, gridtools::ownership::external_gpu)");
      }
      {
        std::string fields;
        std::string sep = "";
        for(auto field : nonTempFields) {
          fields += sep + field.second.Name;
          sep = ", ";
        }
        runFun.addStatement(fullyQualitfiedName + "::run(" + fields + ")");
        runFun.addPreprocessorDirective("ifdef __DSL_SERIALIZE");
        auto outFields = getUsedFields(stencil, {dawn::iir::Field::IntendKind::Output, dawn::iir::Field::IntendKind::InputOutput});
        for (auto outField : outFields) {
          auto fname = metadata.getFieldNameFromAccessID(outField);
          runFun.addStatement("serialize_gpu(" + fname + ", \"" + stencilName + "_" + fname + "\", iter, ni, nj, nk)");
        }
        runFun.addPreprocessorDirective("endif");
        runFun.addStatement("iter++");
      }
    }
    runFun.commit();
  }
}

void CudaCodeGen::generateSetupFunctions(
    std::stringstream& ssSW, const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    CodeGenProperties& codeGenProperties, bool onlyDecl) const {
  const auto& stencils = stencilInstantiation->getStencils();

  // generate the code for each of the stencils
  for(const auto& stencilPtr : stencils) {
    const auto& stencil = *stencilPtr;

    std::string stencilName = "stencil_" + std::to_string(stencil.getStencilID());

    std::string fullyQualitfiedName =
        "dawn_generated::cuda::" + stencilInstantiation->getName() + "::" + stencilName;
    MemberFunction setupFun("void", "setup_" + stencilInstantiation->getName(), ssSW, 0, onlyDecl);
    setupFun.addArg("int i");
    setupFun.addArg("int j");
    setupFun.addArg("int k");
    setupFun.finishArgs();
    if(!onlyDecl) {
      setupFun.addStatement(fullyQualitfiedName +
                            "::setup(gridtools::dawn::domain(i, j, k), 1, 1, 1)");
    }
    setupFun.commit();
  }
}

void CudaCodeGen::generateStaticMembersTrailer(
    std::stringstream& ssSW, const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    CodeGenProperties& codeGenProperties) const {

  const auto& stencils = stencilInstantiation->getStencils();

  // generate the code for each of the stencils
  for(const auto& stencilPtr : stencils) {
    const auto& stencil = *stencilPtr;

    std::string stencilName = "stencil_" + std::to_string(stencil.getStencilID());

    std::string fullyQualitfiedName =
        "dawn_generated::cuda::" + stencilInstantiation->getName() + "::" + stencilName;

    ssSW << "gridtools::dawn::domain " + fullyQualitfiedName +
                "::m_dom = gridtools::dawn::domain(1, 1, 1);";

    if(stencil.isEmpty())
      continue;

    // fields used in the stencil
    const auto stencilFields = stencil.getOrderedFields();

    auto tempFields =
        makeRange(stencilFields, [](std::pair<int, iir::Stencil::FieldInfo> const& p) {
          return p.second.IsTemporary;
        });

    if(!(tempFields.empty())) {
      ssSW << fullyQualitfiedName + "::tmp_meta_data_t " + fullyQualitfiedName +
                "::m_tmp_meta_data(1, 1, 1, 1, 1);";
      for(const auto& fieldPair : tempFields) {
        ssSW << fullyQualitfiedName
             << "::tmp_storage_t " + fullyQualitfiedName + "::" + "m_" + fieldPair.second.Name + ";";
      }
    }

    std::string iterators = "ij";
    for(auto& stage : iterateIIROver<iir::Stage>(stencil)) {
      int index = 0;
      for(const auto& interval : stage->getIterationSpace()) {
        if(interval.has_value()) {
          std::string iterator = iterators.substr(index, 1);
          std::string arrName = "stage" + std::to_string(stage->getStageID()) + "Global" +
                                (char)std::toupper(iterator.at(0)) + "Indices";
          ssSW << "std::array<int, 2> " << fullyQualitfiedName + "::" + arrName + ";";
          index += 1;
        }
      }
    }

    if(iterationSpaceSet_) {
      ssSW << "std::array<unsigned int, 2> " << fullyQualitfiedName + "::globalOffsets;";
    }
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
    generateGlobalsAPI(stencilWrapperClass, globalsMap, codeGenProperties);
  }

  generateStencilWrapperSyncMethod(stencilWrapperClass);

  generateStencilWrapperRun(stencilWrapperClass, stencilInstantiation, codeGenProperties);

  generateStencilWrapperPublicMemberFunctions(stencilWrapperClass, codeGenProperties);

  stencilWrapperClass.commit();

  cudaNamespace.commit();
  dawnNamespace.commit();

  ssSW << "extern \"C\" {\n";
  generateAPIRunFunctions(ssSW, stencilInstantiation, codeGenProperties);
  generateSetupFunctions(ssSW, stencilInstantiation, codeGenProperties);
  ssSW << "}\n";
  generateStaticMembersTrailer(ssSW, stencilInstantiation, codeGenProperties);

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
      paramNameToType.emplace(fieldPair.second.Name, c_dgt + "storage_t");
    }

    iterationSpaceSet_ = hasGlobalIndices(stencil);
    generateStencilClassMembers(stencilClass, stencil, globalsMap, nonTempFields, tempFields,
                                stencilProperties);

    stencilClass.changeAccessibility("public");

    generateStencilClassCtr(stencilClass, stencil, globalsMap, nonTempFields, tempFields,
                            stencilProperties);

    generateStencilSetupMethod(stencilClass, stencil, globalsMap, nonTempFields, tempFields,
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
    Structure& stencilClass, const iir::Stencil& stencil, const ast::GlobalVariableMap& globalsMap,
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

  stencilClass.addMember("static " + c_dgt + "domain", "m_dom");

  if(!tempFields.empty()) {
    stencilClass.addComment("temporary storage declarations");
    addTmpStorageDeclaration(stencilClass, tempFields);
  }
}

void CudaCodeGen::generateStencilClassCtr(
    Structure& stencilClass, const iir::Stencil& stencil, const ast::GlobalVariableMap& globalsMap,
    IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& nonTempFields,
    IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& tempFields,
    std::shared_ptr<StencilProperties> stencilProperties) const {

  auto stencilClassCtr = stencilClass.addConstructor();

  stencilClassCtr.addArg("const " + c_dgt + "domain& dom_");
  if(!globalsMap.empty()) {
    stencilClassCtr.addArg("globals& globals_");
  }
  stencilClassCtr.addArg("int rank");
  stencilClassCtr.addArg("int xcols");
  stencilClassCtr.addArg("int ycols");

  stencilClassCtr.addInit("sbase(\"" + stencilClass.getName() + "\")");
  stencilClassCtr.addStatement("m_dom =dom_");

  if(!globalsMap.empty()) {
    stencilClassCtr.addStatement("m_globals = globals_");
  }

  std::string iterators = "ij";
  for(auto& stage : iterateIIROver<iir::Stage>(stencil)) {
    int index = 0;
    for(const auto& interval : stage->getIterationSpace()) {
      if(interval.has_value()) {
        std::string iterator = iterators.substr(index, 1);
        std::string arrName = "stage" + std::to_string(stage->getStageID()) + "Global" +
                              (char)std::toupper(iterator.at(0)) + "Indices";
        stencilClassCtr.addStatement(
            arrName + " = {" +
            makeIntervalBoundExplicit(iterator, interval.value(), iir::Interval::Bound::lower,
                                      "dom_") +
            " , " +
            makeIntervalBoundExplicit(iterator, interval.value(), iir::Interval::Bound::upper,
                                      "dom_") +
            "}");
      }
      index += 1;
    }
  }

  if(iterationSpaceSet_) {
    stencilClassCtr.addStatement(
        "globalOffsets = {computeGlobalOffsets(rank, m_dom, xcols, ycols)}");

    std::string iterators = "IJ";
    for(auto& stage : iterateIIROver<iir::Stage>(stencil)) {
      for(auto [index, interval] : enumerate(stage->getIterationSpace())) {
        if(interval.has_value()) {
          std::string hostName = "stage" + std::to_string(stage->getStageID()) + "Global" +
                                 iterators.at(index) + "Indices";
          addCudaCopySymbol(stencilClassCtr, hostName, "int");
        }
      }
    }
    addCudaCopySymbol(stencilClassCtr, "globalOffsets", "unsigned");
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
  StencilWrapperConstructor.addArg("const " + c_dgt + "domain& dom");
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
    stencilWrapperClass.addMember(c_dgt + "meta_data_t", "m_meta_data");

    for(int AccessID : metadata.getAccessesOfType<iir::FieldAccessType::InterStencilTemporary>())
      stencilWrapperClass.addMember(
          c_dgt + "storage_t",
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

  for(const auto& fieldID : metadata.getAPIFields()) {
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

void CudaCodeGen::generateStencilSetupMethod(
    Structure& stencilClass, const iir::Stencil& stencil, const ast::GlobalVariableMap& globalsMap,
    IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& nonTempFields,
    IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& tempFields,
    std::shared_ptr<StencilProperties> stencilProperties) const {

  auto stencilClassSetup = stencilClass.addMemberFunction("static void", "setup");

  stencilClassSetup.addArg("const " + c_dgt + "domain& dom_");
  if(!globalsMap.empty()) {
    stencilClassSetup.addArg("globals& globals_");
  }
  stencilClassSetup.addArg("int rank");
  stencilClassSetup.addArg("int xcols");
  stencilClassSetup.addArg("int ycols");

  stencilClassSetup.addStatement("m_dom =dom_");

  if(!globalsMap.empty()) {
    stencilClassSetup.addStatement("m_globals = globals_");
  }

  std::string iterators = "ij";
  for(auto& stage : iterateIIROver<iir::Stage>(stencil)) {
    int index = 0;
    for(const auto& interval : stage->getIterationSpace()) {
      if(interval.has_value()) {
        std::string iterator = iterators.substr(index, 1);
        std::string arrName = "stage" + std::to_string(stage->getStageID()) + "Global" +
                              (char)std::toupper(iterator.at(0)) + "Indices";
        stencilClassSetup.addStatement(
            arrName + " = {" +
            makeIntervalBoundExplicit(iterator, interval.value(), iir::Interval::Bound::lower,
                                      "dom_") +
            " , " +
            makeIntervalBoundExplicit(iterator, interval.value(), iir::Interval::Bound::upper,
                                      "dom_") +
            "}");
      }
      index += 1;
    }
  }

  if(iterationSpaceSet_) {
    stencilClassSetup.addStatement(
        "globalOffsets = {computeGlobalOffsets(rank, m_dom, xcols, ycols)}");
  }

  if(iterationSpaceSet_) {
    stencilClassSetup.addStatement(
        "globalOffsets = {computeGlobalOffsets(rank, m_dom, xcols, ycols)}");

    std::string iterators = "IJ";
    for(auto& stage : iterateIIROver<iir::Stage>(stencil)) {
      for(auto [index, interval] : enumerate(stage->getIterationSpace())) {
        if(interval.has_value()) {
          std::string hostName = "stage" + std::to_string(stage->getStageID()) + "Global" +
                                 iterators.at(index) + "Indices";
          addCudaCopySymbol(stencilClassSetup, hostName, "int");
        }
      }
    }
    addCudaCopySymbol(stencilClassSetup, "globalOffsets", "unsigned");
  }

  addTmpStorageInit(stencilClassSetup, stencil, tempFields);
  stencilClassSetup.commit();
}

void CudaCodeGen::generateStencilRunMethod(
    Structure& stencilClass, const iir::Stencil& stencil,
    const std::shared_ptr<StencilProperties>& stencilProperties,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const std::unordered_map<std::string, std::string>& paramNameToType,
    const ast::GlobalVariableMap& globalsMap) const {
  MemberFunction stencilRunMethod = stencilClass.addMemberFunction("static void", "run", "");
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
  stencilRunMethod.addComment("start()");

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
      stencilRunMethod.addStatement(c_gt + "data_view<" + paramNameToType.at(fieldName) + "> " +
                                    fieldName + "= " + c_gt + "make_device_view(" + fieldName +
                                    "_ds)");
    }

    for(const auto& fieldPair : tempMSFieldsNonLocalCached) {
      const auto fieldName = metadata.getFieldNameFromAccessID(fieldPair.second.getAccessID());
      stencilRunMethod.addStatement(c_gt + "data_view<tmp_storage_t> " + fieldName + "= " + c_gt +
                                    "make_device_view( m_" + fieldName + ")");
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
#ifdef INDEX_VIA_FIELD_BEGIN_MARKERS
      const std::string i0_str = fieldName + ".begin<0>()";
      const std::string i1_str = fieldName + ".begin<1>()";
#else
      const std::string i0_str = "m_dom.iminus()";
      const std::string i1_str = "m_dom.jminus()";
#endif
      const std::string i2_str = "0";
      const std::string index_str = "index(" + i0_str + "," + i1_str + "," + i2_str + ")";
      args += "(" + fieldName + ".data()+" + fieldName + "_ds.get_storage_info_ptr()->" +
              index_str + ")";
      ++idx;
    }

    if(!args.empty() && !tempMSFieldsNonLocalCached.empty())
      args += ",";

    idx = 0;
    for(const auto& fieldPair : tempMSFieldsNonLocalCached) {
      // in some cases (where there are no horizontal extents) we dont use the special tmp index
      // iterator, but rather a normal 3d field index iterator. In that case we pass temporaries
      // in the same manner as normal fields
      if(idx > 0)
        args += ",";
      if(!CodeGeneratorHelper::useTemporaries(multiStagePtr->getParent(), metadata)) {
        const auto fieldName = metadata.getFieldNameFromAccessID(fieldPair.second.getAccessID());
        args += "(" + fieldName + ".data()+ m_" + fieldName + ".get_storage_info_ptr()->index(" +
                fieldName + ".begin<0>(), " + fieldName + ".begin<1>()," + fieldName +
                ".begin<2>()," + fieldName + ".begin<3>(), 0))";
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
    kernelCall += ")";

    stencilRunMethod.addStatement(kernelCall);
    stencilRunMethod.addStatement("}");
  }

  stencilRunMethod.addComment("stopping timers");
  stencilRunMethod.addComment("pause()");

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
    ctr.addStatement(tmpMetadataName_ + " = tmp_meta_data_t(" + std::to_string(blockSize[0]) + "+" +
                     std::to_string(-hMaxExtents.iMinus() + hMaxExtents.iPlus()) + ", " +
                     std::to_string(blockSize[1]) + "+" +
                     std::to_string(-hMaxExtents.jMinus() + hMaxExtents.jPlus()) +
                     ", (dom_.isize()+ " + std::to_string(blockSize[0]) + " - 1) / " +
                     std::to_string(blockSize[0]) + ", (dom_.jsize()+ " +
                     std::to_string(blockSize[1]) + " - 1) / " + std::to_string(blockSize[1]) +
                     ", dom_.ksize() + 2 * " + std::to_string(getVerticalTmpHaloSize(stencil)) +
                     ")");
    for(const auto& fieldPair : tempFields) {
      ctr.addStatement("m_" + fieldPair.second.Name + " = tmp_storage_t(" + tmpMetadataName_ + ")");
    }
  }
}

void CudaCodeGen::generateCHeaderSI(
    std::stringstream& ssSW,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) const {
  using namespace codegen;

  CodeGenProperties codeGenProperties = computeCodeGenProperties(stencilInstantiation.get());

  ssSW << "extern \"C\" {\n";
  generateAPIRunFunctions(ssSW, stencilInstantiation, codeGenProperties, /*onlyDecl=*/true);
  generateSetupFunctions(ssSW, stencilInstantiation, codeGenProperties, /*onlyDecl=*/true);
  ssSW << "}\n";
}

std::string CudaCodeGen::generateCHeader() const {
  std::stringstream ssSW;
  ssSW << "#pragma once\n";
  ssSW << "#include \"driver-includes/defs.hpp\"\n";
  ssSW << "#include \"driver-includes/cuda_utils.hpp\"\n";

  for(const auto& nameStencilCtxPair : context_) {
    std::shared_ptr<iir::StencilInstantiation> stencilInstantiation = nameStencilCtxPair.second;
    generateCHeaderSI(ssSW, stencilInstantiation);
  }

  return ssSW.str();
}

static void
generateF90InterfaceSI(FortranInterfaceModuleGen& fimGen,
                       const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  const auto& stencils = stencilInstantiation->getStencils();
  const auto& globalsMap = stencilInstantiation->getIIR()->getGlobalVariableMap();
  auto globalTypeToFortType = [](const ast::Global& global) {
    switch(global.getType()) {
    case ast::Value::Kind::Boolean:
      return FortranAPI::InterfaceType::BOOLEAN;
    case ast::Value::Kind::Double:
      return FortranAPI::InterfaceType::DOUBLE;
    case ast::Value::Kind::Float:
      return FortranAPI::InterfaceType::FLOAT;
    case ast::Value::Kind::Integer:
      return FortranAPI::InterfaceType::INTEGER;
    case ast::Value::Kind::String:
    default:
      throw std::runtime_error("string globals not supported in cuda ico backend");
    }
  };

  // The following assert is needed because we have only one (user-defined) name for a stencil
  // instantiation (stencilInstantiation->getName()). We could compute a per-stencil name (
  // codeGenProperties.getStencilName(StencilContext::SC_Stencil, stencil.getStencilID()) )
  // however the interface would not be very useful if the name is generated.
  DAWN_ASSERT_MSG(stencils.size() <= 1,
                  "Unable to generate interface. More than one stencil in stencil instantiation.");
  const auto& stencil = *stencils[0];

  std::vector<FortranInterfaceAPI> interfaces = {
      FortranInterfaceAPI("run_" + stencilInstantiation->getName())};

  auto addArgsToAPI = [&](FortranAPI& api, bool includeSavedState, bool optThresholds) {
    for(const auto& global : globalsMap) {
      api.addArg(global.first, globalTypeToFortType(global.second));
    }
    for(auto fieldID : stencilInstantiation->getMetaData().getAPIFields()) {
      api.addArg(
          stencilInstantiation->getMetaData().getNameFromAccessID(fieldID),
          FortranAPI::InterfaceType::DOUBLE /* Unfortunately we need to know at codegen
                                                        time whether we have fields in SP/DP */
          ,
          stencilInstantiation->getMetaData().getFieldDimensions(fieldID).rank());
    }
    if(includeSavedState) {
      for(auto fieldID : getUsedFields(stencil, {dawn::iir::Field::IntendKind::Output,
                                                 dawn::iir::Field::IntendKind::InputOutput})) {
        api.addArg(
            stencilInstantiation->getMetaData().getNameFromAccessID(fieldID) + "_before",
            FortranAPI::InterfaceType::DOUBLE /* Unfortunately we need to know at codegen
                                                          time whether we have fields in SP/DP */
            ,
            stencilInstantiation->getMetaData().getFieldDimensions(fieldID).rank());
      }

      for(auto fieldID : getUsedFields(stencil, {dawn::iir::Field::IntendKind::Output,
                                                 dawn::iir::Field::IntendKind::InputOutput})) {
        if(optThresholds) {
          api.addOptArg(stencilInstantiation->getMetaData().getNameFromAccessID(fieldID) +
                            "_rel_tol",
                        FortranAPI::InterfaceType::DOUBLE);
          api.addOptArg(stencilInstantiation->getMetaData().getNameFromAccessID(fieldID) +
                            "_abs_tol",
                        FortranAPI::InterfaceType::DOUBLE);
        } else {
          api.addArg(stencilInstantiation->getMetaData().getNameFromAccessID(fieldID) + "_rel_tol",
                     FortranAPI::InterfaceType::DOUBLE);
          api.addArg(stencilInstantiation->getMetaData().getNameFromAccessID(fieldID) + "_abs_tol",
                     FortranAPI::InterfaceType::DOUBLE);
        }
      }
    }
  };

  addArgsToAPI(interfaces[0], /*includeSavedState*/ false, false);
  fimGen.addInterfaceAPI(std::move(interfaces[0]));
  std::string fortranIndent = "   ";

  // memory management functions for production interface
  FortranInterfaceAPI setup("setup_" + stencilInstantiation->getName());
  setup.addArg("i", FortranAPI::InterfaceType::INTEGER);
  setup.addArg("j", FortranAPI::InterfaceType::INTEGER);
  setup.addArg("k", FortranAPI::InterfaceType::INTEGER);
  fimGen.addInterfaceAPI(std::move(setup));
}

std::string CudaCodeGen::generateF90Interface(std::string moduleName) const {
  std::stringstream ss;
  IndentedStringStream iss(ss);

  FortranInterfaceModuleGen fimGen(iss, moduleName);

  for(const auto& nameStencilCtxPair : context_) {
    std::shared_ptr<iir::StencilInstantiation> stencilInstantiation = nameStencilCtxPair.second;
    generateF90InterfaceSI(fimGen, stencilInstantiation);
  }

  fimGen.commit();

  return iss.str();
}

std::unique_ptr<TranslationUnit> CudaCodeGen::generateCode() {
  DAWN_LOG(INFO) << "Starting code generation for GTClang ...";

  if(codeGenOptions_.OutputCHeader) {
    fs::path filePath = *codeGenOptions_.OutputCHeader;
    std::ofstream headerFile;
    headerFile.open(filePath);
    if(headerFile) {
      headerFile << generateCHeader();
      headerFile.close();
    } else {
      throw std::runtime_error("Error writing to " + filePath.string() + ": " + strerror(errno));
    }
  }
  if(codeGenOptions_.OutputFortranInterface) {
    fs::path filePath = *codeGenOptions_.OutputFortranInterface;
    std::string moduleName = filePath.filename().replace_extension("").string();
    std::ofstream interfaceFile;
    interfaceFile.open(filePath);
    if(interfaceFile) {
      interfaceFile << generateF90Interface(moduleName);
      interfaceFile.close();
    } else {
      throw std::runtime_error("Error writing to " + filePath.string() + ": " + strerror(errno));
    }
  }

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
  ppDefines.push_back("#include <driver-includes/timer_cuda.hpp>");
  ppDefines.push_back("#include <driver-includes/gridtools_includes.hpp>");
  ppDefines.push_back("#include <driver-includes/serialize.hpp>");
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
