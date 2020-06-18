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

#include "dawn/CodeGen/CXXNaive/CXXNaiveCodeGen.h"
#include "dawn/AST/GridType.h"
#include "dawn/AST/Offsets.h"
#include "dawn/CodeGen/CXXNaive/ASTStencilBody.h"
#include "dawn/CodeGen/CXXNaive/ASTStencilDesc.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/CodeGen/CodeGenProperties.h"
#include "dawn/IIR/Extents.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/Interval.h"
#include "dawn/IIR/Stage.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Exception.h"
#include "dawn/Support/Logger.h"
#include "dawn/Support/StringUtil.h"
#include <algorithm>
#include <string>
#include <vector>

namespace dawn {
namespace codegen {
namespace cxxnaive {

namespace {
std::string makeLoopImpl(int lowerExtent, int upperExtent, const std::string& dim,
                         const std::string& lower, const std::string& upper,
                         const std::string& comparison, const std::string& increment) {
  return "for(int " + dim + " = " + lower + "+" + std::to_string(lowerExtent) + "; " + dim + " " +
         comparison + " " + upper + "+" + std::to_string(upperExtent) + "; " + increment + dim +
         ")";
}

std::string makeIJLoop(int lowerExtent, int upperExtent, const std::string dom,
                       const std::string& dim) {
  return makeLoopImpl(lowerExtent, upperExtent, dim, dim + "Min", dim + "Max", " <= ", "++");
}

std::string makeIntervalBoundReadable(std::string dim, const iir::Interval& interval,
                                      iir::Interval::Bound bound) {
  if(interval.levelIsEnd(bound)) {
    return dim + "Max + " + std::to_string(interval.offset(bound));
  }
  auto notEnd = interval.level(bound);
  if(notEnd == 0) {
    return dim + "Min + " + std::to_string(interval.offset(bound));
  }
  return dim + "Min + " + std::to_string(notEnd + interval.offset(bound));
}
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

std::string makeKLoop(bool isBackward, iir::Interval const& interval) {

  const std::string lower = makeIntervalBoundReadable("k", interval, iir::Interval::Bound::lower);
  const std::string upper = makeIntervalBoundReadable("k", interval, iir::Interval::Bound::upper);

  return isBackward ? makeLoopImpl(0, 0, "k", upper, lower, ">=", "--")
                    : makeLoopImpl(0, 0, "k", lower, upper, "<=", "++");
}
} // namespace

std::unique_ptr<TranslationUnit>
run(const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>&
        stencilInstantiationMap,
    const Options& options) {
  CXXNaiveCodeGen CG(stencilInstantiationMap, options.MaxHaloSize);

  return CG.generateCode();
}

CXXNaiveCodeGen::CXXNaiveCodeGen(const StencilInstantiationContext& ctx, int maxHaloPoint)
    : CodeGen(ctx, maxHaloPoint) {}

CXXNaiveCodeGen::~CXXNaiveCodeGen() {}

std::string CXXNaiveCodeGen::generateStencilInstantiation(
    const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation) {
  using namespace codegen;

  std::stringstream ssSW;

  Namespace dawnNamespace("dawn_generated", ssSW);
  Namespace cxxnaiveNamespace("cxxnaive", ssSW);

  const auto& globalsMap = stencilInstantiation->getIIR()->getGlobalVariableMap();

  Class stencilWrapperClass(stencilInstantiation->getName(), ssSW);
  stencilWrapperClass.changeAccessibility("private");

  CodeGenProperties codeGenProperties = computeCodeGenProperties(stencilInstantiation.get());

  generateStencilFunctions(stencilWrapperClass, stencilInstantiation, codeGenProperties);

  generateStencilClasses(stencilInstantiation, stencilWrapperClass, codeGenProperties);

  generateStencilWrapperMembers(stencilWrapperClass, stencilInstantiation, codeGenProperties);

  generateStencilWrapperCtr(stencilWrapperClass, stencilInstantiation, codeGenProperties);

  generateGlobalsAPI(*stencilInstantiation, stencilWrapperClass, globalsMap, codeGenProperties);

  generateStencilWrapperRun(stencilWrapperClass, stencilInstantiation, codeGenProperties);

  stencilWrapperClass.commit();

  cxxnaiveNamespace.commit();
  dawnNamespace.commit();

  return ssSW.str();
}

void CXXNaiveCodeGen::generateStencilWrapperRun(
    Class& stencilWrapperClass,
    const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation,
    const CodeGenProperties& codeGenProperties) const {

  const auto& metadata = stencilInstantiation->getMetaData();

  // Generate the run method by generate code for the stencil description AST
  MemberFunction runMethod = stencilWrapperClass.addMemberFunction("void", "run", "");

  for(const auto& fieldID : metadata.getAccessesOfType<iir::FieldAccessType::APIField>()) {
    std::string name = metadata.getFieldNameFromAccessID(fieldID);
    runMethod.addArg(codeGenProperties.getParamType(stencilInstantiation, name) + " " + name);
  }

  runMethod.finishArgs();

  // generate the control flow code executing each inner stencil
  ASTStencilDesc stencilDescCGVisitor(stencilInstantiation, codeGenProperties);
  stencilDescCGVisitor.setIndent(runMethod.getIndent());
  for(const auto& statement :
      stencilInstantiation->getIIR()->getControlFlowDescriptor().getStatements()) {
    statement->accept(stencilDescCGVisitor);
    auto str = stencilDescCGVisitor.getCodeAndResetStream();
    if(str.back() == ';')
      str.pop_back();
    runMethod.addStatement(str);
  }

  runMethod.commit();
}
void CXXNaiveCodeGen::generateStencilWrapperCtr(
    Class& stencilWrapperClass,
    const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation,
    const CodeGenProperties& codeGenProperties) const {

  const auto& stencils = stencilInstantiation->getStencils();
  const auto& metadata = stencilInstantiation->getMetaData();
  const auto& globalsMap = stencilInstantiation->getIIR()->getGlobalVariableMap();

  // Generate stencil wrapper constructor
  auto StencilWrapperConstructor = stencilWrapperClass.addConstructor();

  StencilWrapperConstructor.addArg("const " + c_dgt() + "domain& dom");
  StencilWrapperConstructor.addArg("int rank = 1");
  StencilWrapperConstructor.addArg("int xcols = 1");
  StencilWrapperConstructor.addArg("int ycols = 1");

  // add the ctr initialization of each stencil
  for(const auto& stencilPtr : stencils) {
    iir::Stencil& stencil = *stencilPtr;
    if(stencil.isEmpty())
      continue;

    const auto stencilFields = stencil.getOrderedFields();

    const std::string stencilName =
        codeGenProperties.getStencilName(StencilContext::SC_Stencil, stencil.getStencilID());

    std::string initCtr = "m_" + stencilName + "(dom";
    if(!globalsMap.empty()) {
      initCtr += ", m_globals";
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
  StencilWrapperConstructor.startBody();
  StencilWrapperConstructor.addStatement("assert(dom.isize() >= dom.iminus() + dom.iplus())");
  StencilWrapperConstructor.addStatement("assert(dom.jsize() >= dom.jminus() + dom.jplus())");
  StencilWrapperConstructor.addStatement("assert(dom.ksize() >= dom.kminus() + dom.kplus())");
  StencilWrapperConstructor.addStatement("assert(dom.ksize() >= 1)");
  StencilWrapperConstructor.commit();

  StencilWrapperConstructor.commit();
}
void CXXNaiveCodeGen::generateStencilWrapperMembers(
    Class& stencilWrapperClass,
    const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation,
    CodeGenProperties& codeGenProperties) const {

  const auto& metadata = stencilInstantiation->getMetaData();
  const auto& globalsMap = stencilInstantiation->getIIR()->getGlobalVariableMap();

  stencilWrapperClass.addMember("static constexpr const char* s_name =",
                                "\"" + stencilWrapperClass.getName() + "\"");

  if(!globalsMap.empty()) {
    stencilWrapperClass.addMember("globals", "m_globals");
  }

  for(auto stencilPropertiesPair :
      codeGenProperties.stencilProperties(StencilContext::SC_Stencil)) {
    stencilWrapperClass.addMember(stencilPropertiesPair.second->name_,
                                  "m_" + stencilPropertiesPair.second->name_);
  }

  stencilWrapperClass.changeAccessibility("public");
  stencilWrapperClass.addCopyConstructor(Class::ConstructorDefaultKind::Deleted);
  //
  // Members
  //
  // Define allocated memebers if necessary
  if(metadata.hasAccessesOfType<iir::FieldAccessType::InterStencilTemporary>()) {
    stencilWrapperClass.addComment("Members");

    stencilWrapperClass.addMember(c_dgt() + "meta_data_t", "m_meta_data");

    for(int AccessID : metadata.getAccessesOfType<iir::FieldAccessType::InterStencilTemporary>())
      stencilWrapperClass.addMember(c_dgt() + "storage_t",
                                    "m_" + metadata.getFieldNameFromAccessID(AccessID));
  }
}
void CXXNaiveCodeGen::generateStencilClasses(
    const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation,
    Class& stencilWrapperClass, const CodeGenProperties& codeGenProperties) const {

  const auto& stencils = stencilInstantiation->getStencils();
  const auto& globalsMap = stencilInstantiation->getIIR()->getGlobalVariableMap();

  // Stencil members:
  // generate the code for each of the stencils
  for(std::size_t stencilIdx = 0; stencilIdx < stencils.size(); ++stencilIdx) {
    const auto& stencil = *stencils[stencilIdx];

    std::string stencilName =
        codeGenProperties.getStencilName(StencilContext::SC_Stencil, stencil.getStencilID());

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

    Structure stencilClass = stencilWrapperClass.addStruct(stencilName);

    ASTStencilBody stencilBodyCXXVisitor(stencilInstantiation->getMetaData(),
                                         StencilContext::SC_Stencil);

    stencilClass.addComment("Members");
    bool iterationSpaceSet = hasGlobalIndices(stencil);
    if(iterationSpaceSet) {
      generateGlobalIndices(stencil, stencilClass);
    }

    stencilClass.addComment("Temporary storages");
    addTempStorageTypedef(stencilClass, stencil);

    stencilClass.addMember("const " + c_dgt() + "domain", "m_dom");

    if(!globalsMap.empty()) {
      stencilClass.addMember("const globals&", "m_globals");
    }

    stencilClass.addComment("Input/Output storages");

    addTmpStorageDeclaration(stencilClass, tempFields);

    stencilClass.changeAccessibility("public");

    auto stencilClassCtr = stencilClass.addConstructor();

    stencilClassCtr.addArg("const " + c_dgt() + "domain& dom_");
    if(!globalsMap.empty()) {
      stencilClassCtr.addArg("const globals& globals_");
    }
    stencilClassCtr.addArg("int rank");
    stencilClassCtr.addArg("int xcols");
    stencilClassCtr.addArg("int ycols");

    stencilClassCtr.addInit("m_dom(dom_)");
    if(!globalsMap.empty()) {
      stencilClassCtr.addArg("m_globals(globals_)");
    }
    for(auto& stage : iterateIIROver<iir::Stage>(stencil)) {
      if(stage->getIterationSpace()[0].has_value()) {
        stencilClassCtr.addInit(
            "stage" + std::to_string(stage->getStageID()) + "GlobalIIndices({" +
            makeIntervalBoundExplicit("i", stage->getIterationSpace()[0].value(),
                                      iir::Interval::Bound::lower, "dom_") +
            " , " +
            makeIntervalBoundExplicit("i", stage->getIterationSpace()[0].value(),
                                      iir::Interval::Bound::upper, "dom_") +
            "})");
      }
      if(stage->getIterationSpace()[1].has_value()) {
        stencilClassCtr.addInit(
            "stage" + std::to_string(stage->getStageID()) + "GlobalJIndices({" +
            makeIntervalBoundExplicit("j", stage->getIterationSpace()[1].value(),
                                      iir::Interval::Bound::lower, "dom_") +
            " , " +
            makeIntervalBoundExplicit("j", stage->getIterationSpace()[1].value(),
                                      iir::Interval::Bound::upper, "dom_") +
            "})");
      }
    }

    if(iterationSpaceSet) {
      stencilClassCtr.addInit("globalOffsets({computeGlobalOffsets(rank, m_dom, xcols, ycols)})");
    }

    addTmpStorageInit(stencilClassCtr, stencil, tempFields);
    stencilClassCtr.commit();

    // virtual dtor

    // synchronize storages method

    // accumulated extents of API fields
    generateFieldExtentsInfo(stencilClass, nonTempFields, ast::GridType::Cartesian);

    //
    // Run-Method
    //
    MemberFunction stencilRunMethod = stencilClass.addMemberFunction("void", "run", "");
    for(auto it = nonTempFields.begin(); it != nonTempFields.end(); ++it) {
      std::string type = stencilProperties->paramNameToType_.at((*it).second.Name);
      stencilRunMethod.addArg(type + "& " + (*it).second.Name + "_");
    }

    stencilRunMethod.startBody();
    // Compute the loop bounds for readability
    stencilRunMethod.addStatement("int iMin = m_dom.iminus()");
    stencilRunMethod.addStatement("int iMax = m_dom.isize() - m_dom.iplus() - 1");
    stencilRunMethod.addStatement("int jMin = m_dom.jminus()");
    stencilRunMethod.addStatement("int jMax = m_dom.jsize() - m_dom.jplus() - 1");
    stencilRunMethod.addStatement("int kMin = m_dom.kminus()");
    stencilRunMethod.addStatement("int kMax = m_dom.ksize() - m_dom.kplus() - 1");

    for(const auto& fieldPair : nonTempFields) {
      stencilRunMethod.addStatement(fieldPair.second.Name + "_" + ".sync()");
    }
    for(const auto& multiStagePtr : stencil.getChildren()) {

      stencilRunMethod.ss() << "{";

      const iir::MultiStage& multiStage = *multiStagePtr;

      // create all the data views
      for(auto it = nonTempFields.begin(); it != nonTempFields.end(); ++it) {
        const auto fieldName = (*it).second.Name;
        std::string type = stencilProperties->paramNameToType_.at(fieldName);
        stencilRunMethod.addStatement(c_gt() + "data_view<" + type + "> " + fieldName + "= " +
                                      c_gt() + "make_host_view(" + fieldName + "_)");
        stencilRunMethod.addStatement("std::array<int,3> " + fieldName + "_offsets{0,0,0}");
      }
      for(const auto& fieldPair : tempFields) {
        const auto fieldName = fieldPair.second.Name;
        stencilRunMethod.addStatement(c_gt() + "data_view<tmp_storage_t> " + fieldName + "= " +
                                      c_gt() + "make_host_view(m_" + fieldName + ")");
        stencilRunMethod.addStatement("std::array<int,3> " + fieldName + "_offsets{0,0,0}");
      }

      auto intervals_set = multiStage.getIntervals();
      std::vector<iir::Interval> intervals_v;
      std::copy(intervals_set.begin(), intervals_set.end(), std::back_inserter(intervals_v));

      // compute the partition of the intervals
      auto partitionIntervals = iir::Interval::computePartition(intervals_v);
      if((multiStage.getLoopOrder() == iir::LoopOrderKind::Backward))
        std::reverse(partitionIntervals.begin(), partitionIntervals.end());

      for(auto interval : partitionIntervals) {

        // for each interval, we generate naive nested loops
        stencilRunMethod.addBlockStatement(
            makeKLoop((multiStage.getLoopOrder() == iir::LoopOrderKind::Backward), interval),
            [&]() {
              for(const auto& stagePtr : multiStage.getChildren()) {
                iir::Stage& stage = *stagePtr;

                auto const& extents = iir::extent_cast<iir::CartesianExtent const&>(
                    stage.getExtents().horizontalExtent());

                // Check if we need to execute this statement:
                bool hasOverlappingInterval = false;
                for(const auto& doMethodPtr : stage.getChildren()) {
                  hasOverlappingInterval |= (doMethodPtr->getInterval().overlaps(interval));
                }

                if(hasOverlappingInterval) {
                  auto doMethodGenerator = [&]() {
                    // Generate Do-Method
                    for(const auto& doMethodPtr : stage.getChildren()) {
                      const iir::DoMethod& doMethod = *doMethodPtr;
                      if(!doMethod.getInterval().overlaps(interval))
                        continue;
                      for(const auto& stmt : doMethod.getAST().getStatements()) {
                        stmt->accept(stencilBodyCXXVisitor);
                        stencilRunMethod << stencilBodyCXXVisitor.getCodeAndResetStream();
                      }
                    }
                  };

                  stencilRunMethod.addBlockStatement(
                      makeIJLoop(extents.iMinus(), extents.iPlus(), "m_dom", "i"), [&]() {
                        stencilRunMethod.addBlockStatement(
                            makeIJLoop(extents.jMinus(), extents.jPlus(), "m_dom", "j"), [&] {
                              if(std::any_of(stage.getIterationSpace().cbegin(),
                                             stage.getIterationSpace().cend(),
                                             [](const auto& p) -> bool { return p.has_value(); })) {
                                std::string conditional = "if(";
                                if(stage.getIterationSpace()[0]) {
                                  conditional += "checkOffset(stage" +
                                                 std::to_string(stage.getStageID()) +
                                                 "GlobalIIndices[0], stage" +
                                                 std::to_string(stage.getStageID()) +
                                                 "GlobalIIndices[1], globalOffsets[0] + i)";
                                }
                                if(stage.getIterationSpace()[1]) {
                                  if(stage.getIterationSpace()[0]) {
                                    conditional += " && ";
                                  }
                                  conditional += "checkOffset(stage" +
                                                 std::to_string(stage.getStageID()) +
                                                 "GlobalJIndices[0], stage" +
                                                 std::to_string(stage.getStageID()) +
                                                 "GlobalJIndices[1], globalOffsets[1] + j)";
                                }
                                conditional += ")";
                                stencilRunMethod.addBlockStatement(conditional, doMethodGenerator);
                              } else {
                                doMethodGenerator();
                              }
                            });
                      });
                }
              }
            });
      }
      stencilRunMethod.ss() << "}";
    }
    for(const auto& fieldPair : nonTempFields) {
      stencilRunMethod.addStatement(fieldPair.second.Name + "_" + ".sync()");
    }
    stencilRunMethod.commit();
  }
}

void CXXNaiveCodeGen::generateStencilFunctions(
    Class& stencilWrapperClass,
    const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation,
    const CodeGenProperties& codeGenProperties) const {

  const auto& metadata = stencilInstantiation->getMetaData();
  // stencil functions
  //
  // Generate stencil functions code for stencils instantiated by this stencil
  //
  std::unordered_set<std::string> generatedStencilFun;
  size_t idx = 0;
  for(const auto& stencilFun : metadata.getStencilFunctionInstantiations()) {
    std::string stencilFunName = iir::StencilFunctionInstantiation::makeCodeGenName(*stencilFun);
    if(generatedStencilFun.emplace(stencilFunName).second) {

      // Field declaration
      const auto& fields = stencilFun->getCalleeFields();

      if(fields.empty()) {
        throw SemanticError(std::string("No storages referenced in stencil '") +
                                stencilInstantiation->getName() +
                                "', this would result in invalid gridtools code",
                            stencilInstantiation->getMetaData().getFileName(),
                            stencilInstantiation->getMetaData().getStencilLocation());
      }

      // list of template names of the stencil function declaration
      std::vector<std::string> stencilFnTemplates(fields.size());
      // TODO move to capture initialization with C++14
      int n = 0;
      std::generate(stencilFnTemplates.begin(), stencilFnTemplates.end(),
                    [n]() mutable { return "StorageType" + std::to_string(n++); });

      MemberFunction stencilFunMethod = stencilWrapperClass.addMemberFunction(
          std::string("static ") + (stencilFun->hasReturn() ? "double" : "void"), stencilFunName,
          RangeToString(", ", "", "")(stencilFnTemplates,
                                      [](const std::string& str) { return "class " + str; }));

      if(fields.empty() && !stencilFun->hasReturn()) {
        throw SemanticError(std::string("No storages referenced in stencil function '") +
                                stencilInstantiation->getName() +
                                "', this would result in invalid gridtools code",
                            stencilInstantiation->getMetaData().getFileName(),
                            stencilFun->getStencilFunction()->Loc);
      }

      // Each stencil function call will pass the (i,j,k) position
      stencilFunMethod.addArg("const int i");
      stencilFunMethod.addArg("const int j");
      stencilFunMethod.addArg("const int k");

      const auto& stencilProp = codeGenProperties.getStencilProperties(
          StencilContext::SC_StencilFunction, stencilFunName);

      // We need to generate the arguments in order (of the fn call expr)
      for(const auto& exprArg : stencilFun->getArguments()) {
        if(exprArg->Kind != sir::StencilFunctionArg::ArgumentKind::Field)
          continue;
        const std::string argName = exprArg->Name;

        DAWN_ASSERT(stencilProp->paramNameToType_.count(argName));
        const std::string argType = stencilProp->paramNameToType_[argName];
        // each parameter being passed to a stencil function, is wrapped around the param_wrapper
        // that contains the storage and the offset, in order to resolve offset passed to the
        // storage during the function call. For example:
        // fn_call(v(i+1), v(j-1))
        stencilFunMethod.addArg("param_wrapper<" + c_gt() + "data_view<" + argType + ">> pw_" +
                                argName);
      }

      // add global parameter
      if(stencilFun->hasGlobalVariables()) {
        stencilFunMethod.addArg("const globals& m_globals");
      }
      ASTStencilBody stencilBodyCXXVisitor(stencilInstantiation->getMetaData(),
                                           StencilContext::SC_StencilFunction);

      stencilFunMethod.startBody();

      for(std::size_t m = 0; m < fields.size(); ++m) {

        std::string paramName =
            stencilFun->getOriginalNameFromCallerAccessID(fields[m].getAccessID());

        stencilFunMethod << c_gt() << "data_view<StorageType" + std::to_string(m) + "> "
                         << paramName << " = pw_" << paramName << ".dview_;";
        stencilFunMethod << "auto " << paramName << "_offsets = pw_" << paramName << ".offsets_;";
      }
      stencilBodyCXXVisitor.setCurrentStencilFunction(stencilFun);
      stencilBodyCXXVisitor.setIndent(stencilFunMethod.getIndent());
      for(const auto& stmt : stencilFun->getStatements()) {
        stmt->accept(stencilBodyCXXVisitor);
        stencilFunMethod.indentStatment();
        stencilFunMethod << stencilBodyCXXVisitor.getCodeAndResetStream();
      }

      stencilFunMethod.commit();
    }
    idx++;
  }
}

std::unique_ptr<TranslationUnit> CXXNaiveCodeGen::generateCode() {
  DAWN_LOG(INFO) << "Starting code generation for GTClang ...";

  // Generate code for StencilInstantiations
  std::map<std::string, std::string> stencils;
  for(const auto& nameStencilCtxPair : context_) {
    std::string code = generateStencilInstantiation(nameStencilCtxPair.second);
    if(code.empty())
      return nullptr;
    stencils.emplace(nameStencilCtxPair.first, std::move(code));
  }

  std::string globals = generateGlobals(context_, "dawn_generated", "cxxnaive");

  std::vector<std::string> ppDefines;
  auto makeDefine = [](std::string define, int value) {
    return "#define " + define + " " + std::to_string(value);
  };

  ppDefines.push_back(makeDefine("DAWN_GENERATED", 1));
  ppDefines.push_back("#undef DAWN_BACKEND_T");
  ppDefines.push_back("#define DAWN_BACKEND_T CXXNAIVE");
  // ==============------------------------------------------------------------------------------===
  // BENCHMARKTODO: since we're importing two cpp files into the benchmark API we need to set
  // these variables also in the naive code-generation in order to not break it. Once the move to
  // different TU's is completed, this is no longer necessary.
  // [https://github.com/MeteoSwiss-APN/gtclang/issues/32]
  // ==============------------------------------------------------------------------------------===
  CodeGen::addMplIfdefs(ppDefines, 30);
  ppDefines.push_back("#include <driver-includes/gridtools_includes.hpp>");
  ppDefines.push_back("using namespace gridtools::dawn;");
  DAWN_LOG(INFO) << "Done generating code";

  std::string filename = generateFileName(context_);
  return std::make_unique<TranslationUnit>(filename, std::move(ppDefines), std::move(stencils),
                                           std::move(globals));
}

} // namespace cxxnaive
} // namespace codegen
} // namespace dawn
