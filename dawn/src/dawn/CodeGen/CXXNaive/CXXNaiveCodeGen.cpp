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
#include "dawn/CodeGen/CXXNaive/ASTStencilBody.h"
#include "dawn/CodeGen/CXXNaive/ASTStencilDesc.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/CodeGen/CodeGenProperties.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/StringUtil.h"
#include <algorithm>
#include <vector>

namespace dawn {
namespace codegen {
namespace cxxnaive {

namespace {
std::string makeLoopImpl(int iExtent, int jExtent, const std::string& dim, const std::string& lower,
                         const std::string& upper, const std::string& comparison,
                         const std::string& increment) {
  return "for(int " + dim + " = " + lower + "+" + std::to_string(iExtent) + "; " + dim + " " +
         comparison + " " + upper + "+" + std::to_string(jExtent) + "; " + increment + dim + ")";
}

std::string makeIJLoop(int iExtent, int jExtent, const std::string dom, const std::string& dim) {
  return makeLoopImpl(iExtent, jExtent, dim, dom + "." + dim + "minus()",
                      dom + "." + dim + "size() - " + dom + "." + dim + "plus() - 1", " <= ", "++");
}

std::string makeIntervalBound(const std::string dom, iir::Interval const& interval,
                              iir::Interval::Bound bound) {
  return interval.levelIsEnd(bound)
             ? "( " + dom + ".ksize() == 0 ? 0 : (" + dom + ".ksize() - " + dom +
                   ".kplus() - 1)) + " + std::to_string(interval.offset(bound))
             : std::to_string(interval.bound(bound));
}

std::string makeKLoop(const std::string dom, bool isBackward, iir::Interval const& interval) {

  const std::string lower = makeIntervalBound(dom, interval, iir::Interval::Bound::lower);
  const std::string upper = makeIntervalBound(dom, interval, iir::Interval::Bound::upper);

  return isBackward ? makeLoopImpl(0, 0, "k", upper, lower, ">=", "--")
                    : makeLoopImpl(0, 0, "k", lower, upper, "<=", "++");
}
} // namespace

CXXNaiveCodeGen::CXXNaiveCodeGen(stencilInstantiationContext& ctx, DiagnosticsEngine& engine,
                                 int maxHaloPoint)
    : CodeGen(ctx, engine, maxHaloPoint) {}

CXXNaiveCodeGen::~CXXNaiveCodeGen() {}

std::string CXXNaiveCodeGen::generateStencilInstantiation(
    const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation) {
  using namespace codegen;

  std::stringstream ssSW;

  Namespace dawnNamespace("dawn_generated", ssSW);
  Namespace cxxnaiveNamespace("cxxnaive", ssSW);

  const auto& globalsMap = stencilInstantiation->getIIR()->getGlobalVariableMap();

  Class StencilWrapperClass(stencilInstantiation->getName(), ssSW);
  StencilWrapperClass.changeAccessibility("private");

  CodeGenProperties codeGenProperties = computeCodeGenProperties(stencilInstantiation.get());

  generateStencilFunctions(StencilWrapperClass, stencilInstantiation, codeGenProperties);

  generateStencilClasses(stencilInstantiation, StencilWrapperClass, codeGenProperties);

  generateStencilWrapperMembers(StencilWrapperClass, stencilInstantiation, codeGenProperties);

  generateStencilWrapperCtr(StencilWrapperClass, stencilInstantiation, codeGenProperties);

  generateGlobalsAPI(*stencilInstantiation, StencilWrapperClass, globalsMap, codeGenProperties);

  generateStencilWrapperRun(StencilWrapperClass, stencilInstantiation, codeGenProperties);

  StencilWrapperClass.commit();

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
    runMethod.addStatement(stencilDescCGVisitor.getCodeAndResetStream());
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

  StencilWrapperConstructor.addArg("const " + c_gtc() + "domain& dom");

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
      initCtr += ",m_globals";
    }
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
void CXXNaiveCodeGen::generateStencilWrapperMembers(
    Class& stencilWrapperClass,
    const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation,
    CodeGenProperties& codeGenProperties) const {

  const auto& metadata = stencilInstantiation->getMetaData();
  const auto& globalsMap = stencilInstantiation->getIIR()->getGlobalVariableMap();

  stencilWrapperClass.addMember("static constexpr const char* s_name =",
                                Twine("\"") + stencilWrapperClass.getName() + Twine("\""));

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

    stencilWrapperClass.addMember(c_gtc() + "meta_data_t", "m_meta_data");

    for(int AccessID : metadata.getAccessesOfType<iir::FieldAccessType::InterStencilTemporary>())
      stencilWrapperClass.addMember(c_gtc() + "storage_t",
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
    stencilClass.addComment("Temporary storages");
    addTempStorageTypedef(stencilClass, stencil);

    stencilClass.addMember("const " + c_gtc() + "domain", "m_dom");

    if(!globalsMap.empty()) {
      stencilClass.addMember("const globals&", "m_globals");
    }

    stencilClass.addComment("Input/Output storages");

    addTmpStorageDeclaration(stencilClass, tempFields);

    stencilClass.changeAccessibility("public");

    auto stencilClassCtr = stencilClass.addConstructor();

    stencilClassCtr.addArg("const " + c_gtc() + "domain& dom_");
    if(!globalsMap.empty()) {
      stencilClassCtr.addArg("const globals& globals_");
    }

    stencilClassCtr.addInit("m_dom(dom_)");
    if(!globalsMap.empty()) {
      stencilClassCtr.addArg("m_globals(globals_)");
    }

    addTmpStorageInit(stencilClassCtr, stencil, tempFields);
    stencilClassCtr.commit();

    // virtual dtor

    // synchronize storages method

    //
    // Run-Method
    //
    MemberFunction stencilRunMethod = stencilClass.addMemberFunction("void", "run", "");
    for(auto it = nonTempFields.begin(); it != nonTempFields.end(); ++it) {
      std::string type = stencilProperties->paramNameToType_.at((*it).second.Name);
      stencilRunMethod.addArg(type + "& " + (*it).second.Name + "_");
    }

    stencilRunMethod.startBody();

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
            makeKLoop("m_dom", (multiStage.getLoopOrder() == iir::LoopOrderKind::Backward),
                      interval),
            [&]() {
              for(const auto& stagePtr : multiStage.getChildren()) {
                const iir::Stage& stage = *stagePtr;

                auto const& extents = iir::extent_cast<iir::CartesianExtent const&>(
                    stage.getExtents().horizontalExtent());

                stencilRunMethod.addBlockStatement(
                    makeIJLoop(extents.iMinus(), extents.iPlus(), "m_dom", "i"), [&]() {
                      stencilRunMethod.addBlockStatement(
                          makeIJLoop(extents.jMinus(), extents.jPlus(), "m_dom", "j"), [&]() {
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
                          });
                    });
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
        DiagnosticsBuilder diag(DiagnosticsKind::Error,
                                stencilInstantiation->getMetaData().getStencilLocation());
        diag << "no storages referenced in stencil '" << stencilInstantiation->getName()
             << "', this would result in invalid gridtools code";
        diagEngine.report(diag);
        return;
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
        DiagnosticsBuilder diag(DiagnosticsKind::Error, stencilFun->getStencilFunction()->Loc);
        diag << "no storages referenced in stencil function '" << stencilFun->getName()
             << "', this would result in invalid gridtools code";
        diagEngine.report(diag);
        return;
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

  ppDefines.push_back(makeDefine("GRIDTOOLS_CLANG_GENERATED", 1));
  ppDefines.push_back("#define GRIDTOOLS_CLANG_BACKEND_T CXXNAIVE");
  // ==============------------------------------------------------------------------------------===
  // BENCHMARKTODO: since we're importing two cpp files into the benchmark API we need to set
  // these variables also in the naive code-generation in order to not break it. Once the move to
  // different TU's is completed, this is no longer necessary.
  // [https://github.com/MeteoSwiss-APN/gtclang/issues/32]
  // ==============------------------------------------------------------------------------------===
  CodeGen::addMplIfdefs(ppDefines, 30);
  DAWN_LOG(INFO) << "Done generating code";

  std::string filename = generateFileName(context_);
  return std::make_unique<TranslationUnit>(filename, std::move(ppDefines), std::move(stencils),
                                           std::move(globals));
}

} // namespace cxxnaive
} // namespace codegen
} // namespace dawn
