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
#include "dawn/CodeGen/CXXNaive/ASTCodeGenCXXNaiveStencilBody.h"
#include "dawn/CodeGen/CXXNaive/ASTCodeGenCXXNaiveStencilDesc.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/StringUtil.h"
#include <algorithm>

namespace dawn {

static std::string makeLoopImpl(const std::string& dim, const std::string& lower,
                                const std::string& upper, const std::string& comparison,
                                const std::string& increment) {
  return Twine("for(int " + dim + " = " + lower + "; " + dim + " " + comparison + " " + upper +
               "; " + increment + dim + ")")
      .str();
}

static std::string makeIJLoop(const std::string dom, const std::string& dim) {
  return makeLoopImpl(dim, dom + "." + dim + "minus()",
                      dom + "." + dim + "size() - " + dom + "." + dim + "plus() - 1", " <= ", "++");
}

static std::string makeKLoop(const std::string dom, bool isBackward, Interval const& interval) {
  const std::string lower = std::to_string(interval.lowerBound());
  const std::string upper =
      interval.upperIsEnd()
          ? "( " + dom + ".ksize() == 0 ? 0 : (" + dom + ".ksize() - " + dom + ".kplus() - 1))"
          : std::to_string(interval.upperBound());
  return isBackward ? makeLoopImpl("k", upper, lower, ">=", "--")
                    : makeLoopImpl("k", lower, upper, "<=", "++");
}

CXXNaiveCodeGen::CXXNaiveCodeGen(OptimizerContext* context) : CodeGen(context) {}

CXXNaiveCodeGen::~CXXNaiveCodeGen() {}

std::string
CXXNaiveCodeGen::generateStencilInstantiation(const StencilInstantiation* stencilInstantiation) {
  using namespace codegen;

  std::stringstream ssSW, tss;

  Class StencilWrapperClass(stencilInstantiation->getName(), ssSW);
  StencilWrapperClass.changeAccessibility("private");

  Structure paramWrapper = StencilWrapperClass.addStruct("ParamWrapper", "class DataView");
  paramWrapper.addMember("DataView", "dview_");
  paramWrapper.addMember("std::array<int, DataView::storage_info_t::ndims>", "offsets_");

  auto pwClassCtr = paramWrapper.addConstructor();

  pwClassCtr.addArg("DataView dview");
  pwClassCtr.addArg("std::array<int, DataView::storage_info_t::ndims> offsets");
  pwClassCtr.addInit("dview_(dview)");
  pwClassCtr.addInit("offsets_(offsets)");

  pwClassCtr.commit();

  paramWrapper.commit();

  // Generate stencils
  auto& stencils = stencilInstantiation->getStencils();

  // stencil functions
  //
  // Generate stencil functions code for stencils instantiated by this stencil
  //
  std::unordered_set<std::string> generatedStencilFun;
  for(const auto& stencilFun : stencilInstantiation->getStencilFunctionInstantiations()) {
    std::string stencilFunName = StencilFunctionInstantiation::makeCodeGenName(*stencilFun);
    if(generatedStencilFun.emplace(stencilFunName).second) {

      // Field declaration
      const auto& fields = stencilFun->getCalleeFields();

      std::vector<std::string> stencilFnTemplates(fields.size());
      // TODO move to capture initialization with C++14
      int n = 0;
      std::generate(stencilFnTemplates.begin(), stencilFnTemplates.end(),
                    [n]() mutable { return "StorageType" + std::to_string(n++); });

      MemberFunction stencilFunMethod = StencilWrapperClass.addMemberFunction(
          std::string("static ") + (stencilFun->hasReturn() ? "double" : "void"), stencilFunName,
          RangeToString(", ", "", "")(stencilFnTemplates,
                                      [](const std::string& str) { return "class " + str; }));

      std::vector<std::string> arglist;

      if(fields.empty() && !stencilFun->hasReturn()) {
        DiagnosticsBuilder diag(DiagnosticsKind::Error, stencilFun->getStencilFunction()->Loc);
        diag << "no storages referenced in stencil function '" << stencilFun->getName()
             << "', this would result in invalid gridtools code";
        context_->getDiagnostics().report(diag);
        return "";
      }

      // If we have a return argument, we generate a special `__out` field
      //      int accessorID = 0;
      //      if(stencilFun->hasReturn()) {
      //        StencilFunStruct.addStatement("using __out = gridtools::accessor<0, "
      //                                      "gridtools::enumtype::inout, gridtools::extent<0, 0,
      //                                      0, 0, "
      //                                      "0, 0>>");
      //        arglist.push_back("__out");
      //        accessorID++;
      //      }

      // Generate field declarations
      stencilFunMethod.addArg("const int i");
      stencilFunMethod.addArg("const int j");
      stencilFunMethod.addArg("const int k");

      std::unordered_map<std::string, std::string> paramNameToType;
      for(std::size_t m = 0; m < fields.size(); ++m) {

        std::string paramName = stencilFun->getOriginalNameFromCallerAccessID(fields[m].AccessID);
        paramNameToType.emplace(paramName, stencilFnTemplates[m]);

        stencilFunMethod.addArg("ParamWrapper<" + gt() + "data_view<StorageType" +
                                std::to_string(m) + ">> pw_" + paramName);
      }

      ASTCodeGenCXXNaiveStencilBody stencilBodyCXXVisitor(stencilInstantiation, paramNameToType,
                                                          StencilContext::E_StencilFunction);

      stencilFunMethod.startBody();

      for(std::size_t m = 0; m < fields.size(); ++m) {

        std::string paramName = stencilFun->getOriginalNameFromCallerAccessID(fields[m].AccessID);

        stencilFunMethod << gt() << "data_view<StorageType" + std::to_string(m) + "> " << paramName
                         << " = pw_" << paramName << ".dview_;";
        stencilFunMethod << "auto " << paramName << "_offsets = pw_" << paramName << ".offsets_;";
      }

      stencilBodyCXXVisitor.setCurrentStencilFunction(stencilFun.get());
      stencilBodyCXXVisitor.setIndent(stencilFunMethod.getIndent());
      for(const auto& statementAccessesPair : stencilFun->getStatementAccessesPairs()) {
        statementAccessesPair->getStatement()->ASTStmt->accept(stencilBodyCXXVisitor);
        stencilFunMethod.indentStatment();
        stencilFunMethod << stencilBodyCXXVisitor.getCodeAndResetStream();
      }

      stencilFunMethod.commit();
    }
  }

  // Stencil members
  std::vector<std::string> innerStencilNames(stencils.size());

  for(std::size_t stencilIdx = 0; stencilIdx < stencils.size(); ++stencilIdx) {
    const Stencil& stencil = *stencilInstantiation->getStencils()[stencilIdx];

    if(stencil.isEmpty())
      continue;

    //
    // Stencil definition
    //
    const auto& StencilFields = stencil.getFields();

    innerStencilNames[stencilIdx] = "stencil_" + std::to_string(stencilIdx);

    std::vector<std::string> StencilTemplates;
    for(int i = 0; i < StencilFields.size(); ++i)
      StencilTemplates.push_back("StorageType" + std::to_string(i + 1));

    Structure sbase = StencilWrapperClass.addStruct("sbase", "");
    MemberFunction sbase_run = sbase.addMemberFunction("virtual void", "run");
    sbase_run.startBody();
    sbase_run.commit();
    sbase.commit();

    Structure StencilClass = StencilWrapperClass.addStruct(
        innerStencilNames[stencilIdx],
        RangeToString(", ", "", "")(StencilTemplates,
                                    [](const std::string& str) { return "class " + str; }),
        "sbase");
    std::string StencilName = StencilClass.getName();

    std::unordered_map<std::string, std::string> paramNameToType;
    for(int i = 0; i < StencilFields.size(); ++i) {
      paramNameToType.emplace(StencilFields[i].Name, StencilTemplates[i]);
    }

    ASTCodeGenCXXNaiveStencilBody stencilBodyCXXVisitor(stencilInstantiation, paramNameToType,
                                                        StencilContext::E_Stencil);

    StencilClass.addMember("const " + gtc() + "domain&", "m_dom");
    for(int i = 0; i < StencilFields.size(); ++i) {
      StencilClass.addMember(StencilTemplates[i] + "&", "m_" + StencilFields[i].Name);
    }

    StencilClass.changeAccessibility("public");

    auto stencilClassCtr = StencilClass.addConstructor();

    stencilClassCtr.addArg("const " + gtc() + "domain& dom_");
    for(int i = 0; i < StencilFields.size(); ++i)
      stencilClassCtr.addArg(StencilTemplates[i] + "& " + StencilFields[i].Name + "_");
    stencilClassCtr.addInit("m_dom(dom_)");
    for(int i = 0; i < StencilFields.size(); ++i)
      stencilClassCtr.addInit("m_" + StencilFields[i].Name + "(" + StencilFields[i].Name + "_)");

    stencilClassCtr.commit();

    //
    // Do-Method
    //
    MemberFunction StencilDoMethod = StencilClass.addMemberFunction("virtual void", "run", "");

    for(const auto& multiStagePtr : stencil.getMultiStages()) {
      const MultiStage& multiStage = *multiStagePtr;

      for(int i = 0; i < StencilFields.size(); ++i)
        StencilDoMethod.addStatement(gt() + "data_view<" + StencilTemplates[i] + "> " +
                                     StencilFields[i].Name + "= " + gt() + "make_host_view(m_" +
                                     StencilFields[i].Name + ")");

      auto intervals_set = multiStage.getIntervals();
      std::vector<Interval> intervals_v;
      std::copy(intervals_set.begin(), intervals_set.end(), std::back_inserter(intervals_v));

      auto partitionIntervals = Interval::computePartition(intervals_v);

      for(auto interval : partitionIntervals) {
        StencilDoMethod.addBlockStatement(
            makeKLoop("m_dom", (multiStage.getLoopOrder() == LoopOrderKind::LK_Backward), interval),
            [&]() {
              for(const auto& stagePtr : multiStage.getStages()) {
                const Stage& stage = *stagePtr;
                // Check Interval
                stage.getIntervals();

                StencilDoMethod.addBlockStatement(makeIJLoop("m_dom", "i"), [&]() {
                  StencilDoMethod.addBlockStatement(makeIJLoop("m_dom", "j"), [&]() {

                    // Generate Do-Method
                    for(const auto& doMethodPtr : stagePtr->getDoMethods()) {
                      const DoMethod& doMethod = *doMethodPtr;

                      if(!doMethod.getInterval().overlaps(interval))
                        continue;
                      for(const auto& statementAccessesPair :
                          doMethod.getStatementAccessesPairs()) {
                        statementAccessesPair->getStatement()->ASTStmt->accept(
                            stencilBodyCXXVisitor);
                        StencilDoMethod << stencilBodyCXXVisitor.getCodeAndResetStream();
                      }
                    }

                  });
                });
              }
            });
      }
    }
    StencilDoMethod.commit();
  }

  StencilWrapperClass.addMember("static constexpr const char* s_name =",
                                Twine("\"") + StencilWrapperClass.getName() + Twine("\""));

  for(auto innerStencil : innerStencilNames) {
    StencilWrapperClass.addMember("sbase*", "m_" + innerStencil);
  }

  StencilWrapperClass.changeAccessibility("public");
  StencilWrapperClass.addCopyConstructor(Class::Deleted);

  StencilWrapperClass.addComment("Members");
  //
  // Members
  //
  // Define allocated memebers if necessary
  if(stencilInstantiation->hasAllocatedFields()) {
    StencilWrapperClass.addMember(gtc() + "meta_data_t", "m_meta_data");

    for(int AccessID : stencilInstantiation->getAllocatedFieldAccessIDs())
      StencilWrapperClass.addMember(gtc() + "storage_t",
                                    stencilInstantiation->getNameFromAccessID(AccessID) + "_");
  }

  // Generate stencil wrapper constructor
  decltype(stencilInstantiation->getSIRStencil()->Fields) SIRFieldsWithoutTemps;

  std::copy_if(stencilInstantiation->getSIRStencil()->Fields.begin(),
               stencilInstantiation->getSIRStencil()->Fields.end(),
               std::back_inserter(SIRFieldsWithoutTemps),
               [](std::shared_ptr<sir::Field> const& f) { return !(f->IsTemporary); });

  std::vector<std::string> StencilWrapperRunTemplates;
  for(int i = 0; i < SIRFieldsWithoutTemps.size(); ++i)
    StencilWrapperRunTemplates.push_back("StorageType" + std::to_string(i + 1));

  auto StencilWrapperConstructor = StencilWrapperClass.addConstructor(RangeToString(", ", "", "")(
      StencilWrapperRunTemplates, [](const std::string& str) { return "class " + str; }));

  StencilWrapperConstructor.addArg("const " + gtc() + "domain& dom");
  std::string ctrArgs("(dom");
  for(int i = 0; i < SIRFieldsWithoutTemps.size(); ++i) {
    StencilWrapperConstructor.addArg(StencilWrapperRunTemplates[i] + "& " +
                                     SIRFieldsWithoutTemps[i]->Name);
    ctrArgs += "," + SIRFieldsWithoutTemps[i]->Name;
  }

  for(std::size_t stencilIdx = 0; stencilIdx < stencils.size(); ++stencilIdx) {

    const Stencil& stencil = *stencilInstantiation->getStencils()[stencilIdx];

    if(stencil.isEmpty())
      continue;

    const auto& StencilFields = stencil.getFields();

    std::string initCtr =
        "m_" + innerStencilNames[stencilIdx] + "(new " + innerStencilNames[stencilIdx];

    // TODO HERE
    for(int i = 0; i < StencilFields.size(); ++i) {
      initCtr +=
          (i != 0 ? "," : "<") + (stencilInstantiation->isAllocatedField(StencilFields[i].AccessID)
                                      ? (gtc().str() + "storage_t")
                                      : (std::string("StorageType") + std::to_string(i + 1)));
    }
    initCtr += ">(dom";
    for(auto field : StencilFields) {
      initCtr += "," + (stencilInstantiation->isAllocatedField(field.AccessID) ? (field.Name + "_")
                                                                               : (field.Name));
    }
    initCtr += ") )";
    StencilWrapperConstructor.addInit(initCtr);
  }
  if(stencilInstantiation->hasAllocatedFields()) {
    StencilWrapperConstructor.addInit("m_meta_data(dom.isize(), dom.jsize(), dom.ksize())");
    for(int AccessID : stencilInstantiation->getAllocatedFieldAccessIDs())
      StencilWrapperConstructor.addInit(
          stencilInstantiation->getNameFromAccessID(AccessID) + "_(m_meta_data,\"" +
          stencilInstantiation->getNameFromAccessID(AccessID) + "\")");
  }

  StencilWrapperConstructor.commit();

  // Generate the run method by generate code for the stencil description AST
  MemberFunction RunMethod = StencilWrapperClass.addMemberFunction("void", "run", "");

  RunMethod.finishArgs();
  // Create the StencilID -> stencil name map
  std::unordered_map<int, std::vector<std::string>> stencilIDToStencilNameMap;
  for(std::size_t i = 0; i < stencils.size(); ++i)
    stencilIDToStencilNameMap[stencils[i]->getStencilID()].emplace_back(innerStencilNames[i]);

  ASTCodeGenCXXNaiveStencilDesc stencilDescCGVisitor(stencilInstantiation,
                                                     stencilIDToStencilNameMap);
  stencilDescCGVisitor.setIndent(RunMethod.getIndent());
  for(const auto& statement : stencilInstantiation->getStencilDescStatements()) {
    statement->ASTStmt->accept(stencilDescCGVisitor);
    RunMethod.addStatement(stencilDescCGVisitor.getCodeAndResetStream());
  }

  RunMethod.commit();

  StencilWrapperClass.commit();

  // Remove trailing ';' as this is retained by Clang's Rewriter
  std::string str = ssSW.str();
  str[str.size() - 2] = ' ';

  return str;
}

std::unique_ptr<TranslationUnit> CXXNaiveCodeGen::generateCode() {
  //  DAWN_ASSERT_MSG(0, "naive codegen: not yet implement");
  DAWN_LOG(INFO) << "Starting code generation for GTClang ...";

  // Generate code for StencilInstantiations
  std::map<std::string, std::string> stencils;
  for(const auto& nameStencilCtxPair : context_->getStencilInstantiationMap()) {
    std::string code = generateStencilInstantiation(nameStencilCtxPair.second.get());
    if(code.empty())
      return nullptr;
    stencils.emplace(nameStencilCtxPair.first, std::move(code));
  }

  // TODO:
  std::string globals = "";

  std::vector<std::string> ppDefines;
  auto makeDefine = [](std::string define, int value) {
    return "#define " + define + " " + std::to_string(value);
  };

  auto makeIfNDef = [](std::string define, int value) {
    return "#ifndef " + define + "\n #define " + define + " " + std::to_string(value) + "\n#endif";
  };

  ppDefines.push_back(makeDefine("GRIDTOOLS_CLANG_GENERATED", 1));
  ppDefines.push_back(
      makeDefine("GRIDTOOLS_CLANG_HALO_EXTEND", context_->getOptions().MaxHaloPoints));
  ppDefines.push_back(makeIfNDef("BOOST_RESULT_OF_USE_TR1", 1));
  ppDefines.push_back(makeIfNDef("BOOST_NO_CXX11_DECLTYPE", 1));

  DAWN_LOG(INFO) << "Done generating code";

  return make_unique<TranslationUnit>(context_->getSIR()->Filename, std::move(ppDefines),
                                      std::move(stencils), std::move(globals));
}

} // namespace dawn
