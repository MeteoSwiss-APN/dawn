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

namespace dawn {

static std::string makeLoopImpl(const std::string& dim, const std::string& lower,
                                const std::string& upper, const std::string& comparison,
                                const std::string& increment) {
  return Twine("for(int " + dim + " = " + lower + "; " + dim + " " + comparison + " " + upper +
               "; " + increment + dim + ")")
      .str();
}

static std::string makeIJLoop(const std::string& dim) {
  return makeLoopImpl(dim, "dom_." + dim + "minus()",
                      "dom_." + dim + "size() - dom_." + dim + "plus() - 1", " <= ", "++");
}

static std::string makeKLoop(bool isBackward, Interval const& interval) {
  const std::string lower = std::to_string(interval.lowerBound());
  const std::string upper = interval.upperIsEnd()
                                ? "( dom_.ksize() == 0 ? 0 : (dom_.ksize() - dom_.kplus() - 1))"
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

  auto gt = []() { return Twine("gridtools::"); };
  auto gtc = []() { return Twine("gridtools::clang::"); };
  auto gt_enum = []() { return Twine("gridtools::enumtype::"); };

  Class StencilWrapperClass(stencilInstantiation->getName(), ssSW);
  StencilWrapperClass.changeAccessibility("private");

  // Generate stencils
  auto& stencils = stencilInstantiation->getStencils();
  ASTCodeGenCXXNaiveStencilBody stencilBodyCXXVisitor(stencilInstantiation);

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

    Structure StencilClass = StencilWrapperClass.addStruct(innerStencilNames[stencilIdx], "");
    //        RangeToString(", ", "", "")(StencilTemplates,
    //                                    [](const std::string& type) { return "class " + type; }));
    std::string StencilName = StencilClass.getName();

    //
    // Members
    //
    StencilClass.changeAccessibility("public");

    //
    // Do-Method
    //
    std::vector<std::string> StencilTemplates;
    for(int i = 0; i < StencilFields.size(); ++i)
      StencilTemplates.push_back("StorageType" + std::to_string(i + 1));

    MemberFunction StencilDoMethod = StencilClass.addMemberFunction(
        "static void", "run",
        RangeToString(", ", "", "")(StencilTemplates,
                                    [](const std::string& str) { return "class " + str; }));

    StencilDoMethod.addArg("const " + gtc() + "domain& dom_");
    for(int i = 0; i < StencilFields.size(); ++i)
      StencilDoMethod.addArg(StencilTemplates[i] + "& " + StencilFields[i].Name + "_");

    for(const auto& multiStagePtr : stencil.getMultiStages()) {
      const MultiStage& multiStage = *multiStagePtr;

      for(int i = 0; i < StencilFields.size(); ++i)
        StencilDoMethod.addStatement(gt() + "data_view<" + StencilTemplates[i] + "> " +
                                     StencilFields[i].Name + "= " + gt() + "make_host_view(" +
                                     StencilFields[i].Name + "_)");

      auto intervals_set = multiStage.getIntervals();
      std::vector<Interval> intervals_v;
      std::copy(intervals_set.begin(), intervals_set.end(), std::back_inserter(intervals_v));

      auto partitionIntervals = Interval::computePartition(intervals_v);

      for(auto interval : partitionIntervals) {
        StencilDoMethod.addBlockStatement(
            makeKLoop((multiStage.getLoopOrder() == LoopOrderKind::LK_Backward), interval), [&]() {
              for(const auto& stagePtr : multiStage.getStages()) {
                const Stage& stage = *stagePtr;
                std::cout << "HOOOOOOO" << interval.lowerBound() << " " << interval.upperBound()
                          << std::endl;
                // Check Interval
                stage.getIntervals();

                StencilDoMethod.addBlockStatement(makeIJLoop("i"), [&]() {
                  StencilDoMethod.addBlockStatement(makeIJLoop("j"), [&]() {

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

  StencilWrapperClass.changeAccessibility("public");
  StencilWrapperClass.addCopyConstructor(Class::Deleted);

  // Generate stencil wrapper constructor
  auto SIRFieldsWithoutTemps = stencilInstantiation->getSIRStencil()->Fields;
  for(auto it = SIRFieldsWithoutTemps.begin(); it != SIRFieldsWithoutTemps.end();)
    if((*it)->IsTemporary)
      it = SIRFieldsWithoutTemps.erase(it);
    else
      ++it;

  std::vector<std::string> StencilWrapperRunTemplates;
  for(int i = 0; i < SIRFieldsWithoutTemps.size(); ++i)
    StencilWrapperRunTemplates.push_back("S" + std::to_string(i + 1));

  auto StencilWrapperConstructor = StencilWrapperClass.addConstructor(RangeToString(", ", "", "")(
      StencilWrapperRunTemplates, [](const std::string& str) { return "class " + str; }));

  StencilWrapperConstructor.addArg("const " + gtc() + "domain& dom");
  std::string ctrArgs("(dom");
  for(int i = 0; i < SIRFieldsWithoutTemps.size(); ++i) {
    StencilWrapperConstructor.addArg(StencilWrapperRunTemplates[i] + "& " +
                                     SIRFieldsWithoutTemps[i]->Name);
    ctrArgs += "," + SIRFieldsWithoutTemps[i]->Name;
  }
  ctrArgs += ")";
  StencilWrapperConstructor.addStatement("run" + ctrArgs);

  StencilWrapperConstructor.commit();

  // Generate the run method by generate code for the stencil description AST
  MemberFunction RunMethod = StencilWrapperClass.addMemberFunction(
      "static void", "run",
      RangeToString(", ", "", "")(StencilWrapperRunTemplates,
                                  [](const std::string& str) { return "class " + str; }));

  RunMethod.addArg("const " + gtc() + "domain& dom");
  for(int i = 0; i < SIRFieldsWithoutTemps.size(); ++i)
    RunMethod.addArg(StencilWrapperRunTemplates[i] + "& " + SIRFieldsWithoutTemps[i]->Name);

  for(int i = 0; i < SIRFieldsWithoutTemps.size(); ++i)
    RunMethod.addStatement("static_assert(" + gt() + "is_data_store<" +
                           StencilWrapperRunTemplates[i] + ">::value, \"argument '" +
                           SIRFieldsWithoutTemps[i]->Name + "' is not a 'data_store' (" +
                           decimalToOrdinal(i + 2) + " argument invalid)\")");

  // Create the StencilID -> stencil name map
  std::unordered_map<int, std::vector<std::string>> stencilIDToStencilNameMap;
  for(std::size_t i = 0; i < stencils.size(); ++i)
    stencilIDToStencilNameMap[stencils[i]->getStencilID()].emplace_back(innerStencilNames[i]);

  ASTCodeGenCXXNaiveStencilDesc stencilDescCGVisitor(stencilInstantiation,
                                                     stencilIDToStencilNameMap);
  stencilDescCGVisitor.setIndent(RunMethod.getIndent());
  for(const auto& statement : stencilInstantiation->getStencilDescStatements()) {
    statement->ASTStmt->accept(stencilDescCGVisitor);
    RunMethod << stencilDescCGVisitor.getCodeAndResetStream();
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
