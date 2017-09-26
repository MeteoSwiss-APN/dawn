//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gsl/CodeGen/GTClangNaiveCXXCodeGen.h"
#include "gsl/CodeGen/CXXUtil.h"
#include "gsl/Optimizer/OptimizerContext.h"
#include "gsl/Optimizer/StencilInstantiation.h"
#include "gsl/SIR/SIR.h"
#include "gsl/Support/Assert.h"
#include "gsl/Support/Logging.h"
#include "gsl/Support/StringUtil.h"

namespace gsl {

std::string makeLoopImpl(const std::string& dim, const std::string& lower, const std::string& upper,
                         const std::string& comparison, const std::string& increment) {
  return Twine("for(int " + dim + " = " + lower + "; " + dim + " " + comparison + " " + upper +
               "; " + increment + dim + ")")
      .str();
}

std::string makeIJLoop(const std::string& dim) {
  return makeLoopImpl(dim, "dom." + dim + "minus()",
                      "dom." + dim + "size() - dom." + dim + "plus() - 1", " <= ", "++");
}

std::string makeKLoop(bool isBackward) {
  const std::string lower = "dom.kminus()";
  const std::string upper = "(dom.ksize() == 0 ? 0 : (dom.ksize() - dom.kplus() - 1))";
  return isBackward ? makeLoopImpl("k", upper, lower, ">=", "--")
                    : makeLoopImpl("k", lower, upper, "<=", "++");
}

std::string makeBound(int level, int offset) {
  std::string offsetStr;
  if(offset != 0)
    offsetStr = level > 0 ? "+" + std::to_string(offset) : std::to_string(offset);
  switch(level) {
  case sir::Interval::Start:
    return "dom.kminus()" + offsetStr;
  case sir::Interval::End:
    return "(dom.ksize() == 0 ? 0 : (dom.ksize() - dom.kplus() - 1))" + offsetStr;
  default:
    return std::to_string(level) + offsetStr;
  }
}

GTClangNaiveCXXCodeGen::GTClangNaiveCXXCodeGen(OptimizerContext* context) : CodeGen(context) {}

GTClangNaiveCXXCodeGen::~GTClangNaiveCXXCodeGen() {}

std::string GTClangNaiveCXXCodeGen::generateStencilInstantiation(
    const StencilInstantiation* stencilInstantiation) {
  using namespace codegen;

  std::stringstream ssSW, tss;

  Class StencilWrapperClass(stencilInstantiation->getName(), ssSW);
  StencilWrapperClass.changeAccessibility("private");

  // Generate stencils
  auto& stencils = stencilInstantiation->getStencils();
  for(std::size_t stencilIdx = 0; stencilIdx < stencils.size(); ++stencilIdx) {
    const Stencil& stencil = *stencilInstantiation->getStencils()[stencilIdx];

    if(stencil.isEmpty())
      continue;

    //
    // Stencil definition
    //
    const auto& StencilFields = stencil.getFields();
    std::vector<std::string> StencilTemplates;
    for(int i = 0; i < StencilFields.size(); ++i)
      StencilTemplates.push_back("StorageType" + std::to_string(i + 1));

    Structure StencilClass = StencilWrapperClass.addStruct(
        Twine("stencil_") + Twine(stencilIdx),
        RangeToString(", ", "", "")(StencilTemplates,
                                    [](const std::string& type) { return "class " + type; }));
    std::string StencilName = StencilClass.getName();

    //
    // Members
    //
    StencilClass.addMember("gridtools::clang::domain", "dom");
    for(int i = 0; i < StencilFields.size(); ++i)
      StencilClass.addMember(StencilTemplates[i] + "&", StencilFields[i].Name);

    StencilClass.changeAccessibility("public");

    //
    // Constructor
    //
    auto StencilConstructor = StencilClass.addConstructor(RangeToString(", ", "", "")(
        StencilTemplates, [](const std::string& str) { return "class " + str; }));

    // Arguments
    StencilConstructor.addArg("const gridtools::clang::domain& dom_");
    for(int i = 0; i < StencilFields.size(); ++i)
      StencilConstructor.addArg(StencilTemplates[i] + "& " + StencilFields[i].Name + "_");

    // Initializers
    StencilConstructor.addInit("dom(dom_)");
    for(int i = 0; i < StencilFields.size(); ++i)
      StencilConstructor.addInit(StencilFields[i].Name + "(" + StencilFields[i].Name + "_)");

    StencilConstructor.commit();

    //
    // Do-Method
    //
    MemberFunction StencilDoMethod = StencilClass.addMemberFunction("void", "Do");

    for(const auto& multiStagePtr : stencil.getMultiStages()) {
      const MultiStage& multiStage = *multiStagePtr;
      StencilDoMethod.addBlockStatement(
          makeKLoop(multiStage.getLoopOrder() == LoopOrderKind::LK_Backward), [&]() {
            for(const auto& stagePtr : multiStage.getStages()) {
              const Stage& stage = *stagePtr;

              // Check Interval
              stage.getIntervals();

              StencilDoMethod.addBlockStatement(makeIJLoop("i"), [&]() {
                StencilDoMethod.addBlockStatement(makeIJLoop("j"), [&]() {

                });
              });
            }
          });
    }
    StencilDoMethod.commit();
  }

  StencilWrapperClass.commit();

  // Remove trailing ';' as this is retained by Clang's Rewriter
  std::string str = ssSW.str();
  str[str.size() - 2] = ' ';

  return str;
}

std::unique_ptr<TranslationUnit> GTClangNaiveCXXCodeGen::generateCode() {
  GSL_ASSERT_MSG(0, "naive codegen: not yet implement");
  GSL_LOG(INFO) << "Starting code generation for GTClang ...";

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
  auto define = [](std::string define, int value) {
    return "#define " + define + " " + std::to_string(value);
  };

  auto defineIfNDef = [](std::string define, int value) {
    return "#ifndef " + define + "\n #define " + define + " " + std::to_string(value) + "\n#endif";
  };

  ppDefines.push_back(define("GRIDTOOLS_CLANG_GENERATED", 1));
  ppDefines.push_back(define("GRIDTOOLS_CLANG_HALO_EXTEND", context_->getOptions().MaxHaloPoints));
  ppDefines.push_back(defineIfNDef("BOOST_RESULT_OF_USE_TR1", 1));
  ppDefines.push_back(defineIfNDef("BOOST_NO_CXX11_DECLTYPE", 1));

  GSL_LOG(INFO) << "Done generating code";

  return make_unique<TranslationUnit>(context_->getSIR()->Filename, std::move(ppDefines),
                                      std::move(stencils), std::move(globals));
}

} // namespace gsl
