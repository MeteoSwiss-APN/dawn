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

#include "dawn/IIR/DependencyGraphStage.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/Renaming.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Json.h"
#include "dawn/Support/StringUtil.h"
#include "dawn/Support/Twine.h"
#include "dawn/Support/Unreachable.h"
#include <algorithm>
#include <iostream>
#include <numeric>

namespace dawn {
namespace iir {

IIR::IIR() : metadata_(std::make_shared<StencilMetaInformation>()) {}

IIR::IIR(OptimizerContext* creator)
    : metadata_(std::make_shared<StencilMetaInformation>()), creator_(creator) {
  DAWN_ASSERT(creator);
}

void IIR::updateFromChildren() {}

std::unique_ptr<IIR> IIR::clone() const {

  auto cloneIIR = make_unique<IIR>(creator_);
  cloneIIR->cloneChildrenFrom(*this, cloneIIR);
  cloneIIR->getMetaData()->clone(*getMetaData());
  //  cloneIIR->getMetaData() = getMetaData();
  return cloneIIR;
}

Options& IIR::getOptions() { return creator_->getOptions(); }

const DiagnosticsEngine& IIR::getDiagnostics() const { return creator_->getDiagnostics(); }
DiagnosticsEngine& IIR::getDiagnostics() { return creator_->getDiagnostics(); }

const HardwareConfig& IIR::getHardwareConfiguration() const {
  return creator_->getHardwareConfiguration();
}

HardwareConfig& IIR::getHardwareConfiguration() { return creator_->getHardwareConfiguration(); }

void IIR::dumpTreeAsJson(std::string filename, std::string passName = "") {
  json::json jout;

  int i = 0;
  for(const auto& stencil : getChildren()) {
    json::json jStencil;

    int j = 0;
    for(const auto& multiStage : stencil->getChildren()) {
      json::json jMultiStage;
      jMultiStage["LoopOrder"] = loopOrderToString(multiStage->getLoopOrder());

      int k = 0;
      const auto& stages = multiStage->getChildren();
      for(const auto& stage : stages) {
        json::json jStage;

        int l = 0;
        for(const auto& doMethod : stage->getChildren()) {
          json::json jDoMethod;

          jDoMethod["Interval"] = doMethod->getInterval().toString();

          const auto& statementAccessesPairs = doMethod->getChildren();
          for(std::size_t m = 0; m < statementAccessesPairs.size(); ++m) {
            jDoMethod["Stmt_" + std::to_string(m)] = ASTStringifer::toString(
                statementAccessesPairs[m]->getStatement()->ASTStmt, 0, false);
            jDoMethod["Accesses_" + std::to_string(m)] =
                statementAccessesPairs[m]->getAccesses()->reportAccesses(this);
          }

          jStage["Do_" + std::to_string(l++)] = jDoMethod;
        }

        jMultiStage["Stage_" + std::to_string(k++)] = jStage;
      }

      jStencil["MultiStage_" + std::to_string(j++)] = jMultiStage;
    }

    if(passName.empty())
      jout[getMetaData()->getName()]["Stencil_" + std::to_string(i)] = jStencil;
    else
      jout[passName][getMetaData()->getName()]["Stencil_" + std::to_string(i)] = jStencil;
    ++i;
  }

  std::ofstream fs(filename, std::ios::out | std::ios::trunc);
  if(!fs.is_open()) {
    DiagnosticsBuilder diag(DiagnosticsKind::Error, SourceLocation());
    diag << "file system error: cannot open file: " << filename;
    getDiagnostics().report(diag);
  }

  fs << jout.dump(2) << std::endl;
  fs.close();
}

template <int Level>
struct PrintDescLine {
  PrintDescLine(const Twine& name) {
    std::cout << MakeIndent<Level>::value << format("\e[1;3%im", Level) << name.str() << "\n"
              << MakeIndent<Level>::value << "{\n\e[0m";
  }
  ~PrintDescLine() { std::cout << MakeIndent<Level>::value << format("\e[1;3%im}\n\e[0m", Level); }
};

void IIR::dump() const {
  std::cout << "IIR : " << getMetaData()->getName() << "\n";

  int i = 0;
  for(const auto& stencil : getChildren()) {
    PrintDescLine<1> iline("Stencil_" + Twine(i));

    int j = 0;
    const auto& multiStages = stencil->getChildren();
    for(const auto& multiStage : multiStages) {
      PrintDescLine<2> jline(Twine("MultiStage_") + Twine(j) + " [" +
                             loopOrderToString(multiStage->getLoopOrder()) + "]");

      int k = 0;
      const auto& stages = multiStage->getChildren();
      for(const auto& stage : stages) {
        PrintDescLine<3> kline(Twine("Stage_") + Twine(k));

        int l = 0;
        const auto& doMethods = stage->getChildren();
        for(const auto& doMethod : doMethods) {
          PrintDescLine<4> lline(Twine("Do_") + Twine(l) + " " +
                                 doMethod->getInterval().toString());

          const auto& statementAccessesPairs = doMethod->getChildren();
          for(std::size_t m = 0; m < statementAccessesPairs.size(); ++m) {
            std::cout << "\e[1m"
                      << ASTStringifer::toString(statementAccessesPairs[m]->getStatement()->ASTStmt,
                                                 5 * DAWN_PRINT_INDENT)
                      << "\e[0m";
            std::cout << statementAccessesPairs[m]->getAccesses()->toString(this,
                                                                            6 * DAWN_PRINT_INDENT)
                      << "\n";
          }
          l += 1;
        }
        std::cout << "\e[1m" << std::string(4 * DAWN_PRINT_INDENT, ' ')
                  << "Extents: " << stage->getExtents() << std::endl
                  << "\e[0m";
        k += 1;
      }
      j += 1;
    }
    ++i;
  }
  std::cout.flush();
}

} // namespace iir
} // namespace dawn
