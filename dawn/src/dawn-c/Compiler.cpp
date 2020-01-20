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

#include "dawn-c/Compiler.h"
#include "dawn-c/ErrorHandling.h"
#include "dawn-c/util/Allocate.h"
#include "dawn-c/util/CompilerWrapper.h"
#include "dawn-c/util/OptionsWrapper.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/Support/STLExtras.h"
#include "dawn/Support/Unreachable.h"
#include <iostream>
#include <memory>
using namespace dawn::util;

static DawnDiagnosticsKind getDawnDiagnosticsKind(dawn::DiagnosticsKind diag) {
  switch(diag) {
  case dawn::DiagnosticsKind::Note:
    return DD_Note;
  case dawn::DiagnosticsKind::Warning:
    return DD_Warning;
  case dawn::DiagnosticsKind::Error:
    return DD_Error;
  default:
    dawn_unreachable("invalid dawn::DiagnosticsKind");
  }
}

static void dawnDefaultDiagnosticsHandler(DawnDiagnosticsKind diag, int line, int column,
                                          const char* filename, const char* msg) {
  std::cerr << filename << ":" << line << ":" << column << ": ";
  switch(diag) {
  case DD_Note:
    std::cerr << "note";
    break;
  case DD_Warning:
    std::cerr << "warning";
    break;
  case DD_Error:
    std::cerr << "error";
    break;
  default:
    dawn_unreachable("invalid DawnDiagnosticsKind");
  }
  std::cerr << ": " << msg << std::endl;
}

static dawnDiagnosticsHandler_t DiagnosticsHandler = dawnDefaultDiagnosticsHandler;

void dawnReportDiagnostic(DawnDiagnosticsKind diag, int line, int column, const char* filename,
                          const char* msg) {
  DiagnosticsHandler(diag, line, column, filename, msg);
}

void dawnInstallDiagnosticsHandler(dawnDiagnosticsHandler_t handler) {
  DiagnosticsHandler = handler ? handler : dawnDefaultDiagnosticsHandler;
}

dawnTranslationUnit_t* dawnCompile(const char* SIR, size_t size, const dawnOptions_t* options) {
  dawnTranslationUnit_t* translationUnit = nullptr;

  // Deserialize the SIR
  try {
    std::string sirStr(SIR, size);
    auto inMemorySIR =
        dawn::SIRSerializer::deserializeFromString(sirStr, dawn::SIRSerializer::Format::Byte);

    // Prepare options
    std::unique_ptr<dawn::Options> compileOptions = std::make_unique<dawn::Options>();
    if(options)
      toConstOptionsWrapper(options)->setDawnOptions(compileOptions.get());

    // Run the compiler
    dawn::DawnCompiler compiler(*compileOptions.get());
    auto TU = compiler.compile(inMemorySIR);

    // Report diganostics
    if(compiler.getDiagnostics().hasDiags()) {
      for(const auto& diag : compiler.getDiagnostics().getQueue())
        dawnReportDiagnostic(getDawnDiagnosticsKind(diag->getDiagKind()),
                             diag->getSourceLocation().Line, diag->getSourceLocation().Column,
                             diag->getFilename().c_str(), diag->getMessage().c_str());
    }

    if(!TU || compiler.getDiagnostics().hasErrors())
      throw std::runtime_error("compilation failed");

    translationUnit = allocate<dawnTranslationUnit_t>();
    translationUnit->Impl = new dawn::codegen::TranslationUnit(std::move(*TU.get()));
    translationUnit->OwnsData = 1;

  } catch(std::exception& e) {
    dawnFatalError(e.what());
  }

  return translationUnit;
}
