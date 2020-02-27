//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _       _
//                        | |     | |
//                    __ _| |_ ___| | __ _ _ __   __ _
//                   / _` | __/ __| |/ _` | '_ \ / _` |
//                  | (_| | || (__| | (_| | | | | (_| |
//                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
//                    __/ |                       __/ |
//                   |___/                       |___/
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gtclang/Driver/Driver.h"
#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/Compiler/Options.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/Support/FileSystem.h"
#include "gtclang/Driver/CompilerInstance.h"
#include "gtclang/Driver/OptionsParser.h"
#include "gtclang/Frontend/GTClangASTAction.h"
#include "gtclang/Frontend/GTClangContext.h"
#include "gtclang/Frontend/GTClangIncludeChecker.h"
#include "gtclang/Frontend/GTClangPreprocessorAction.h"
#include "gtclang/Support/ClangCompat/FileSystem.h"
#include "gtclang/Support/Config.h"
#include "gtclang/Support/Logger.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_os_ostream.h"
#include <cstdio>
#include <ctime>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>

namespace {

/// @brief Get current time-stamp
const std::string currentDateTime() {
  std::time_t now = time(0);
  char buf[80];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d  %X", std::localtime(&now));
  return buf;
}

/// @brief Extract the DAWN options from the GTClang options
dawn::Options makeDawnOptions(const gtclang::Options& options) {
  dawn::Options dawnOptions;
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  dawnOptions.NAME = options.NAME;
#include "dawn/Compiler/Options.inc"
#undef OPT
  return dawnOptions;
}

} // anonymous namespace

namespace gtclang {

bool Driver::isInitialized = false;

ReturnValue Driver::run(const llvm::SmallVectorImpl<const char*>& args) {
  // Print a stack trace if we signal out
  if(!isInitialized) {
    // we must call this only once, otherwise we register a signal on each call
    llvm::sys::PrintStackTraceOnErrorSignal(args[0]);
    isInitialized = true;
  }
  llvm::PrettyStackTraceProgram X(args.size(), args.data());

  // Call llvm_shutdown() on exit
  llvm::llvm_shutdown_obj Y;

  // Create SIR as return value
  std::shared_ptr<dawn::SIR> SIR = nullptr;

  // Initialize the GTClangContext
  std::unique_ptr<GTClangContext> context = std::make_unique<GTClangContext>();

  // Parse command-line options
  OptionsParser optionsParser(&context->getOptions());
  llvm::SmallVector<const char*, 16> clangArgs;
  if(!optionsParser.parse(args, clangArgs))
    return ReturnValue{1, SIR};

  // Initialize the Logger
  auto logger = std::make_unique<Logger>();
  if(context->getOptions().Verbose)
    dawn::Logger::getSingleton().registerLogger(logger.get());

  const std::string fileName = clangArgs[1];

  GTClangIncludeChecker includeChecker;
  if(clangArgs.size() > 1)
    includeChecker.Update(fileName);

  // Create GTClang
  std::unique_ptr<clang::CompilerInstance> GTClang(createCompilerInstance(clangArgs));

  int ret = 0;
  if(GTClang) {
    std::unique_ptr<clang::FrontendAction> PPAction(new GTClangPreprocessorAction(context.get()));
    ret |= !GTClang->ExecuteAction(*PPAction);

    if(ret == 0) {
      std::unique_ptr<GTClangASTAction> ASTAction(new GTClangASTAction(context.get()));
      ret |= !GTClang->ExecuteAction(*ASTAction);
      SIR = ASTAction->getSIR();
    }

    if(ret == 0) {
      auto options = makeDawnOptions(context->getOptions());
      dawn::DawnCompiler Compiler(options);
      auto DawnTranslationUnit = Compiler.compile(SIR);

      // Report diagnostics from Dawn
      if(Compiler.getDiagnostics().hasDiags()) {
        for(const auto& diags : Compiler.getDiagnostics().getQueue()) {
          context->getDiagnostics().report(*diags);
        }
        DAWN_LOG(INFO) << "Errors in Dawn. Aborting";
      }

      // Determine filename of generated file (by default we append "_gen" to the filename)
      std::string generatedFileName;
      if(context->getOptions().OutputFile.empty())
        generatedFileName = std::string(fs::path(fileName).filename().stem()) + "_gen.cpp";
      else {
        generatedFileName = context->getOptions().OutputFile;
      }

      std::string code = DawnTranslationUnit->getGlobals() + "\n\n";
      for(auto p : DawnTranslationUnit->getStencils()) {
        code += p.second;
      }

      if(context->getOptions().WriteSIR) {
        const std::string generatedSIR =
            std::string(fs::path(generatedFileName).filename().stem()) + ".sir";
        DAWN_LOG(INFO) << "Generating SIR file " << generatedSIR;

        if(context->getOptions().SIRFormat == "json") {
          dawn::SIRSerializer::serialize(generatedSIR, SIR.get(),
                                         dawn::SIRSerializer::Format::Json);
        } else if(context->getOptions().SIRFormat == "byte") {
          dawn::SIRSerializer::serialize(generatedSIR, SIR.get(),
                                         dawn::SIRSerializer::Format::Byte);

        } else {
          dawn_unreachable("Unknown SIRFormat option");
        }
      }

      // Do we generate code?
      if(!context->getOptions().CodeGen) {
        DAWN_LOG(INFO) << "Skipping code-generation";
        return {0, SIR};
      }

      std::shared_ptr<llvm::raw_ostream> ost;
      std::error_code ec;
      llvm::sys::fs::OpenFlags flags = clang_compat::llvm::sys::fs::OpenFlags::OF_Text;
      if(context->getOptions().OutputFile.empty()) {
        ost = std::make_shared<llvm::raw_os_ostream>(std::cout);
      } else {
        ost = std::make_shared<llvm::raw_fd_ostream>(generatedFileName, ec, flags);
      }

      // Write file to specified output
      DAWN_LOG(INFO) << "Writing file to output... ";

      // Print a header
      *ost << dawn::format("// gtclang (%s)\n// based on LLVM/Clang (%s), Dawn (%s)\n",
                           GTCLANG_FULL_VERSION_STR, LLVM_VERSION_STRING, DAWN_VERSION_STR);
      *ost << "// Generated on " << currentDateTime() << "\n\n";

      // Add the macro definitions
      for(const auto& macroDefine : DawnTranslationUnit->getPPDefines())
        *ost << macroDefine << "\n";

      ost->write(code.data(), code.size());
      if(ec.value())
        context->getDiagnostics().report(Diagnostics::err_fs_error) << ec.message();
    }
    DAWN_LOG(INFO) << "Compilation finished " << (ret ? "with errors" : "successfully");
  }

  includeChecker.Restore();

  return ReturnValue{ret, SIR};
}

} // namespace gtclang
