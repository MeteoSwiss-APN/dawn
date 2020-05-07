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
#include "gtclang/Driver/CompilerInstance.h"
#include "gtclang/Driver/OptionsParser.h"
#include "gtclang/Frontend/GTClangASTAction.h"
#include "gtclang/Frontend/GTClangContext.h"
#include "gtclang/Frontend/GTClangIncludeChecker.h"
#include "gtclang/Frontend/GTClangPreprocessorAction.h"
#include "gtclang/Support/Logger.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"

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
  std::shared_ptr<dawn::SIR> returnSIR = nullptr;

  // Initialize the GTClangContext
  std::unique_ptr<GTClangContext> context = std::make_unique<GTClangContext>();

  // Parse command-line options
  OptionsParser optionsParser(&context->getOptions());
  llvm::SmallVector<const char*, 16> clangArgs;
  if(!optionsParser.parse(args, clangArgs))
    return ReturnValue{1, returnSIR};

  // Save existing formatter and set to gtclang
  auto infoFormatter = dawn::info.formatter();
  auto warnFormatter = dawn::warn.formatter();
  dawn::info.formatter(makeGTClangFormatter("[INFO]"));
  dawn::warn.formatter(makeGTClangFormatter("[WARNING]"));

  GTClangIncludeChecker includeChecker;
  if(clangArgs.size() > 1)
    includeChecker.Update(clangArgs[1]);

  // Create GTClang
  std::unique_ptr<clang::CompilerInstance> GTClang(createCompilerInstance(clangArgs));

  int ret = 0;
  if(GTClang) {
    std::unique_ptr<clang::FrontendAction> PPAction(new GTClangPreprocessorAction(context.get()));
    ret |= !GTClang->ExecuteAction(*PPAction);

    if(ret == 0) {
      std::unique_ptr<GTClangASTAction> ASTAction(new GTClangASTAction(context.get()));
      ret |= !GTClang->ExecuteAction(*ASTAction);
      returnSIR = ASTAction->getSIR();
    }
    DAWN_LOG(INFO) << "Compilation finished " << (ret ? "with errors" : "successfully");
  }

  includeChecker.Restore();

  // Reset formatters
  dawn::info.formatter(infoFormatter);
  dawn::warn.formatter(warnFormatter);

  return ReturnValue{ret, returnSIR};
}

std::shared_ptr<dawn::SIR> run(const std::string& fileName, const ParseOptions& options) {
  // Initialize the GTClangContext
  auto context = std::make_unique<gtclang::GTClangContext>();

  // Skip Dawn
  context->useDawn() = false;

  // Set options that were passed in
  context->getOptions().DumpPP = options.DumpPP;
  context->getOptions().ConfigFile = options.ConfigFile;
  context->getOptions().DumpAST = options.DumpAST;
  context->getOptions().ReportPassPreprocessor = options.ReportPassPreprocessor;
  context->getOptions().Verbose = options.Verbose;

  llvm::SmallVector<const char*, 4> clangArgs;
  clangArgs.push_back("gtc-parse");
  clangArgs.push_back(fileName.c_str());

  // Save existing formatter and set to gtclang
  auto infoFormatter = dawn::info.formatter();
  auto warnFormatter = dawn::warn.formatter();
  dawn::info.formatter(makeGTClangFormatter("[INFO]"));
  dawn::warn.formatter(makeGTClangFormatter("[WARNING]"));

  gtclang::GTClangIncludeChecker includeChecker;
  if(clangArgs.size() > 1)
    includeChecker.Update(clangArgs[1]);

  // Create GTClang
  std::unique_ptr<clang::CompilerInstance> GTClang(gtclang::createCompilerInstance(clangArgs));

  // Create SIR as return value
  std::shared_ptr<dawn::SIR> stencilIR = nullptr;

  int ret = 0;
  if(GTClang) {
    std::unique_ptr<clang::FrontendAction> PPAction(
        new gtclang::GTClangPreprocessorAction(context.get()));
    ret |= !GTClang->ExecuteAction(*PPAction);

    if(ret == 0) {
      std::unique_ptr<gtclang::GTClangASTAction> ASTAction(
          new gtclang::GTClangASTAction(context.get()));
      ret |= !GTClang->ExecuteAction(*ASTAction);
      stencilIR = ASTAction->getSIR();
    }
    DAWN_LOG(INFO) << "Compilation finished " << (ret ? "with errors" : "successfully");
  }

  includeChecker.Restore();

  // Reset formatters
  dawn::info.formatter(infoFormatter);
  dawn::warn.formatter(warnFormatter);

  return stencilIR;
}

} // namespace gtclang
