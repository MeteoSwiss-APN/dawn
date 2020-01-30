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

  // Initialize the Logger
  auto logger = std::make_unique<Logger>();
  auto* oldLogger = dawn::Logger::getSingleton().getLogger();
  if(context->getOptions().Verbose)
    dawn::Logger::getSingleton().registerLogger(logger.get());

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

  // Cleanup (restore the old logger)
  if(context->getOptions().Verbose)
    dawn::Logger::getSingleton().registerLogger(oldLogger);

  includeChecker.Restore();

  return ReturnValue{ret, returnSIR};
}

} // namespace gtclang
