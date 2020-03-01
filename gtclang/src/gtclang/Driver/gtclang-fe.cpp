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

#include "gtclang/Driver/CompilerInstance.h"
#include "gtclang/Driver/Driver.h"
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

#include <cxxopts.hpp>

int main(int argc, char* argv[]) {
  cxxopts::Options options("gtclang-fe",
                           "Geophysical fluid dynamics DSL frontend for the Dawn toolchain");
  options.positional_help("InputFile").show_positional_help();

  // clang-format off
  options.add_options()
    ("f,format", "SIR format [binary,json].", cxxopts::value<std::string>()->default_value("json"))
    ("o,out", "Output SIR filename. If not set, writes SIR to stdout.", cxxopts::value<std::string>())
    ("v,verbose", "Use verbose output. If set, use -o or --out to redirect SIR.")
    ("i,input", "Input DSL file.", cxxopts::value<std::string>())
    ("h,help", "Display usage.");
  // clang-format on
  options.parse_positional({"input"});

  auto result = options.parse(argc, argv);

  if(result.count("help") || argc == 1) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  // Create SIR as return value
  std::shared_ptr<dawn::SIR> returnSIR = nullptr;

  // Initialize the GTClangContext
  std::unique_ptr<gtclang::GTClangContext> context = std::make_unique<gtclang::GTClangContext>();

  // Skip Dawn
  context->useDawn() = false;

  // Set options from cxxopts
  context->getOptions().Verbose = result["verbose"].as<bool>();
  context->getOptions().SIRFormat = result["format"].as<std::string>();

  const std::string InputFile = result["input"].as<std::string>();
  llvm::SmallVector<const char*, 4> clangArgs;
  clangArgs.push_back(argv[0]);
  clangArgs.push_back(InputFile.c_str());

  // we must call this only once
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::PrettyStackTraceProgram X(argc, argv);

  // Call llvm_shutdown() on exit
  llvm::llvm_shutdown_obj Y;

  // Initialize the Logger
  auto logger = std::make_unique<gtclang::Logger>();
  if(context->getOptions().Verbose)
    dawn::Logger::getSingleton().registerLogger(logger.get());

  gtclang::GTClangIncludeChecker includeChecker;
  if(clangArgs.size() > 1)
    includeChecker.Update(clangArgs[1]);

  // Create GTClang
  std::unique_ptr<clang::CompilerInstance> GTClang(gtclang::createCompilerInstance(clangArgs));

  int ret = 0;
  if(GTClang) {
    std::unique_ptr<clang::FrontendAction> PPAction(
        new gtclang::GTClangPreprocessorAction(context.get()));
    ret |= !GTClang->ExecuteAction(*PPAction);

    if(ret == 0) {
      std::unique_ptr<gtclang::GTClangASTAction> ASTAction(
          new gtclang::GTClangASTAction(context.get()));
      ret |= !GTClang->ExecuteAction(*ASTAction);
      returnSIR = ASTAction->getSIR();
    }
    DAWN_LOG(INFO) << "Compilation finished " << (ret ? "with errors" : "successfully");
  }

  // Write SIR to stdout or file

  includeChecker.Restore();

  return 0;
}
