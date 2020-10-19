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

#include "dawn/Serialization/SIRSerializer.h"
#include "gtclang/Driver/CompilerInstance.h"
#include "gtclang/Driver/Driver.h"
#include "gtclang/Driver/Options.h"
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
#include <iostream>

int main(int argc, char* argv[]) {
  cxxopts::Options options("gtc-parse",
                           "Geophysical fluid dynamics DSL frontend for the Dawn toolchain");
  options.positional_help("[DSL file. If unset, reads from stdin]");

  // clang-format off
  options.add_options()
    ("i,input", "Input DSL file.", cxxopts::value<std::string>())
    ("f,format", "Output SIR format [json | byte].", cxxopts::value<std::string>()->default_value("json"))
    ("o,out", "Output SIR filename. If unset, writes SIR to stdout.", cxxopts::value<std::string>())
    ("h,help", "Display usage.");

  options.add_options("Frontend")
    ("dump-ast", "Dump the clang AST.", cxxopts::value<bool>()->default_value("false"))
    ("dump-pp", "Dump the preprocessed file.", cxxopts::value<bool>()->default_value("false"));

  // clang-format on
  options.parse_positional({"input"});

  const int numArgs = argc;
  auto result = options.parse(argc, argv);

  if(result.count("help") || numArgs == 1) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  gtclang::ParseOptions parseOptions;

  // Set options from cxxopts
  parseOptions.DumpAST = result["dump-ast"].as<bool>();
  parseOptions.DumpPP = result["dump-pp"].as<bool>();

  // we must call this only once
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::PrettyStackTraceProgram X(argc, argv);

  // Call llvm_shutdown() on exit
  llvm::llvm_shutdown_obj Y;

  // Create SIR as return value
  auto returnSIR = run(result["input"].as<std::string>(), parseOptions);

  // Parse format to enumeration
  dawn::SIRSerializer::Format format;
  const auto sirFormat = result["format"].as<std::string>();
  if(sirFormat == "json") {
    format = dawn::SIRSerializer::Format::Json;
  } else if(sirFormat == "byte") {
    format = dawn::SIRSerializer::Format::Byte;
  } else {
    throw std::runtime_error(std::string("Unknown SIR format: ") + sirFormat +
                             ". Options are [json | byte]");
  }

  // Write SIR to stdout or file
  if(result.count("out") > 0) {
    dawn::SIRSerializer::serialize(result["out"].as<std::string>(), returnSIR.get(), format);
  } else if(result.count("dump-ast") > 0 || result.count("dump-pp") > 0) {
    DAWN_LOG(INFO) << "dump-ast or dump-pp present. Skipping serialization.";
  } else {
    const std::string sirString = dawn::SIRSerializer::serializeToString(returnSIR.get(), format);
    std::cout << sirString;
  }

  return 0;
}
