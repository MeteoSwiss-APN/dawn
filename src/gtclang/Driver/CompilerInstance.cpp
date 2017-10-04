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
#include "dawn/Support/Compiler.h"
#include "gtclang/Support/Config.h"
#include "gtclang/Support/Logger.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include <memory>
#include <string>
#include <utility>

namespace gtclang {

namespace {

std::string getExecutablePath(const char* argv0) {
  // This just needs to be some symbol in the binary;
  // C++ doesn't allow taking the address of ::main however.
  void* main_addr = (void*)(intptr_t)getExecutablePath;
  return llvm::sys::fs::getMainExecutable(argv0, main_addr);
}

} // namespace anonymous

clang::CompilerInstance* createCompilerInstance(llvm::SmallVectorImpl<const char*>& args) {
  using namespace clang;
  using namespace llvm;

  void* mainAddr = (void*)(intptr_t)getExecutablePath;
  std::string path = getExecutablePath(args[0]);

  // Setup diagnostics engine
  IntrusiveRefCntPtr<DiagnosticOptions> diagnosticOptions = new DiagnosticOptions;
  TextDiagnosticPrinter* diagnosticClient = new TextDiagnosticPrinter(errs(), &*diagnosticOptions);

  IntrusiveRefCntPtr<clang::DiagnosticIDs> diagnosticID(new DiagnosticIDs());
  DiagnosticsEngine diagnostics(diagnosticID, &*diagnosticOptions, diagnosticClient);

  // Setup driver
  driver::Driver driver(path, llvm::sys::getDefaultTargetTriple(), diagnostics);
  driver.setTitle("gridtools clang");

  args.push_back("-fsyntax-only");
  std::unique_ptr<driver::Compilation> compilation(driver.BuildCompilation(args));
  if(!compilation)
    return nullptr;

  // We expect to get back exactly one command job, if we didn't something failed. Extract that
  // job from the compilation
  const driver::JobList& jobs = compilation->getJobs();
  if(jobs.size() != 1 || !isa<driver::Command>(*jobs.begin())) {
    llvm::errs() << "error: expected exactly one input file\n";
    return nullptr;
  }

  const driver::Command& command = cast<driver::Command>(*jobs.begin());
  if(StringRef(command.getCreator().getName()) != "clang") {
    diagnostics.Report(clang::diag::err_fe_expected_clang_command);
    return nullptr;
  }

  // Initialize a compiler invocation object from the clang (-cc1) arguments
  driver::ArgStringList& ccArgs = const_cast<driver::ArgStringList&>(command.getArguments());

  // NOTE: This is a kind of a hack. The problem is that Clang tools are meant to be run from the
  // the same binary directory as Clang itself and thus rely on finding the internal header files in
  // `../lib/clang/X.X.X/include`. However, this is usually not the case for us! We just pass the
  // include dirs manually to cc1 which we grabbed from llvm-config in CMake.
  ccArgs.push_back("-internal-isystem");
  ccArgs.push_back(GTCLANG_CLANG_RESSOURCE_INCLUDE_PATH);

#ifdef DAWN_ON_APPLE
  // On Mac OSX it is even more severe as it usually can't even find the STL headers ...
  ccArgs.push_back("-internal-isystem");
  ccArgs.push_back(GTCLANG_CLANG_RESSOURCE_INCLUDE_PATH "/../../../../include/c++/v1/");
#endif

  std::unique_ptr<CompilerInvocation> CI(new CompilerInvocation);
  CompilerInvocation::CreateFromArgs(*CI, const_cast<const char**>(ccArgs.data()),
                                     const_cast<const char**>(ccArgs.data()) + ccArgs.size(),
                                     diagnostics);
  CI->getFrontendOpts().DisableFree = false;

  // Show the invocation, with -v
  if(CI->getHeaderSearchOpts().Verbose) {
    errs() << "gtclang invocation:\n";
    jobs.Print(errs(), "\n", true);
    errs() << "\n";
  }

  // Create a compiler instance to handle the actual work.
  DAWN_LOG(INFO) << "Creating GTClang compiler instance ...";
  CompilerInstance* GTClang = new CompilerInstance;
  GTClang->setInvocation(CI.release());

  // Create the compilers actual diagnostics engine
  GTClang->createDiagnostics();
  if(!GTClang->hasDiagnostics())
    return nullptr;

  // Check that we are atleast in C++11 mode and correct if necessary
  auto& langOpts = GTClang->getLangOpts();
  if(!langOpts.CPlusPlus11 && !langOpts.CPlusPlus14 && !langOpts.CPlusPlus1z) {
    DAWN_LOG(WARNING) << "C++98 mode detected; switchting to C++11";
    langOpts.CPlusPlus11 = 1;
  }

  // Infer the builtin (ressource) include path if unspecified
  if(GTClang->getHeaderSearchOpts().UseBuiltinIncludes &&
     GTClang->getHeaderSearchOpts().ResourceDir.empty())
    GTClang->getHeaderSearchOpts().ResourceDir =
        CompilerInvocation::GetResourcesPath(args[0], mainAddr);

  // Add the path to our DSL runtime
  SmallVector<StringRef, 2> DSLIncludes;
  StringRef(GTCLANG_DSL_INCLUDES).split(DSLIncludes, ';');
  for(const auto& path : DSLIncludes) {
    DAWN_LOG(INFO) << "Adding DSL include path: " << path.str();
    GTClang->getHeaderSearchOpts().AddPath(path, clang::frontend::System, false, false);
  }
  return GTClang;
}

} // namespace gtclang
