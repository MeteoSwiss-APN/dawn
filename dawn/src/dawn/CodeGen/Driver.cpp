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

#include "dawn/CodeGen/Driver.h"
#include "dawn/CodeGen/CXXNaive-ico/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/CXXNaive/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/CXXOpt/CXXOptCodeGen.h"
#include "dawn/CodeGen/Cuda-ico/CudaIcoCodeGen.h"
#include "dawn/CodeGen/Cuda/CudaCodeGen.h"
#include "dawn/CodeGen/GridTools/GTCodeGen.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Support/Logger.h"

#include <stdexcept>

#include "dawn/Support/ClangCompat/FileUtil.h"
#include "dawn/Support/ClangCompat/VirtualFileSystem.h"

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Basic/FileManager.h"
#include "clang/Format/Format.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"

namespace dawn {
namespace codegen {

codegen::Backend parseBackendString(const std::string& backendStr) {
  if(backendStr == "gt" || backendStr == "gridtools") {
    return codegen::Backend::GridTools;
  } else if(backendStr == "naive" || backendStr == "cxxnaive" || backendStr == "c++-naive") {
    return codegen::Backend::CXXNaive;
  } else if(backendStr == "opt" || backendStr == "cxxopt" || backendStr == "c++-opt") {
    return codegen::Backend::CXXOpt;
  } else if(backendStr == "ico" || backendStr == "naive-ico" || backendStr == "c++-naive-ico") {
    return codegen::Backend::CXXNaiveIco;
  } else if(backendStr == "cuda" || backendStr == "CUDA") {
    return codegen::Backend::CUDA;
  } else if(backendStr == "cuda-ico" || backendStr == "CUDAIco" || backendStr == "CUDA-Ico" ||
            backendStr == "CUDA-ICO") {
    return codegen::Backend::CUDAIco;
  } else {
    throw std::invalid_argument("Backend not supported");
  }
}

std::unique_ptr<TranslationUnit>
run(const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>& context,
    Backend backend, const Options& options) {
  switch(backend) {
  case Backend::CUDA:
    return cuda::run(context, options);
  case Backend::CXXNaive:
    return cxxnaive::run(context, options);
  case Backend::CXXNaiveIco:
    return cxxnaiveico::run(context, options);
  case Backend::GridTools:
    return gt::run(context, options);
  case Backend::CUDAIco:
    return cudaico::run(context, options);
  case Backend::CXXOpt:
    return cxxopt::run(context, options);
  }
  // This line should not be needed but the compiler seems to complain if it is not present.
  return nullptr;
}

std::string run(const std::map<std::string, std::string>& stencilInstantiationMap,
                dawn::IIRSerializer::Format format, dawn::codegen::Backend backend,
                const dawn::codegen::Options& options) {
  std::map<std::string, std::shared_ptr<dawn::iir::StencilInstantiation>> internalMap;
  for(auto [name, instStr] : stencilInstantiationMap) {
    internalMap.insert(
        std::make_pair(name, dawn::IIRSerializer::deserializeFromString(instStr, format)));
  }
  return dawn::codegen::generate(dawn::codegen::run(internalMap, backend, options));
}

/// @brief Run code generation on a single stencil instantiation
std::unique_ptr<TranslationUnit>
run(const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation, Backend backend,
    const Options& options) {
  return run({{stencilInstantiation->getName(), stencilInstantiation}}, backend, options);
}

std::string generate(const std::unique_ptr<TranslationUnit>& translationUnit) {
  std::string code;
  for(const auto& p : translationUnit->getPPDefines())
    code += p + "\n";

  code += translationUnit->getGlobals() + "\n\n";
  for(const auto& p : translationUnit->getStencils())
    code += p.second;

  // Setup diagnostics engine
  clang::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagnosticOptions = new clang::DiagnosticOptions;
  auto* diagnosticClient = new clang::TextDiagnosticPrinter(llvm::errs(), &*diagnosticOptions);

  clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagnosticID(new clang::DiagnosticIDs());
  clang::DiagnosticsEngine diagnostics(diagnosticID, &*diagnosticOptions, diagnosticClient);

  // Create memory buffer of the code
  std::unique_ptr<llvm::MemoryBuffer> codeBuffer = llvm::MemoryBuffer::getMemBuffer(code);

  // Create in-memory filesystem
  clang::IntrusiveRefCntPtr<clang_compat::llvm::vfs::InMemoryFileSystem> memFS(
      new clang_compat::llvm::vfs::InMemoryFileSystem);
  clang::FileManager files(clang::FileSystemOptions(), memFS);
  memFS->addFileNoOwn("<irrelevant>", 0, codeBuffer.get());

  // Create in-memory file
  clang::SourceManager sources(diagnostics, files);
  clang::FileID ID = sources.createFileID(clang_compat::FileUtil::getFile(files, "<irrelevant>"),
                                          clang::SourceLocation(), clang::SrcMgr::C_User);

  clang::SourceLocation start = sources.getLocForStartOfFile(ID);
  clang::SourceLocation end = sources.getLocForEndOfFile(ID);

  unsigned offset = sources.getFileOffset(start);
  unsigned length = sources.getFileOffset(end) - offset;
  std::vector<clang::tooling::Range> ranges{clang::tooling::Range{offset, length}};

  // Define same style as in .clang-format dawn file
  bool incompleteFormat = false;
  clang::format::FormatStyle style =
      clang::format::getLLVMStyle(clang::format::FormatStyle::LanguageKind::LK_Cpp);
  style.PointerAlignment = clang::format::FormatStyle::PAS_Left;
  style.ColumnLimit = 100;
  style.SpaceBeforeParens = clang::format::FormatStyle::SBPO_Never;
  style.AlwaysBreakTemplateDeclarations = clang::format::FormatStyle::BTDS_Yes;

  // Run reformat on the entire file (i.e our code snippet)
  clang::tooling::Replacements replacements =
      clang::format::reformat(style, codeBuffer->getBuffer(), ranges, "X.cpp", &incompleteFormat);

  auto result_formatted = clang::tooling::applyAllReplacements(codeBuffer->getBuffer(), replacements);
  DAWN_LOG(INFO) << "Done reformatting stencil code: " << (result_formatted.takeError() ? "FAIL" : "Success");

  return result_formatted.get();
}

} // namespace codegen
} // namespace dawn
