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

#include "gsl/Compiler/GSLCompiler.h"
#include "gsl/SIR/SIR.h"
#include "gsl/SIR/SIRSerializerJSON.h"
#include "gsl/Support/Config.h"
#include "gsl/Support/Format.h"
#include "gsl/Support/StringUtil.h"
#include "gtclang/Frontend/ClangFormat.h"
#include "gtclang/Frontend/GTClangASTConsumer.h"
#include "gtclang/Frontend/GTClangASTVisitor.h"
#include "gtclang/Frontend/GTClangContext.h"
#include "gtclang/Frontend/GlobalVariableParser.h"
#include "gtclang/Frontend/StencilParser.h"
#include "gtclang/Support/Config.h"
#include "gtclang/Support/FileUtil.h"
#include "gtclang/Support/Logger.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/Version.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstdio>
#include <ctime>
#include <iostream>
#include <memory>
#include <system_error>

namespace gtclang {

/// @brief Get current time-stamp
static const std::string currentDateTime() {
  std::time_t now = time(0);
  char buf[80];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d  %X", std::localtime(&now));
  return buf;
}

/// @brief Extract the GSL options from the GTClang options
static std::unique_ptr<gsl::Options> makeGSLOptions(const Options& options) {
  auto GSLOptions = llvm::make_unique<gsl::Options>();
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  GSLOptions->NAME = options.NAME;
#include "gsl/Compiler/Options.inc"
#undef OPT
  return GSLOptions;
}

GTClangASTConsumer::GTClangASTConsumer(GTClangContext* context, std::string file)
    : context_(context), file_(file) {
  GSL_LOG(INFO) << "Creating ASTVisitor ... ";
  visitor_ = llvm::make_unique<GTClangASTVisitor>(context);
}

void GTClangASTConsumer::HandleTranslationUnit(clang::ASTContext& ASTContext) {
  context_->setASTContext(&ASTContext);
  if(!context_->hasDiagnostics())
    context_->setDiagnostics(&ASTContext.getDiagnostics());

  GSL_LOG(INFO) << "Parsing translation unit... ";

  clang::TranslationUnitDecl* TU = ASTContext.getTranslationUnitDecl();
  for(auto& decl : TU->decls())
    visitor_->TraverseDecl(decl);

  GSL_LOG(INFO) << "Done parsing translation unit";

  if(context_->getASTContext().getDiagnostics().hasErrorOccurred()) {
    GSL_LOG(INFO) << "Erros occurred. Aborting";
    return;
  }

  GSL_LOG(INFO) << "Generating SIR ... ";
  auto& SM = context_->getSourceManager();

  // Assemble SIR
  std::unique_ptr<gsl::SIR> SIR = llvm::make_unique<gsl::SIR>();
  SIR->Filename = SM.getFileEntryForID(SM.getMainFileID())->getName();

  const StencilParser& stencilParser = visitor_->getStencilParser();

  for(const auto& stencilPair : stencilParser.getStencilMap()) {
    SIR->Stencils.emplace_back(stencilPair.second);
    SIR->Stencils.back()->Attributes = context_->getStencilAttribute(stencilPair.second->Name);
  }

  for(const auto& stencilPair : stencilParser.getStencilFunctionMap()) {
    SIR->StencilFunctions.emplace_back(stencilPair.second);
    SIR->StencilFunctions.back()->Attributes =
        context_->getStencilAttribute(stencilPair.second->Name);
  }

  const GlobalVariableParser& globalsParser = visitor_->getGlobalVariableParser();
  SIR->GlobalVariableMap = globalsParser.getGlobalVariableMap();
  
  if(context_->getOptions().DumpSIR) {
    gsl::SIRSerializerJSON::serialize("sir.json", SIR.get());
    SIR->dump();
  }

  // Set the backend
  gsl::GSLCompiler::CodeGenKind codeGen = gsl::GSLCompiler::CG_GTClang;
  if(context_->getOptions().Backend == "gridtools")
    codeGen = gsl::GSLCompiler::CG_GTClang;
  else if(context_->getOptions().Backend == "c++")
    codeGen = gsl::GSLCompiler::CG_GTClangNaiveCXX;
  else {
    context_->getDiagnostics().report(Diagnostics::err_invalid_option)
        << ("-backend=" + context_->getOptions().Backend)
        << gsl::RangeToString(", ", "", "")(std::vector<std::string>{"gridtools", "c++"});
  }

  // Compile the SIR to GridTools
  gsl::GSLCompiler Compiler(makeGSLOptions(context_->getOptions()).get());
  std::unique_ptr<gsl::TranslationUnit> GSLTranslationUnit = Compiler.compile(SIR.get(), codeGen);

  // Report diagnostics from GSL
  if(Compiler.getDiagnostics().hasDiags()) {
    for(const auto& diags : Compiler.getDiagnostics().getQueue()) {
      context_->getDiagnostics().report(*diags);
    }
  }

  if(!GSLTranslationUnit || Compiler.getDiagnostics().hasErrors()) {
    GSL_LOG(INFO) << "Errors in GSL. Aborting";
    return;
  }

  // Do we generate code?
  if(!context_->getOptions().CodeGen) {
    GSL_LOG(INFO) << "Skipping code-generation";
    return;
  }

  // Create new in-memory FS
  llvm::IntrusiveRefCntPtr<clang::vfs::InMemoryFileSystem> memFS(
      new clang::vfs::InMemoryFileSystem);
  clang::FileManager files(clang::FileSystemOptions(), memFS);
  clang::SourceManager sources(context_->getASTContext().getDiagnostics(), files);

  // Get a copy of the main-file's code
  std::unique_ptr<llvm::MemoryBuffer> generatedCode =
      llvm::MemoryBuffer::getMemBufferCopy(SM.getBufferData(SM.getMainFileID()));

  // Determine filename of generated file (by default we append "_gen" to the filename)
  std::string generatedFilename;
  if(context_->getOptions().OutputFile.empty())
    generatedFilename = llvm::StringRef(file_).split(".cpp").first.str() + "_gen.cpp";
  else {
    const auto& OutputFile = context_->getOptions().OutputFile;
    llvm::SmallVector<char, 40> path(OutputFile.data(), OutputFile.data() + OutputFile.size());
    llvm::sys::fs::make_absolute(path);
    generatedFilename.assign(path.data(), path.size());
  }

  // Create the generated file
  GSL_LOG(INFO) << "Creating generated file " << generatedFilename;
  clang::FileID generatedFileID =
      createInMemoryFile(generatedFilename, generatedCode.get(), sources, files, memFS.get());

  // Replace clang DSL with gridtools
  clang::Rewriter rewriter(sources, context_->getASTContext().getLangOpts());
  for(const auto& stencilPair : stencilParser.getStencilMap()) {
    clang::CXXRecordDecl* stencilDecl = stencilPair.first;
    if(rewriter.ReplaceText(stencilDecl->getSourceRange(),
                            stencilPair.second->Attributes.has(gsl::sir::Attr::AK_NoCodeGen)
                                ? "class " + stencilPair.second->Name + "{}"
                                : GSLTranslationUnit->getStencils().at(stencilPair.second->Name)))
      context_->getDiagnostics().report(Diagnostics::err_fs_error) << gsl::format(
          "unable to replace stencil code at: %s", stencilDecl->getLocation().printToString(SM));
  }

  // Replace globals struct
  if(!globalsParser.isEmpty() && !GSLTranslationUnit->getGlobals().empty()) {
    if(rewriter.ReplaceText(globalsParser.getRecordDecl()->getSourceRange(),
                            GSLTranslationUnit->getGlobals()))
      context_->getDiagnostics().report(Diagnostics::err_fs_error)
          << gsl::format("unable to replace globals code at: %s",
                         globalsParser.getRecordDecl()->getLocation().printToString(SM));
  }

  // Remove the code from stencil-functions
  for(const auto& stencilFunPair : stencilParser.getStencilFunctionMap()) {
    clang::CXXRecordDecl* stencilFunDecl = stencilFunPair.first;
    rewriter.ReplaceText(stencilFunDecl->getSourceRange(),
                         "class " + stencilFunPair.second->Name + "{}");
  }

  std::string code;
  llvm::raw_string_ostream os(code);
  rewriter.getEditBuffer(generatedFileID).write(os);
  os.flush();
  
  // Format the file
  if(context_->getOptions().ClangFormat) {
    ClangFormat clangformat(context_);
    code = clangformat.format(code);
  }

  // Write file to disk
  GSL_LOG(INFO) << "Writing file to disk... ";
  std::error_code ec;
  llvm::sys::fs::OpenFlags flags = llvm::sys::fs::OpenFlags::F_RW;
  llvm::raw_fd_ostream fout(generatedFilename, ec, flags);

  // Print a header
  fout << gsl::format("// gtclang (%s) based on LLVM/Clang (%s), GSL (%s)\n",
                      GTCLANG_VERSION_STRING, LLVM_VERSION_STRING, GSL_VERSION_STRING);
  fout << "// Generated on " << currentDateTime() << "\n\n";

  // Add the macro definitions
  for(const auto& macroDefine : GSLTranslationUnit->getPPDefines())
    fout << macroDefine << "\n";

  fout.write(code.data(), code.size());
  if(ec.value())
    context_->getDiagnostics().report(Diagnostics::err_fs_error) << ec.message();
}

} // namespace gtclang
