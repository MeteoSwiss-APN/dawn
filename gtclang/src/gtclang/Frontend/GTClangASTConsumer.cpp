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

#include "gtclang/Frontend/GTClangASTConsumer.h"
#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/Support/Config.h"
#include "dawn/Support/FileSystem.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/StringUtil.h"
#include "dawn/Support/Unreachable.h"
#include "gtclang/Frontend/ClangFormat.h"
#include "gtclang/Frontend/GTClangASTAction.h"
#include "gtclang/Frontend/GTClangASTVisitor.h"
#include "gtclang/Frontend/GTClangContext.h"
#include "gtclang/Frontend/GlobalVariableParser.h"
#include "gtclang/Frontend/StencilParser.h"
#include "gtclang/Support/ClangCompat/FileSystem.h"
#include "gtclang/Support/ClangCompat/VirtualFileSystem.h"
#include "gtclang/Support/Config.h"
#include "gtclang/Support/FileUtil.h"
#include "gtclang/Support/Logger.h"
#include "gtclang/Support/StringUtil.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Basic/Version.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_os_ostream.h"
#include <cstdio>
#include <ctime>
#include <iostream>
#include <memory>
#include <system_error>

namespace gtclang {

namespace {

/// @brief Get current time-stamp
static const std::string currentDateTime() {
  std::time_t now = time(0);
  char buf[80];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d  %X", std::localtime(&now));
  return buf;
}

/// @brief Extract the DAWN options from the GTClang options
static dawn::Options makeDAWNOptions(const Options& options) {
  dawn::Options DAWNOptions;
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  DAWNOptions.NAME = options.NAME;
#include "dawn/Compiler/Options.inc"
#undef OPT
  return DAWNOptions;
}

} // anonymous namespace

GTClangASTConsumer::GTClangASTConsumer(GTClangContext* context, const std::string& file,
                                       GTClangASTAction* parentAction)
    : context_(context), file_(file), parentAction_(parentAction) {
  DAWN_LOG(INFO) << "Creating ASTVisitor ... ";
  visitor_ = std::make_unique<GTClangASTVisitor>(context);
}

void GTClangASTConsumer::HandleTranslationUnit(clang::ASTContext& ASTContext) {
  context_->setASTContext(&ASTContext);
  if(!context_->hasDiagnostics())
    context_->setDiagnostics(&ASTContext.getDiagnostics());

  DAWN_LOG(INFO) << "Parsing translation unit... ";

  clang::TranslationUnitDecl* TU = ASTContext.getTranslationUnitDecl();

  if(context_->getOptions().DumpAST) {
    TU->dumpColor();
  }

  for(auto& decl : TU->decls())
    visitor_->TraverseDecl(decl);

  DAWN_LOG(INFO) << "Done parsing translation unit";

  if(context_->getASTContext().getDiagnostics().hasErrorOccurred()) {
    DAWN_LOG(INFO) << "Erros occurred. Aborting";
    return;
  }

  DAWN_LOG(INFO) << "Generating SIR ... ";
  auto& SM = context_->getSourceManager();

  // Assemble SIR
  std::shared_ptr<dawn::SIR> SIR = std::make_shared<dawn::SIR>(dawn::ast::GridType::Cartesian);
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
    SIR->dump();
  }

  parentAction_->setSIR(SIR);

  // Compile the SIR to GridTools
  dawn::DawnCompiler Compiler(makeDAWNOptions(context_->getOptions()));
  std::unique_ptr<dawn::codegen::TranslationUnit> DawnTranslationUnit = Compiler.compile(SIR);

  // Report diagnostics from Dawn
  if(Compiler.getDiagnostics().hasDiags()) {
    for(const auto& diags : Compiler.getDiagnostics().getQueue()) {
      context_->getDiagnostics().report(*diags);
    }
  }

  if(!DawnTranslationUnit || Compiler.getDiagnostics().hasErrors()) {
    DAWN_LOG(INFO) << "Errors in Dawn. Aborting";
    return;
  }

  // Determine filename of generated file (by default we append "_gen" to the filename)
  std::string generatedFileName;
  if(context_->getOptions().OutputFile.empty())
    generatedFileName = std::string(fs::path(file_).filename().stem()) + "_gen.cpp";
  else {
    generatedFileName = context_->getOptions().OutputFile;
  }

  if(context_->getOptions().WriteSIR) {
    const std::string generatedSIR =
        std::string(fs::path(generatedFileName).filename().stem()) + ".sir";
    DAWN_LOG(INFO) << "Generating SIR file " << generatedSIR;

    if(context_->getOptions().SIRFormat == "json") {
      dawn::SIRSerializer::serialize(generatedSIR, SIR.get(), dawn::SIRSerializer::Format::Json);
    } else if(context_->getOptions().SIRFormat == "byte") {
      dawn::SIRSerializer::serialize(generatedSIR, SIR.get(), dawn::SIRSerializer::Format::Byte);

    } else {
      dawn_unreachable("Unknown SIRFormat option");
    }
  }

  // Do we generate code?
  if(!context_->getOptions().CodeGen) {
    DAWN_LOG(INFO) << "Skipping code-generation";
    return;
  }

  // Create new in-memory FS
  llvm::IntrusiveRefCntPtr<clang_compat::llvm::vfs::InMemoryFileSystem> memFS(
      new clang_compat::llvm::vfs::InMemoryFileSystem);
  clang::FileManager files(clang::FileSystemOptions(), memFS);
  context_->getASTContext().getDiagnostics().Reset();
  clang::SourceManager sources(context_->getASTContext().getDiagnostics(), files);

  std::string code;
  if(context_->getOptions().Serialized) {
    DAWN_LOG(INFO) << "Data was loaded from serialized IR, codegen ";
    std::cout << "Data was loaded from serialized IR, codegen " << std::endl;

    code += DawnTranslationUnit->getGlobals();

    code += "\n\n";

    for(auto p : DawnTranslationUnit->getStencils()) {
      code += p.second;
    }
  } else {
    int num_stencils_generated = 0;

    // Get a copy of the main-file's code
    std::unique_ptr<llvm::MemoryBuffer> generatedCode =
        llvm::MemoryBuffer::getMemBufferCopy(SM.getBufferData(SM.getMainFileID()));

    // Create the generated file
    DAWN_LOG(INFO) << "Creating generated file " << generatedFileName;
    clang::FileID generatedFileID =
        createInMemoryFile(generatedFileName, generatedCode.get(), sources, files, memFS.get());

    // Replace clang DSL with gridtools
    clang::Rewriter rewriter(sources, context_->getASTContext().getLangOpts());
    for(const auto& stencilPair : stencilParser.getStencilMap()) {
      clang::CXXRecordDecl* stencilDecl = stencilPair.first;
      bool skipNewLines = false;
      auto semiAfterDef = clang::Lexer::findLocationAfterToken(
          stencilDecl->getSourceRange().getEnd(), clang::tok::semi, sources,
          context_->getASTContext().getLangOpts(), skipNewLines);
      if(rewriter.ReplaceText(
             clang::SourceRange(stencilDecl->getSourceRange().getBegin(), semiAfterDef),
             stencilPair.second->Attributes.has(dawn::sir::Attr::Kind::NoCodeGen)
                 ? ""
                 : DawnTranslationUnit->getStencils().at(
                       DawnTranslationUnit->getStencils().count("<restored>") > 0
                           ? "<restored>"
                           : stencilPair.second->Name))) {
        context_->getDiagnostics().report(Diagnostics::err_fs_error) << dawn::format(
            "unable to replace stencil code at: %s", stencilDecl->getLocation().printToString(SM));
      } else {
        num_stencils_generated++;
      }
      if(context_->getOptions().DeserializeIIR != "" && num_stencils_generated > 1) {
        DAWN_LOG(ERROR)
            << "more than one stencil present in DSL but only one stencil deserialized from IIR";
        return;
      }
    }

    // Replace globals struct
    if(!globalsParser.isEmpty() && !DawnTranslationUnit->getGlobals().empty()) {
      bool skipNewLines = false;
      auto semiAfterDef = clang::Lexer::findLocationAfterToken(
          globalsParser.getRecordDecl()->getSourceRange().getEnd(), clang::tok::semi, sources,
          context_->getASTContext().getLangOpts(), skipNewLines);
      if(rewriter.ReplaceText(
             clang::SourceRange(globalsParser.getRecordDecl()->getSourceRange().getBegin(),
                                semiAfterDef),
             DawnTranslationUnit->getGlobals()))
        context_->getDiagnostics().report(Diagnostics::err_fs_error)
            << dawn::format("unable to replace globals code at: %s",
                            globalsParser.getRecordDecl()->getLocation().printToString(SM));
    }

    // Replace interval
    for(const clang::VarDecl* a : visitor_->getIntervalDecls()) {
      bool skipNewLines = false;
      auto semiAfterDef = clang::Lexer::findLocationAfterToken(
          a->getSourceRange().getEnd(), clang::tok::semi, sources,
          context_->getASTContext().getLangOpts(), skipNewLines);
      rewriter.ReplaceText(clang::SourceRange(a->getSourceRange().getBegin(), semiAfterDef), "");
    }

    // Remove the code from stencil-functions
    for(const auto& stencilFunPair : stencilParser.getStencilFunctionMap()) {
      clang::CXXRecordDecl* stencilFunDecl = stencilFunPair.first;
      bool skipNewLines = false;
      auto semiAfterDef = clang::Lexer::findLocationAfterToken(
          stencilFunDecl->getSourceRange().getEnd(), clang::tok::semi, sources,
          context_->getASTContext().getLangOpts(), skipNewLines);
      rewriter.ReplaceText(
          clang::SourceRange(stencilFunDecl->getSourceRange().getBegin(), semiAfterDef), "");
    }
    llvm::raw_string_ostream os(code);
    rewriter.getEditBuffer(generatedFileID).write(os);
    os.flush();
  }

  // Format the file
  if(context_->getOptions().ClangFormat) {
    ClangFormat clangformat(context_);
    code = clangformat.format(code);
  }

  std::shared_ptr<llvm::raw_ostream> ost;
  std::error_code ec;
  llvm::sys::fs::OpenFlags flags = clang_compat::llvm::sys::fs::OpenFlags::OF_Text;
  if(context_->getOptions().OutputFile.empty()) {
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
    context_->getDiagnostics().report(Diagnostics::err_fs_error) << ec.message();
}

} // namespace gtclang
