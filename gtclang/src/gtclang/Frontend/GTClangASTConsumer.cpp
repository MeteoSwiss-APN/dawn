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
#include "dawn/AST/GridType.h"
#include "dawn/CodeGen/Driver.h"
#include "dawn/CodeGen/TranslationUnit.h"
#include "dawn/Compiler/Driver.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/Support/FileSystem.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/Logger.h"
#include "gtclang/Frontend/ClangFormat.h"
#include "gtclang/Frontend/Diagnostics.h"
#include "gtclang/Frontend/GTClangASTAction.h"
#include "gtclang/Frontend/GTClangASTVisitor.h"
#include "gtclang/Frontend/GTClangContext.h"
#include "gtclang/Support/ClangCompat/FileSystem.h"
#include "gtclang/Support/ClangCompat/VirtualFileSystem.h"
#include "gtclang/Support/Config.h"
#include "gtclang/Support/FileUtil.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <memory>
#include <string>

namespace gtclang {

namespace {

/// @brief Get current time-stamp
static const std::string currentDateTime() {
  std::time_t now = time(0);
  char buf[80];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d  %X", std::localtime(&now));
  return buf;
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
    DAWN_LOG(INFO) << "Errors occurred. Aborting";
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
    SIR->dump(std::cout);
  }

  parentAction_->setSIR(SIR);

  // Return early if not using dawn (gtc-parse)
  if(!context_->useDawn())
    return;

  // Compile the SIR using Dawn
  std::list<dawn::PassGroup> passGroup;

  if(context_->getOptions().SSA)
    passGroup.push_back(dawn::PassGroup::SSA);

  if(context_->getOptions().PrintStencilGraph)
    passGroup.push_back(dawn::PassGroup::PrintStencilGraph);

  if(context_->getOptions().SetStageName || context_->getOptions().DefaultOptimization)
    passGroup.push_back(dawn::PassGroup::SetStageName);

  if(context_->getOptions().StageReordering || context_->getOptions().DefaultOptimization)
    passGroup.push_back(dawn::PassGroup::StageReordering);

  if(context_->getOptions().MergeStages || context_->getOptions().DefaultOptimization)
    passGroup.push_back(dawn::PassGroup::StageMerger);

  if(std::any_of(SIR->Stencils.begin(), SIR->Stencils.end(),
                 [](const std::shared_ptr<dawn::sir::Stencil>& stencilPtr) {
                   return stencilPtr->Attributes.has(dawn::sir::Attr::Kind::MergeTemporaries);
                 }) ||
     context_->getOptions().TemporaryMerger) {
    passGroup.push_back(dawn::PassGroup::TemporaryMerger);
  }

  if(context_->getOptions().Inlining)
    passGroup.push_back(dawn::PassGroup::Inlining);

  if(context_->getOptions().IntervalPartitioning)
    passGroup.push_back(dawn::PassGroup::IntervalPartitioning);

  if(context_->getOptions().TmpToStencilFunction)
    passGroup.push_back(dawn::PassGroup::TmpToStencilFunction);

  if(context_->getOptions().SetNonTempCaches)
    passGroup.push_back(dawn::PassGroup::SetNonTempCaches);

  if(context_->getOptions().SetCaches || context_->getOptions().DefaultOptimization)
    passGroup.push_back(dawn::PassGroup::SetCaches);

  if(context_->getOptions().SetBlockSize || context_->getOptions().DefaultOptimization)
    passGroup.push_back(dawn::PassGroup::SetBlockSize);

  if(context_->getOptions().DataLocalityMetric)
    passGroup.push_back(dawn::PassGroup::DataLocalityMetric);

  if(context_->getOptions().SetLoopOrder)
    passGroup.push_back(dawn::PassGroup::SetLoopOrder);

  if(context_->getOptions().MultiStageMerger)
    passGroup.push_back(dawn::PassGroup::MultiStageMerger);

  // if nothing is passed, we fill the group with the default if no-optimization is not specified
  if(!context_->getOptions().DisableOptimization && passGroup.size() == 0)
    passGroup = dawn::defaultPassGroups();

  if(context_->getOptions().DisableOptimization && passGroup.size() > 0)
    DAWN_ASSERT_MSG(false, "Inconsistent arguments: no-opt present together with optimization");

  // Inline at end if serializing or if the codegen backend is CUDA
  if(context_->getOptions().SerializeIIR ||
     (context_->getOptions().CodeGen &&
      dawn::codegen::parseBackendString(context_->getOptions().Backend) ==
          dawn::codegen::Backend::CUDA))
    passGroup.push_back(dawn::PassGroup::Inlining);

  // Determine filename of generated file (by default we append "_gen" to the filename)
  const std::string generatedPrefix = fs::path(file_).filename().stem();
  std::string generatedFileName;
  if(context_->getOptions().OutputFile.empty()) {
    generatedFileName = generatedPrefix + "_gen.cpp";
  } else {
    generatedFileName = context_->getOptions().OutputFile;
  }

  dawn::Options optimizerOptions;
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  optimizerOptions.NAME = context_->getOptions().NAME;
#include "dawn/Optimizer/Options.inc"
#undef OPT
  dawn::codegen::Options codegenOptions;
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  codegenOptions.NAME = context_->getOptions().NAME;
#include "dawn/CodeGen/Options.inc"
#undef OPT

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

  auto stencilInstantiationMap = dawn::run(SIR, passGroup, optimizerOptions);

  // Do we generate code?
  if(!context_->getOptions().CodeGen) {
    DAWN_LOG(INFO) << "Skipping code generation";
    return;
  }

  auto DawnTranslationUnit = dawn::codegen::run(
      stencilInstantiationMap, dawn::codegen::parseBackendString(context_->getOptions().Backend),
      codegenOptions);

  // Create new in-memory FS
  llvm::IntrusiveRefCntPtr<clang_compat::llvm::vfs::InMemoryFileSystem> memFS(
      new clang_compat::llvm::vfs::InMemoryFileSystem);
  clang::FileManager files(clang::FileSystemOptions(), memFS);
  context_->getASTContext().getDiagnostics().Reset();
  clang::SourceManager sources(context_->getASTContext().getDiagnostics(), files);

  std::string code;
  if(context_->getOptions().Serialized) {
    DAWN_LOG(INFO) << "Data was loaded from serialized IR, codegen ";
    // Would call generate here but that adds PPDefines as well
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
