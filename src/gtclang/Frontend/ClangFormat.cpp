
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

#include "gtclang/Frontend/ClangFormat.h"
#include "gtclang/Frontend/GTClangContext.h"
#include "gtclang/Support/FileUtil.h"
#include "gtclang/Support/Logger.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Version.h"
#include "clang/Format/Format.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"

namespace gtclang {

ClangFormat::ClangFormat(GTClangContext* context) : context_(context) {}

std::string ClangFormat::format(const std::string& code) {
  if(code.empty())
    return code;

  DAWN_LOG(INFO) << "Reformatting stencil code";

  // Create memory buffer of the code
  std::unique_ptr<llvm::MemoryBuffer> codeBuffer = llvm::MemoryBuffer::getMemBuffer(code);

  // Create in-memory FS
  clang::IntrusiveRefCntPtr<clang::vfs::InMemoryFileSystem> memFS(
      new clang::vfs::InMemoryFileSystem);
  clang::FileManager files(clang::FileSystemOptions(), memFS);

  // Create in-memory file
  clang::SourceManager sources(context_->getDiagnosticsEngine(), files);
  clang::FileID ID =
      createInMemoryFile("<irrelevant>", codeBuffer.get(), sources, files, memFS.get());

  clang::SourceLocation start = sources.getLocForStartOfFile(ID);
  clang::SourceLocation end = sources.getLocForEndOfFile(ID);

  unsigned offset = sources.getFileOffset(start);
  unsigned length = sources.getFileOffset(end) - offset;
  std::vector<clang::tooling::Range> ranges{clang::tooling::Range{offset, length}};

  // Run reformat on the entire file (i.e our code snippet)
  bool incompleteFormat = false;
  clang::format::FormatStyle style =
      clang::format::getGoogleStyle(clang::format::FormatStyle::LanguageKind::LK_Cpp);
  style.ColumnLimit = 120;

  clang::tooling::Replacements replacements =
      clang::format::reformat(style, codeBuffer->getBuffer(), ranges, "X.cpp", &incompleteFormat);

  std::string changedCode =
      clang::tooling::applyAllReplacements(codeBuffer->getBuffer(), replacements).get();

  DAWN_LOG(INFO) << "Done reformatting stencil code: "
                 << (changedCode.empty() ? "FAIL" : "Success");
  return changedCode.empty() ? code : changedCode;
}

} // namespace gtclang
