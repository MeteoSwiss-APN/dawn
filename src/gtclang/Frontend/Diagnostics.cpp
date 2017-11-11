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

#include "gtclang/Frontend/Diagnostics.h"
#include "dawn/Support/Assert.h"
#include "clang/Basic/SourceManager.h"

namespace gtclang {

namespace {

struct StaticDiagInfoRec {
  unsigned ID;
  clang::DiagnosticIDs::Level Level;
  const char* Desc;
};

} // anonymous namespace

static StaticDiagInfoRec StaticDiagInfo[]{
#define DIAG(ENUM, LEVEL, DESC)                                                                    \
  { unsigned(-1), LEVEL, DESC }                                                                    \
  ,
#include "gtclang/Frontend/DiagnosticsKind.inc"
#undef DIAG
};

Diagnostics::Diagnostics(clang::DiagnosticsEngine* diags) : diags_(diags) {
  DAWN_ASSERT(diags_);
  auto& diagsID = diags_->getDiagnosticIDs();
  for(unsigned i = 0; i < DiagKind::num_diags; ++i)
    StaticDiagInfo[i].ID =
        diagsID->getCustomDiagID(StaticDiagInfo[i].Level, StaticDiagInfo[i].Desc);
}

clang::DiagnosticBuilder Diagnostics::report(clang::SourceLocation loc, DiagKind kind) {
  return diags_->Report(loc, StaticDiagInfo[kind].ID);
}

clang::DiagnosticBuilder Diagnostics::report(clang::SourceRange range, DiagKind kind) {
  return (diags_->Report(clang::SourceLocation(), StaticDiagInfo[kind].ID) << range);
}

void Diagnostics::report(const dawn::DiagnosticsMessage& diagMsg) {

  auto& SM = diags_->getSourceManager();
  auto& diagsID = diags_->getDiagnosticIDs();

  auto makeCustomDiag = [&](dawn::DiagnosticsKind kind, const std::string& msg) -> unsigned {
    switch(kind) {
    case dawn::DiagnosticsKind::Note:
      return diagsID->getCustomDiagID(clang::DiagnosticIDs::Level::Note, msg);
    case dawn::DiagnosticsKind::Warning:
      return diagsID->getCustomDiagID(clang::DiagnosticIDs::Level::Warning, msg);
    case dawn::DiagnosticsKind::Error:
      return diagsID->getCustomDiagID(clang::DiagnosticIDs::Level::Error, msg);
    default:
      llvm_unreachable("invalid diag ID");
    }
  };

  auto makeSourceLocation = [&](const dawn::SourceLocation& loc) -> clang::SourceLocation {
    if(!loc.isValid())
      return clang::SourceLocation();
    else
      return SM.translateLineCol(SM.getMainFileID(), loc.Line, loc.Column);
  };

  diags_->Report(makeSourceLocation(diagMsg.getSourceLocation()),
                 makeCustomDiag(diagMsg.getDiagKind(), diagMsg.getMessage()));
}

void Diagnostics::reportRaw(clang::DiagnosticsEngine& diag, clang::SourceLocation loc,
                            clang::DiagnosticIDs::Level level, const std::string& msg) {
  auto& diagsID = diag.getDiagnosticIDs();
  unsigned ID = diagsID->getCustomDiagID(level, msg);
  diag.Report(loc, ID);
}

} // namespace gtclang
