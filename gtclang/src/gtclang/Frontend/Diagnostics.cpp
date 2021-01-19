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
#define DIAG(ENUM, LEVEL, DESC) {unsigned(-1), LEVEL, DESC},
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

void Diagnostics::reportRaw(clang::DiagnosticsEngine& diag, clang::SourceLocation loc,
                            clang::DiagnosticIDs::Level level, const std::string& msg) {
  auto& diagsID = diag.getDiagnosticIDs();
  unsigned ID = diagsID->getCustomDiagID(level, msg);
  diag.Report(loc, ID);
}

} // namespace gtclang
