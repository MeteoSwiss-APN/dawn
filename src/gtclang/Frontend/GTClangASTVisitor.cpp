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

#include "gtclang/Frontend/GTClangASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Logging.h"
#include "gtclang/Frontend/GTClangContext.h"
#include "gtclang/Support/FileUtil.h"
#include "clang/AST/ASTContext.h"

namespace gtclang {

GTClangASTVisitor::GTClangASTVisitor(GTClangContext* context)
    : context_(context), globalVariableParser_(context),
      stencilParser_(context, globalVariableParser_) {}

bool GTClangASTVisitor::VisitCXXRecordDecl(clang::CXXRecordDecl* recordDecl) {
  using namespace llvm;
  using namespace clang;

  const auto& SM = context_->getSourceManager();

  // Only parse declaration of the main-file
  if(SM.getFileID(recordDecl->getLocation()) == SM.getMainFileID()) {

    // Check for `globals`
    if(recordDecl->getIdentifier() && recordDecl->getIdentifier()->getName() == "globals") {

      if(globalVariableParser_.isEmpty())
        globalVariableParser_.parseGlobals(recordDecl);
      else {
        context_->getDiagnostics().report(recordDecl->getLocation(),
                                          Diagnostics::err_globals_multiple_definition);
        context_->getDiagnostics().report(globalVariableParser_.getRecordDecl()->getLocation(),
                                          Diagnostics::note_previous_definition);
      }

    } else {
      // Iterate base classes
      for(const CXXBaseSpecifier& base : recordDecl->bases()) {
        StringRef baseTypeName = base.getType().getBaseTypeIdentifier()->getName();

        // Found `gridtools::clang::stencil`
        if(baseTypeName == "stencil") {
          auto name = recordDecl->getIdentifier()->getName().str();

          DAWN_LOG(INFO) << "Parsing stencil `" << name << "` at "
                        << getFilename(base.getLocStart().printToString(SM)).str();

          stencilParser_.parseStencil(recordDecl, name);
        }

        // Found `gridtools::clang::stencil_function`
        else if(baseTypeName == "stencil_function") {
          auto name = recordDecl->getIdentifier()->getName().str();

          DAWN_LOG(INFO) << "Parsing stencil function `" << name << "` at "
                        << getFilename(base.getLocStart().printToString(SM)).str();

          stencilParser_.parseStencilFunction(recordDecl, name);
        }
      }
    }
  }

  return true;
}

} // namespace gtclang
