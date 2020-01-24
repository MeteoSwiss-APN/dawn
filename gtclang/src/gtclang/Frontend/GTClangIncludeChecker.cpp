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

#include "gtclang/Frontend/GTClangIncludeChecker.h"

namespace gtclang {

bool GTClangIncludeChecker::UpdateHeader(std::string& sourceFile) {
  using clang::StringRef;

  std::vector<std::string> includes = {"gtclang_dsl_defs/gtclang_dsl.hpp"};
  std::vector<std::string> namespaces = {"gtclang::dsl"};

  std::ifstream ifs(sourceFile);
  std::string PPCode((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

  StringRef PPCodeRef(PPCode);
  llvm::SmallVector<StringRef, 100> PPCodeLines;
  PPCodeRef.split(PPCodeLines, '\n');

  for(int i = 0; i < PPCodeLines.size(); ++i) {
    StringRef line = PPCodeLines[i];
    if(line.find("#include") != StringRef::npos) {
      int index = -1;
      for(int j = 0; j < includes.size() && index < 0; j++) {
        if(line.find(includes[j]) != StringRef::npos) {
          index = j;
        }
      }
      if(index >= 0) {
        includes.erase(includes.begin() + index);
      }
    } else if(line.find("namespace") != StringRef::npos) {
      int index = -1;
      for(int j = 0; j < namespaces.size() && index < 0; j++) {
        if(line.find(namespaces[j]) != StringRef::npos) {
          index = j;
        }
      }
      if(index >= 0) {
        namespaces.erase(namespaces.begin() + index);
      }
    } else if(line.find("stencil") != StringRef::npos) {
      break;
    }
  }

  if(includes.size() > 0 || namespaces.size() > 0) {
    size_t pos = sourceFile.rfind('.');
    sourceFile = sourceFile.substr(0, pos) + "~" + sourceFile.substr(pos);

    std::ofstream ofs(sourceFile);
    for(const std::string& include : includes)
      ofs << "#include \"" << include << "\"\n";
    for(const std::string& nspace : namespaces)
      ofs << "using namespace " << nspace << ";\n";

    ofs << "\n";
    for(int i = 0; i < PPCodeLines.size(); ++i)
      ofs << PPCodeLines[i].str() << "\n";
    ofs.close();

    return true;
  }

  return false;
}

} // namespace gtclang
