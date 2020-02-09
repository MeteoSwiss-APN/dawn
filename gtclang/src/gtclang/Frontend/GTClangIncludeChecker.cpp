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
#include "dawn/Support/FileSystem.h"
#include <fstream>

namespace gtclang {

GTClangIncludeChecker::GTClangIncludeChecker() { updated_ = false; }

void GTClangIncludeChecker::Update(const std::string& sourceFile) {
  using llvm::StringRef;

  // Read the source file
  sourceFile_ = sourceFile;
  std::ifstream ifs(sourceFile_);
  std::string PPCode((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

  // These are the minimum required includes and namespaces to compile the code
  std::vector<std::string> includes = {"gtclang_dsl_defs/gtclang_dsl.hpp"};
  std::vector<std::string> namespaces = {"gtclang::dsl"};

  // Split the preprocessor code into lines
  StringRef PPCodeRef(PPCode);
  llvm::SmallVector<StringRef, 100> PPCodeLines;
  PPCodeRef.split(PPCodeLines, '\n');

  // Scan the header for the required includes and namespaces
  ScanHeader(PPCodeLines, includes, namespaces);

  if(includes.size() > 0 || namespaces.size() > 0)
    WriteFile(PPCodeLines, includes, namespaces);
}

void GTClangIncludeChecker::Restore() {
  if(updated_) {
    fs::copy(sourceFile_ + "~", sourceFile_, fs::copy_options::overwrite_existing);
    std::remove((sourceFile_ + "~").c_str());
  }
}

void GTClangIncludeChecker::ScanHeader(const llvm::SmallVector<llvm::StringRef, 100>& PPCodeLines,
                                       std::vector<std::string>& includes,
                                       std::vector<std::string>& namespaces) {
  using llvm::StringRef;
  // Iterate over the file lines and remove includes and namespaces that already exist
  //   from the lists of required ones.
  int index;
  for(int i = 0; i < PPCodeLines.size(); ++i) {
    StringRef line = PPCodeLines[i];
    if(line.find("#include") != StringRef::npos) {
      index = -1;
      for(int j = 0; j < includes.size() && index < 0; j++) {
        if(line.find(includes[j]) != StringRef::npos)
          index = j;
      }
      if(index >= 0)
        includes.erase(includes.begin() + index);
    } else if(line.find("namespace") != StringRef::npos) {
      index = -1;
      for(int j = 0; j < namespaces.size() && index < 0; j++) {
        if(line.find(namespaces[j]) != StringRef::npos)
          index = j;
      }
      if(index >= 0)
        namespaces.erase(namespaces.begin() + index);
    } else if(line.find("stencil") != StringRef::npos)
      break;
  }
}

void GTClangIncludeChecker::WriteFile(const llvm::SmallVector<llvm::StringRef, 100>& PPCodeLines,
                                      std::vector<std::string>& includes,
                                      std::vector<std::string>& namespaces) {

  // Create a backup of the source file
  std::error_code copyError;
  fs::copy(sourceFile_, sourceFile_ + "~", fs::copy_options::overwrite_existing, copyError);
  updated_ = !copyError;

  // Rewrite the source file with the missing includes and namespaces added
  if(updated_) {
    std::ofstream ofs(sourceFile_);
    for(const std::string& include : includes)
      ofs << "#include \"" << include << "\"\n";
    for(const std::string& nspace : namespaces)
      ofs << "using namespace " << nspace << ";\n";

    ofs << "\n";
    for(int i = 0; i < PPCodeLines.size(); ++i)
      ofs << PPCodeLines[i].str() << "\n";
    ofs.close();
  }
}

} // namespace gtclang
