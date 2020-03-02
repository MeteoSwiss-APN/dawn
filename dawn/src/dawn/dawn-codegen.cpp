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

#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/Compiler/Options.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Serialization/IIRSerializer.h"

#include <cxxopts.hpp>
#include <fstream>
#include <memory>

template <typename T>
std::string toString(const T& t) {
  return std::to_string(t);
}

std::string toString(const char* t) { return t; }

std::string toString(const std::string& t) { return t; }

int main(int argc, char* argv[]) {
  cxxopts::Options options("dawn-codegen", "Code generation for the Dawn DSL compiler toolchain");

  // clang-format off
  options.add_options()
    ("i,input", "Input IIR file. If unset, uses stdin.", cxxopts::value<std::string>())
    ("f,format", "Input IIR format [json, binary].", cxxopts::value<std::string>()->default_value("json"))
    ("o,out", "Output filename. If unset, writes code to stdout.", cxxopts::value<std::string>())
    ("v,verbose", "Set verbosity level to info. If set, use -o or --out to redirect code to file.")
    ("b,backend", "Backend code generator: [gridtools|gt, c++-naive|naive, cxx-naive-ico|naive-ico, cuda, cxx-opt].", cxxopts::value<std::string>()->default_value("c++-naive"))
    ("h,help", "Display usage.")
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  (OPTION, HELP, cxxopts::value<TYPE>()->default_value(toString(DEFAULT_VALUE)))
#include "dawn/CodeGen/Options.inc"
#undef OPT
    ;
  // clang-format on

  options.parse_positional({"input"});

  const int numArgs = argc;
  auto result = options.parse(argc, argv);

  if(result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  // Get input = string
  std::string input;
  if(result.count("input")) {
    std::ifstream t(result["input"].as<std::string>());
    input.insert(input.begin(), (std::istreambuf_iterator<char>(t)),
                 std::istreambuf_iterator<char>());
  } else {
    std::istreambuf_iterator<char> begin(std::cin), end;
    input.insert(input.begin(), begin, end);
  }
  dawn::Options dawnOptions;
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  dawnOptions.NAME = result[OPTION].as<TYPE>();
#include "dawn/CodeGen/Options.inc"
#undef OPT
  dawnOptions.Backend = result["backend"].as<std::string>();
  dawn::DawnCompiler compiler(dawnOptions);

  const std::string formatString = result["format"].as<std::string>();
  const dawn::IIRSerializer::Format format = formatString == "json"
                                                 ? dawn::IIRSerializer::Format::Json
                                                 : dawn::IIRSerializer::Format::Byte;
  std::shared_ptr<dawn::iir::StencilInstantiation> internalIR =
      dawn::IIRSerializer::deserializeFromString(input, format);
  std::map<std::string, std::shared_ptr<dawn::iir::StencilInstantiation>> stencilInstantiationMap;
  stencilInstantiationMap.emplace("restoredIIR", internalIR);
  std::unique_ptr<dawn::codegen::TranslationUnit> translationUnit =
      compiler.generate(stencilInstantiationMap);

  std::string code;
  for(auto p : translationUnit->getPPDefines())
    code += p + "\n";

  code += translationUnit->getGlobals() + "\n\n";
  for(auto p : translationUnit->getStencils())
    code += p.second;

  if(result.count("out") > 0) {
    std::ofstream out(result["out"].as<std::string>());
    out << code << std::endl;
  } else {
    std::cout << code << std::endl;
  }

  return 0;
}
