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

#include "dawn/CodeGen/Driver.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Serialization/IIRSerializer.h"

#include <cxxopts.hpp>
#include <fstream>
#include <memory>

enum class SerializationFormat { Byte, Json };

// toString is used because #DEFAULT_VALUE in the options does not work consistently for both string
// and non-string values
template <typename T>
std::string toString(const T& t) {
  return std::to_string(t);
}
std::string toString(const char* t) { return t; }
std::string toString(const std::string& t) { return t; }

std::shared_ptr<dawn::iir::StencilInstantiation> deserializeInput(const std::string& input) {
  std::shared_ptr<dawn::iir::StencilInstantiation> internalIR = nullptr;

  SerializationFormat format = SerializationFormat::Byte;
  {
    try {
      internalIR =
          dawn::IIRSerializer::deserializeFromString(input, dawn::IIRSerializer::Format::Byte);
      format = SerializationFormat::Byte;
    } catch(...) {
      internalIR = nullptr;
    }
  }
  if(!internalIR) {
    try {
      internalIR =
          dawn::IIRSerializer::deserializeFromString(input, dawn::IIRSerializer::Format::Json);
      format = SerializationFormat::Json;
    } catch(...) {
      // Exhausted possibilities, so throw
      throw std::runtime_error("Cannot deserialize input");
    }
  }

  // Deserialize again, only this time do not catch exceptions
  if(format == SerializationFormat::Byte) {
    internalIR =
        dawn::IIRSerializer::deserializeFromString(input, dawn::IIRSerializer::Format::Byte);
  } else {
    internalIR =
        dawn::IIRSerializer::deserializeFromString(input, dawn::IIRSerializer::Format::Json);
  }

  return internalIR;
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("dawn-codegen", "Code generation for the Dawn DSL compiler toolchain");
  options.positional_help("[IIR file. If unset, reads from stdin]");

  // clang-format off
  options.add_options()
    ("i,input", "Input IIR file. If unset, uses stdin.", cxxopts::value<std::string>())
    ("o,out", "Output filename. If unset, writes code to stdout.", cxxopts::value<std::string>())
    ("v,verbose", "Set verbosity level to info. If set, use -o or --out to redirect code to file.")
    ("b,backend", "Backend code generator: [gridtools|gt, c++-naive|naive, cxx-naive-ico|naive-ico, cuda, cuda-ico, cxx-opt].",
        cxxopts::value<std::string>()->default_value("c++-naive"))
    ("h,help", "Display usage.");

    options.add_options("CodeGen")
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  (OPTION, HELP, cxxopts::value<TYPE>()->default_value(toString(DEFAULT_VALUE)))
#include "dawn/CodeGen/Options.inc"
#undef OPT
    ;
  // clang-format on

  // This is how the positional argument is specified
  options.parse_positional({"input"});

  const int numArgs = argc;
  auto result = options.parse(argc, argv);

  if(result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  // Get input = string
  std::string input;
  if(result.count("input") > 0) {
    std::ifstream t(result["input"].as<std::string>());
    input.insert(input.begin(), (std::istreambuf_iterator<char>(t)),
                 std::istreambuf_iterator<char>());
  } else {
    std::istreambuf_iterator<char> begin(std::cin), end;
    input.insert(input.begin(), begin, end);
  }

  auto internalIR = deserializeInput(input);

  std::map<std::string, std::shared_ptr<dawn::iir::StencilInstantiation>> stencilInstantiationMap{
      {"restoredIIR", internalIR}};

  dawn::codegen::Backend backend =
      dawn::codegen::parseBackendString(result["backend"].as<std::string>());

  dawn::codegen::Options codegenOptions;
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  codegenOptions.NAME = result[OPTION].as<TYPE>();
#include "dawn/CodeGen/Options.inc"
#undef OPT
  auto translationUnit = dawn::codegen::run(stencilInstantiationMap, backend, codegenOptions);

  auto code = dawn::codegen::generate(translationUnit);

  if(result.count("out") > 0) {
    std::ofstream out(result["out"].as<std::string>());
    out << code << std::endl;
  } else {
    std::cout << code << std::endl;
  }

  return 0;
}
