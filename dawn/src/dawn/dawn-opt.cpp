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
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/Support/FileSystem.h"
#include "dawn/Support/Json.h"

#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <string>

enum class IRType { SIR, IIR };

template <typename T>
std::string toString(const T& t) {
  return std::to_string(t);
}

std::string toString(const char* t) { return t; }

std::string toString(const std::string& t) { return t; }

int main(int argc, char* argv[]) {
  cxxopts::Options options("dawn-opt", "Optimizer for the Dawn DSL compiler toolchain");

  // clang-format off
  options.add_options()
    ("i,input", "Input file. If unset, reads from stdin.", cxxopts::value<std::string>())
    ("f,format", "Input SIR format [json,binary].", cxxopts::value<std::string>()->default_value("json"))
    ("t,type", "Type of input [sir,iir]. Deduced if --format=json.", cxxopts::value<std::string>()->default_value("sir"))
    ("o,out", "Output IIR filename. If unset, writes IIR to stdout.", cxxopts::value<std::string>())
    ("v,verbose", "Set verbosity level to info. If set, use -o or --out to redirect IIR.")
    ("passes", "Ordered list of pass groups to run. See DawnCompiler.h for list.", cxxopts::value<std::vector<std::string>>())
    ("h,help", "Display usage.")
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  (OPTION, HELP, cxxopts::value<TYPE>()->default_value(toString(DEFAULT_VALUE)))
#include "dawn/Optimizer/Options.inc"
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

  // Determine the type (IIR or SIR)
  const std::string formatString = result["format"].as<std::string>();
  IRType type = result["type"].as<std::string>() == "sir" ? IRType::SIR : IRType::IIR;
  if(formatString == "json") {
    // Deduce representation type
    auto json = dawn::json::json::parse(input);
    if(json.count("internalIR") > 0)
      type = IRType::IIR;
  }

  dawn::Options dawnOptions;
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  dawnOptions.NAME = result[OPTION].as<TYPE>();
#include "dawn/Optimizer/Options.inc"
#undef OPT
  dawn::DawnCompiler compiler(dawnOptions);

  std::map<std::string, std::shared_ptr<dawn::iir::StencilInstantiation>> optimizedSIM;

  // Deserialize
  if(type == IRType::SIR) {
    const dawn::SIRSerializer::Format format = formatString == "json"
                                                   ? dawn::SIRSerializer::Format::Json
                                                   : dawn::SIRSerializer::Format::Byte;
    std::shared_ptr<dawn::SIR> stencilIR =
        dawn::SIRSerializer::deserializeFromString(input, format);
    optimizedSIM = compiler.optimize(compiler.lowerToIIR(stencilIR));
  } else {
    const dawn::IIRSerializer::Format format = formatString == "json"
                                                   ? dawn::IIRSerializer::Format::Json
                                                   : dawn::IIRSerializer::Format::Byte;
    std::shared_ptr<dawn::iir::StencilInstantiation> internalIR =
        dawn::IIRSerializer::deserializeFromString(input, format);
    std::map<std::string, std::shared_ptr<dawn::iir::StencilInstantiation>> stencilInstantiationMap;
    stencilInstantiationMap.emplace("restoredIIR", internalIR);
    optimizedSIM = compiler.optimize(stencilInstantiationMap);
  }

  for(auto& [name, instantiation] : optimizedSIM) {
    // TODO Check output format here
    std::cout << dawn::IIRSerializer::serializeToString(instantiation) << std::endl;
  }

  return 0;
}
