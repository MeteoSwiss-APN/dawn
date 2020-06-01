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

#include "dawn/Compiler/Driver.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/Driver.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/Support/FileSystem.h"
#include "dawn/Support/Logger.h"

#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>

enum class SerializationFormat { Byte, Json };
enum class IRType { SIR, IIR };

// toString is used because #DEFAULT_VALUE in the options does not work consistently for both string
// and non-string values
template <typename T>
std::string toString(const T& t) {
  return std::to_string(t);
}
std::string toString(const char* t) { return t; }
std::string toString(const std::string& t) { return t; }

dawn::PassGroup parsePassGroup(const std::string& passGroup) {
  if(passGroup == "SSA" || passGroup == "ssa")
    return dawn::PassGroup::SSA;
  else if(passGroup == "PrintStencilGraph" || passGroup == "print-stencil-graph")
    return dawn::PassGroup::PrintStencilGraph;
  else if(passGroup == "SetStageName" || passGroup == "set-stage-name")
    return dawn::PassGroup::SetStageName;
  else if(passGroup == "StageReordering" || passGroup == "stage-reordering")
    return dawn::PassGroup::StageReordering;
  else if(passGroup == "StageMerger" || passGroup == "stage-merger")
    return dawn::PassGroup::StageMerger;
  else if(passGroup == "TemporaryMerger" || passGroup == "temporary-merger" ||
          passGroup == "tmp-merger")
    return dawn::PassGroup::TemporaryMerger;
  else if(passGroup == "Inlining" || passGroup == "inlining")
    return dawn::PassGroup::Inlining;
  else if(passGroup == "IntervalPartitioning" || passGroup == "interval-partitioning")
    return dawn::PassGroup::IntervalPartitioning;
  else if(passGroup == "TmpToStencilFunction" || passGroup == "tmp-to-stencil-function" ||
          passGroup == "tmp-to-stencil-fcn" || passGroup == "tmp-to-function" ||
          passGroup == "tmp-to-fcn")
    return dawn::PassGroup::TmpToStencilFunction;
  else if(passGroup == "SetNonTempCaches" || passGroup == "set-non-tmp-caches" ||
          passGroup == "set-nontmp-caches")
    return dawn::PassGroup::SetNonTempCaches;
  else if(passGroup == "SetCaches" || passGroup == "set-caches")
    return dawn::PassGroup::SetCaches;
  else if(passGroup == "SetBlockSize" || passGroup == "set-block-size")
    return dawn::PassGroup::SetBlockSize;
  else if(passGroup == "DataLocalityMetric" || passGroup == "data-locality-metric")
    return dawn::PassGroup::DataLocalityMetric;
  else if(passGroup == "MultiStageMerger" || passGroup == "multistage-merger")
    return dawn::PassGroup::MultiStageMerger;
  else
    throw std::runtime_error(std::string("Unknown pass group: ") + passGroup);
}

std::tuple<std::shared_ptr<dawn::SIR>, std::shared_ptr<dawn::iir::StencilInstantiation>,
           SerializationFormat>
deserializeInput(const std::string& input) {
  std::shared_ptr<dawn::SIR> stencilIR = nullptr;
  std::shared_ptr<dawn::iir::StencilInstantiation> internalIR = nullptr;

  SerializationFormat format = SerializationFormat::Byte;

  // Try SIR first
  IRType type = IRType::SIR;
  {
    try {
      stencilIR =
          dawn::SIRSerializer::deserializeFromString(input, dawn::SIRSerializer::Format::Byte);
      format = SerializationFormat::Byte;
    } catch(...) {
      stencilIR = nullptr;
    }
  }
  if(!stencilIR) {
    try {
      stencilIR =
          dawn::SIRSerializer::deserializeFromString(input, dawn::SIRSerializer::Format::Json);
      format = SerializationFormat::Json;
    } catch(...) {
      stencilIR = nullptr;
    }
  }
  // Then try IIR
  if(!stencilIR) {
    type = IRType::IIR;
    try {
      internalIR =
          dawn::IIRSerializer::deserializeFromString(input, dawn::IIRSerializer::Format::Byte);
      format = SerializationFormat::Byte;
    } catch(...) {
      internalIR = nullptr;
    }
  }
  if(!internalIR && !stencilIR) {
    try {
      internalIR =
          dawn::IIRSerializer::deserializeFromString(input, dawn::IIRSerializer::Format::Json);
      format = SerializationFormat::Json;
    } catch(...) {
      internalIR = nullptr;
    }
  }

  // Deserialize again, only this time do not catch exceptions
  switch(type) {
  case IRType::SIR: {
    if(format == SerializationFormat::Byte) {
      stencilIR =
          dawn::SIRSerializer::deserializeFromString(input, dawn::SIRSerializer::Format::Byte);
    } else {
      stencilIR =
          dawn::SIRSerializer::deserializeFromString(input, dawn::SIRSerializer::Format::Json);
    }
    break;
  }
  case IRType::IIR: {
    if(format == SerializationFormat::Byte) {
      internalIR =
          dawn::IIRSerializer::deserializeFromString(input, dawn::IIRSerializer::Format::Byte);
    } else {
      internalIR =
          dawn::IIRSerializer::deserializeFromString(input, dawn::IIRSerializer::Format::Json);
    }
    break;
  }
  }

  return {stencilIR, internalIR, format};
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("dawn-opt", "Optimizer for the Dawn DSL compiler toolchain");
  options.positional_help("[SIR or IIR file. If unset, reads from stdin]");

  // clang-format off
  options.add_options()
    ("input", "Input file. If unset, reads from stdin.", cxxopts::value<std::string>())
    ("o,out", "Output IIR filename. If unset, writes IIR to stdout.", cxxopts::value<std::string>())
    ("v,verbose", "Set verbosity level to info. If set, use -o or --out to redirect IIR.")
    ("default-opt", "Add default groups before those in --pass-groups.")
    ("p,pass-groups",
        "Comma-separated ordered list of pass groups to run. See dawn/Compiler/Driver.h for list. If unset and --default-opts is not passed, only lowers to IIR.",
        cxxopts::value<std::vector<std::string>>()->default_value({}))
    ("h,help", "Display usage.");

    options.add_options("Pass")
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  (OPTION, HELP, cxxopts::value<TYPE>()->default_value(toString(DEFAULT_VALUE)))
#include "dawn/Optimizer/Options.inc"
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

  // Determine the list of pass groups to run
  std::list<dawn::PassGroup> passGroups;
  if(result.count("default-opt") > 0) {
    passGroups = dawn::defaultPassGroups();
  }

  for(auto pg : result["pass-groups"].as<std::vector<std::string>>()) {
    passGroups.push_back(parsePassGroup(pg));
  }

  // Until stencil functions are added to the IIR...
  passGroups.push_back(dawn::PassGroup::Inlining);

  // Get the input from file or stdin
  std::string input;
  if(result.count("input")) {
    std::ifstream t(result["input"].as<std::string>());
    input.insert(input.begin(), (std::istreambuf_iterator<char>(t)),
                 std::istreambuf_iterator<char>());
  } else {
    std::istreambuf_iterator<char> begin(std::cin), end;
    input.insert(input.begin(), begin, end);
  }

  auto [stencilIR, internalIR, format] = deserializeInput(input);

  // Create a dawn::Options struct for the driver
  dawn::Options optimizerOptions;
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  optimizerOptions.NAME = result[OPTION].as<TYPE>();
#include "dawn/Optimizer/Options.inc"
#undef OPT

  // Call optimizer
  std::map<std::string, std::shared_ptr<dawn::iir::StencilInstantiation>> optimizedSIM;
  if(stencilIR) {
    optimizedSIM = dawn::run(stencilIR, passGroups, optimizerOptions);
  } else {
    std::map<std::string, std::shared_ptr<dawn::iir::StencilInstantiation>> stencilInstantiationMap{
        {"restoredIIR", internalIR}};
    optimizedSIM = dawn::run(stencilInstantiationMap, passGroups, optimizerOptions);
  }

  if(optimizedSIM.size() > 1) {
    DAWN_LOG(WARNING) << "More than one StencilInstantiation is not supported in IIR";
  }

  for(auto& [name, instantiation] : optimizedSIM) {
    dawn::IIRSerializer::Format iirFormat = (format == SerializationFormat::Byte)
                                                ? dawn::IIRSerializer::Format::Byte
                                                : dawn::IIRSerializer::Format::Json;
    if(result.count("out"))
      dawn::IIRSerializer::serialize(result["out"].as<std::string>(), instantiation, iirFormat);
    else if(!optimizerOptions.DumpStencilInstantiation) {
      std::cout << dawn::IIRSerializer::serializeToString(instantiation, iirFormat);
    } else {
      DAWN_LOG(INFO) << "dump-si present. Skipping serialization.";
    }
  }

  return 0;
}
