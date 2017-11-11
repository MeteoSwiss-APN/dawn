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

#include "gtclang/Driver/OptionsParser.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Compiler.h"
#include "dawn/Support/Config.h"
#include "dawn/Support/Format.h"
#include "gtclang/Support/Config.h"
#include "gtclang/Support/StringUtil.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace gtclang {

namespace {

/// @brief Print version and exit
DAWN_ATTRIBUTE_NORETURN static void versionPrinter() {
  llvm::outs() << dawn::format("gtclang (%s)\nbased on LLVM/Clang (%s), Dawn (%s)\n",
                               GTCLANG_FULL_VERSION_STR, LLVM_VERSION_STRING, DAWN_VERSION_STR);
  llvm::outs().flush();
  std::exit(0);
}

static std::string makeDefaultString(std::string defaultValue) {
  return (defaultValue.empty() ? "" : (" [default: " + defaultValue + "]"));
}

static std::string makeDefaultString(const char* defaultValue) {
  return makeDefaultString(std::string(defaultValue));
}

static std::string makeDefaultString(bool defaultValue) {
  return (defaultValue ? " [default: ON]" : " [default: OFF]");
}

static std::string makeDefaultString(int defaultValue) {
  return " [default: " + std::to_string(defaultValue) + "]";
}

template <bool HasDefaultValue>
struct DefaultString {
  template <class T>
  std::string operator()(const T&) {
    return std::string();
  }
};

template <>
struct DefaultString<true> {
  template <class T>
  std::string operator()(const T& defaultValue) {
    return makeDefaultString(defaultValue);
  }
};

/// @brief Print Help and exit
DAWN_ATTRIBUTE_NORETURN static void helpPrinter() {
  const int maxLineLen = 80;

  llvm::outs() << "OVERVIEW: gtclang - gridtools clang DSL compiler\n\n";
  llvm::outs() << "USAGE: gtclang [options] file -- [clang-options]\n\n";
  llvm::outs() << "OPTIONS:\n";
  llvm::outs() << splitString("Options not recognized by gtclang are automatically forwarded to "
                              "clang. Options after '--' are directly passed to clang. Options "
                              "starting with '-f' can be negated with '-fno-'.\n\n",
                              maxLineLen, 1);

  auto printOption = [&](std::string option, std::string help) -> void {
    const int maxOptionLen = 15;

    if(option.size() >= maxOptionLen)
      llvm::outs() << "  " << option << "\n"
                   << dawn::format("%s.\n", splitString(help, maxLineLen, maxOptionLen + 3, true));
    else
      llvm::outs() << dawn::format("  %-15s %s.\n", option,
                                   splitString(help, maxLineLen, maxOptionLen + 3, false));
  };

  printOption("-help, -h", "Display available options");
  printOption("-version", "Display version information");

#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  {                                                                                                \
    std::string optionname = std::string("-") + (F_GROUP ? "f" : "") + OPTION;                     \
    if(HAS_VALUE)                                                                                  \
      optionname += std::string("=") + VALUE_NAME;                                                 \
    std::string shortopt(OPTION_SHORT);                                                            \
    if(!shortopt.empty()) {                                                                        \
      optionname += ", -" + shortopt;                                                              \
      if(HAS_VALUE)                                                                                \
        optionname += VALUE_NAME;                                                                  \
    }                                                                                              \
    printOption(optionname, HELP + DefaultString < F_GROUP || HAS_VALUE > ()(DEFAULT_VALUE));      \
  }
#include "gtclang/Driver/Options.inc"

  llvm::outs() << "\nDAWN OPTIONS:\n";

#include "dawn/Compiler/Options.inc"
#undef OPT

  llvm::outs().flush();
  std::exit(0);
}

/// @brief Extract the value string and convert to the appropriate type
template <class T>
struct ExtractValue;

template <>
struct ExtractValue<int> {
  int operator()(const char* value) {
    DAWN_ASSERT(value);
    return std::atoi(value);
  }
};

template <>
struct ExtractValue<std::string> {
  std::string operator()(const char* value) {
    DAWN_ASSERT(value);
    return value;
  }
};

/// @brief Set the option to `true` or assign the extracted value. Returns return true on success
template <bool HasValue>
struct SetOption {
  template <class T>
  bool operator()(T& option, const char* value, bool negateOption) {
    (void)negateOption;
    if(!value)
      return false;
    option = ExtractValue<T>()(value);
    return true;
  }
};

template <>
struct SetOption<false> {
  template <class T>
  bool operator()(T& option, const char* value, bool negateOption) {
    (void)value;
    option = negateOption ? false : true;
    return true;
  }
};

} // anonymous namespace

OptionsParser::OptionsParser(Options* options) : options_(options) {

// Fill the OptionsMap
//  1.) Insert the long option (e.g "foo") into the options map
//  2.) Insert the mapping of short option to long option into the alias map
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  {                                                                                                \
    auto ret = optionsMap_.insert(                                                                 \
        OptionsMap::value_type{OPTION, [](Options* op, const char* value, bool negatedOption) {    \
                                 return SetOption<HAS_VALUE>()(op->NAME, value, negatedOption);    \
                               }});                                                                \
    DAWN_ASSERT_MSG(ret.second, "Option \"" OPTION "\" registered twice!");                        \
    std::string shortStr(OPTION_SHORT);                                                            \
    if(!shortStr.empty()) {                                                                        \
      auto it = optionsAliasMap_.find(OPTION_SHORT);                                               \
      if(it == optionsAliasMap_.end())                                                             \
        it = optionsAliasMap_                                                                      \
                 .insert(OptionsAliasMap::value_type(shortStr, std::vector<llvm::StringRef>{}))    \
                 .first;                                                                           \
      it->second.push_back(llvm::StringRef(ret.first->first));                                     \
    }                                                                                              \
  }
#include "dawn/Compiler/Options.inc"
#include "gtclang/Driver/Options.inc"
#undef OPT
}

bool OptionsParser::parse(const llvm::SmallVectorImpl<const char*>& args,
                          llvm::SmallVectorImpl<const char*>& clangArgs) {
  using namespace llvm;
  clangArgs.clear();

  for(std::size_t i = 0; i < args.size(); ++i) {
    StringRef arg = args[i];

    // Remaining arguments are passed to Clang
    if(StringRef(arg) == "--") {
      clangArgs.insert(clangArgs.end(), args.data() + i + 1, args.data() + args.size());
      break;
    }

    // We skip positional arguments and potential arguments of options
    if(!arg.startswith("-") && !arg.startswith("--")) {
      clangArgs.push_back(arg.data());
      continue;
    }

    // We don't allocate any memory here, we only operate on the strings of the options (from the
    // OptionsMap) and the current arg string
    std::vector<llvm::StringRef> options;
    llvm::StringRef value;

    // Check if this is a POSIX Style option:
    //
    // Case 1: Option of style "-o XXX"
    // Case 2: Option of style "-oXXX
    //
    bool Case1 = arg[0] == '-' && arg.size() == 2 && arg[1] != '-' && ((i + 1) != args.size()) &&
                 args[i + 1][0] != '-';
    bool Case2 = !Case1 && arg[0] == '-' && arg.size() > 2 && arg[1] != '-';

    bool skipNextOption = false;
    if(Case1 || Case2) {
      auto it = optionsAliasMap_.find(std::string(1, arg[1]));
      if(it != optionsAliasMap_.end()) {
        options = it->second;
        if(Case1) {
          value = args[i + 1];
          skipNextOption = true;
        } else {
          value = arg.substr(2);
        }
      }
    }

    // Handle normal options
    if(options.empty()) {
      // Extract value if any
      auto pair = arg.split('=');
      options.push_back(pair.first);
      value = pair.second;

      // Treat options with '--' and '-' equivalently i.e remove them
      while(options[0].front() == '-')
        options[0] = options[0].drop_front();
    }

    // The f-group options can be negated (check if we have fno-XXX or fXXX)
    std::vector<bool> isNegated(options.size(), false);
    for(std::size_t i = 0; i < options.size(); ++i) {
      if(options[i].startswith("fno-")) {
        isNegated[i] = true;
        options[i] = options[i].drop_front(4);
      } else if(options[i].startswith("f"))
        options[i] = options[i].drop_front();
    }

    // Handle special options
    if(options[0] == "version")
      versionPrinter();
    if(options[0] == "help" || (options.size() == 1 && options[0] == "h"))
      helpPrinter();

    // Handle the rest of GTClang options
    bool optionMatch = false;
    for(std::size_t i = 0; i < options.size(); ++i) {
      auto it = optionsMap_.find(options[i].str());
      if(it != optionsMap_.end()) {
        // Extract and convert the potential value
        if(!it->second(options_, value.size() == 0 ? nullptr : value.data(), isNegated[i])) {
          llvm::errs() << "error: expected argument for option '" << arg.str() << "'\n";
          return false;
        } else {
          optionMatch = true;
          break;
        }
      }
    }

    if(!optionMatch) {
      // Option not recognized, maybe it was meant for Clang?
      clangArgs.push_back(arg.data());
    } else if(skipNextOption)
      // Skip the next option as we already consumed it
      i += 1;
  }
  return true;
}

} // namespace gtclang
