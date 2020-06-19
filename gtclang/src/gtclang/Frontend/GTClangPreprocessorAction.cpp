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

#include "gtclang/Frontend/GTClangPreprocessorAction.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Format.h"
#include "gtclang/Frontend/GTClangContext.h"
#include "gtclang/Support/Logger.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/StringRef.h"

#include <iostream>
#include <unordered_set>
#include <vector>

namespace gtclang {

namespace {

template <char delimiter>
class WordDelimitedBy : public std::string {};

template <char delimiter>
std::istream& operator>>(std::istream& is, WordDelimitedBy<delimiter>& output) {
  std::getline(is, output, delimiter);
  return is;
}

template <typename Iter, typename Cont>
bool is_last(Iter iter, const Cont& cont) {
  return (iter != cont.end()) && (std::next(iter) == cont.end());
}

/// @brief Lex the source file and generate replacements for the enhanced gridtools clang DSL
class GTClangLexer {
  clang::CompilerInstance& compiler_;
  clang::Preprocessor& PP_;
  clang::DiagnosticsEngine& diag_;
  clang::SourceManager& SM_;

  bool done_;
  bool hasError_;

  /// Check if we found a `globals`
  bool hasGlobals_;

  /// Map of stencil and stencil functions to their parsed attributes from pragmas
  std::unordered_map<std::string, dawn::sir::Attr> attributeMap_;

  /// The current token
  clang::Token token_;

  /// Vector of replacements which need to be applied to the current source file to make it valid
  /// C++ code
  std::vector<clang::tooling::Replacement> replacements_;

  /// Replacements which need more context before they can be registered (currently only used to
  /// determine the return-type of Do methods)
  std::stack<std::pair<clang::SourceLocation, clang::SourceLocation>> replacementCandiates_;

private:
  enum StencilKind { SK_Invalid, SK_Stencil, SK_StencilFunction };

  static const char* toString(StencilKind stencilKind) {
    DAWN_ASSERT(stencilKind != SK_Invalid);
    return stencilKind == SK_Stencil ? "stencil" : "stencil_function";
  }

  /// @brief Register a `replacement` in the source range `[from, to]`
  void registerReplacement(clang::SourceLocation from, clang::SourceLocation to,
                           const std::string& replacement) {
    replacements_.push_back(clang::tooling::Replacement(
        SM_, clang::CharSourceRange::getTokenRange(from, to), replacement));
  }

  /// @brief Report an error
  void reportError(clang::SourceLocation loc, const std::string& msg) {
    if(!hasError_) {
      Diagnostics::reportRaw(diag_, loc, clang::DiagnosticIDs::Error, msg);
      hasError_ = true;
    }
  }

  /// @brief Check if an error has occured
  bool hasError() const { return diag_.hasErrorOccurred() || hasError_; }

  /// @brief Lex until we find a token in the main-file
  /// @returns `true` on success
  bool lexNext() {
    if(done_ || hasError())
      return false;

    do {
      PP_.Lex(token_);

      if(token_.is(clang::tok::eof) || hasError()) {
        done_ = true;
        return false;
      }

      if(SM_.getFileID(token_.getLocation()) == SM_.getMainFileID())
        return true;

    } while(true);
    return false;
  }

  /// @brief Consume `N` tokens from the token stream
  void consumeTokens(unsigned N) {
    while(N--)
      PP_.Lex(token_);
  }

  /// @brief Consume the specified `namespaces` by peeking ahead. `peekedTokens` will be incremented
  /// accordingly.
  /// @{
  void peekNamespaces(clang::ArrayRef<clang::StringRef> namespaceQualifiers,
                      unsigned& peekedTokens) {
    for(const auto& namespaceQualifier : namespaceQualifiers)
      if(PP_.LookAhead(peekedTokens).is(clang::tok::identifier) &&
         PP_.LookAhead(peekedTokens).getIdentifierInfo()->getName() == namespaceQualifier &&
         PP_.LookAhead(peekedTokens + 1).is(clang::tok::coloncolon))
        peekedTokens += 2;
  }

  void peekNamespace(clang::StringRef namespaceQualifier, unsigned& peekedTokens) {
    return peekNamespaces(llvm::ArrayRef<clang::StringRef>{namespaceQualifier}, peekedTokens);
  }
  /// @}

  bool peekUntilImpl(clang::tok::TokenKind tokenKind, unsigned& peekedTokens, bool accumulate,
                     std::string& accumulatedTokenStr) {
    using namespace clang;

    Token token;
    bool foundToken = false;

    while(true) {
      // Peek the next token ...
      token = PP_.LookAhead(peekedTokens);

      if(token.is(tok::eof))
        break;

      // ... is it our token?
      if(token.is(tokenKind)) {
        // Yes! Done.
        foundToken = true;
        break;
      }

      // No! Accumulte the token string
      if(accumulate)
        accumulatedTokenStr += PP_.getSpelling(token);
      peekedTokens++;
    }

    return foundToken;
  }

  /// @brief Peek until the `token` is found and accumulate all peeked tokens. `peekedTokens` will
  /// be incremented accordingly.
  /// @returns `true` if the token was found
  bool peekAndAccumulateUntil(clang::tok::TokenKind tokenKind, unsigned& peekedTokens,
                              std::string& accumulatedTokenStr) {
    return peekUntilImpl(tokenKind, peekedTokens, true, accumulatedTokenStr);
  }

  /// @brief Peek until the `token` is found and and incremented `peekedTokens` accordingly
  /// @returns `true` if the token was found
  bool peekUntil(clang::tok::TokenKind tokenKind, unsigned& peekedTokens) {
    std::string unusedStr;
    return peekUntilImpl(tokenKind, peekedTokens, false, unusedStr);
  }

  /// @brief Lex the argument list of a Do-Method
  void lexDoMethodArgList(StencilKind stencilKind, int numDoMethods, const std::string& name,
                          const clang::SourceLocation& loc) {
    using namespace clang;

    // We already lexed '(' of the argument list
    int parenNestingLevel = 1;

    while(lexNext()) {
      parenNestingLevel += token_.is(tok::l_paren);
      parenNestingLevel -= token_.is(tok::r_paren);
      if(parenNestingLevel == 0)
        break;

      // Replace `k_from` and `k_to`
      auto replaceIntervalKeyword = [&](const std::string& keyword) {
        if(token_.is(tok::identifier) && token_.getIdentifierInfo()->getName() == keyword)
          registerReplacement(token_.getLocation(), token_.getLocation(),
                              "interval" + std::to_string(numDoMethods) + " " + keyword);
      };

      replaceIntervalKeyword("k_from");
      replaceIntervalKeyword("k_to");
    }

    if(parenNestingLevel != 0) {
      reportError(
          loc, dawn::format(
                   "unbalanced parenthesis '%s' detected in argument-list of Do-Method of %s '%s'",
                   (parenNestingLevel > 0 ? ")" : "("), toString(stencilKind), name));
    } else {
      // Consume the last ')'
      consumeTokens(1);
    }
  }

  /// @brief Lex the body of a Do-Method
  void lexDoMethodBody(StencilKind stencilKind, const std::string& name,
                       const clang::SourceLocation& loc,
                       const std::unordered_set<std::string>& storages,
                       std::unordered_set<std::string>& storagesAllocatedOnTheFly) {
    using namespace clang;

    // We already lexed '{' of the body
    int curlyBracesNestingLevel = 1;

    while(lexNext()) {
      curlyBracesNestingLevel += token_.is(tok::l_brace);
      curlyBracesNestingLevel -= token_.is(tok::r_brace);
      if(curlyBracesNestingLevel == 0)
        break;

      // We found a `return`, if we parsed a `Do` before we can replace it with `double Do`
      if(token_.is(tok::kw_return) && !replacementCandiates_.empty()) {
        registerReplacement(replacementCandiates_.top().first, replacementCandiates_.top().second,
                            "double Do");
        replacementCandiates_.pop();
      }

      // Replace `STORAGE[...]` with `STORAGE(...)` where STORAGE is the name of a storage of the
      // stencil or stencil function
      if(token_.is(tok::identifier) && PP_.LookAhead(0).is(tok::l_square) &&
         (storages.count(token_.getIdentifierInfo()->getName().str()) ||
          storagesAllocatedOnTheFly.count(token_.getIdentifierInfo()->getName().str()))) {
        SourceLocation lSquareLoc = PP_.LookAhead(0).getLocation();
        unsigned peekedTokens = 1;

        // If we do not find a matching `]` there is def. something fishy.
        if(!peekUntil(tok::r_square, peekedTokens)) {
          reportError(
              lSquareLoc,
              dawn::format(
                  "unbalanced brace ']' detected in storage access '%s' in Do-Method of %s '%s'",
                  token_.getIdentifierInfo()->getName().str(), toString(stencilKind), name));
        } else {
          SourceLocation rSquareLoc = PP_.LookAhead(peekedTokens).getLocation();

          // Replace `[` and `]` with `(` and `)`
          registerReplacement(lSquareLoc, lSquareLoc, "(");
          registerReplacement(rSquareLoc, rSquareLoc, ")");
          consumeTokens(peekedTokens);
        }
      }

      // Replace `vertical_region(ARG_1, ARG_2)` with `for(auto k : {ARG_1, ARG_2})`
      if(token_.is(tok::identifier) && token_.getIdentifierInfo()->getName() == "vertical_region") {
        unsigned peekedTokens = 0;

        // Check for '('
        if(!PP_.LookAhead(peekedTokens++).is(tok::l_paren))
          continue;

        // Check for 'ARG_1' until ','
        std::string Arg1;
        if(!peekAndAccumulateUntil(tok::comma, peekedTokens, Arg1))
          continue;

        // Consume ','
        peekedTokens++;

        // Check for 'ARG_2' until ')'
        std::string Arg2;
        if(!peekAndAccumulateUntil(tok::r_paren, peekedTokens, Arg2))
          continue;

        registerReplacement(token_.getLocation(), PP_.LookAhead(peekedTokens).getLocation(),
                            dawn::format("for(auto __k_loopvar__ : {%s, %s})", Arg1, Arg2));

        consumeTokens(peekedTokens);
      }

      // Replace `iteration_space(ARG_1, ARG_2, ARG_3, ARG_4, ARG_5, ARG_6)` with `for(auto k :
      // {ARG_1, ARG_2, ARG_3, ARG_4, ARG_5, ARG_6})`
      if(token_.is(tok::identifier) && token_.getIdentifierInfo()->getName() == "iteration_space") {
        unsigned peekedTokens = 0;
        std::string intervalBounds;
        // Check for '('
        if(!PP_.LookAhead(peekedTokens++).is(tok::l_paren))
          continue;
        if(peekAndAccumulateUntil(tok::r_paren, peekedTokens, intervalBounds)) {
          // Split the comma separated string
          std::istringstream iss(intervalBounds);
          std::vector<std::string> curBounds{std::istream_iterator<WordDelimitedBy<','>>(iss),
                                             std::istream_iterator<WordDelimitedBy<','>>()};
          std::string replacement = "for(auto __k_indexrange__ : {";
          const std::array<char, 3> coordChar{'i', 'j', 'k'};

          auto boundIter = std::begin(curBounds);
          auto charIter = std::begin(coordChar);

          while(charIter != std::end(coordChar)) {
            bool forwardChar = false, forwardBound = false;
            if(boundIter != std::end(curBounds)) {
              const std::string boundStart = *boundIter;
              auto nextIter = std::next(boundIter);
              const std::string boundEnd = *nextIter;
              const std::string errMsg = std::string("failed parsing iteration_space argument: ") +
                                         "(" + boundStart + "," + boundEnd + ").";
              if(!std::any_of(charIter, std::end(coordChar),
                              [boundStart](const char& other) { return boundStart[0] == other; })) {
                // If boundStart is not any coordChar
                reportError(token_.getLocation(),
                            errMsg + " Iteration space argument does not begin with {i, j, k}");
              } else if(boundStart[0] != boundEnd[0]) {
                // Is *charIter, but does not match.
                // Example: boundStart = i_start, boundEnd = j_end
                reportError(token_.getLocation(), errMsg + " Dimensions do not match");
              } else if(boundStart[0] != *charIter) {
                // Add default
                replacement +=
                    std::string(1, *charIter) + "_start, " + std::string(1, *charIter) + "_end";
                forwardChar = true;
              } else {
                replacement += boundStart + ", " + boundEnd;
                forwardBound = true;
                forwardChar = true;
              }
            } else {
              // Add default
              replacement +=
                  std::string(1, *charIter) + "_start, " + std::string(1, *charIter) + "_end";
              forwardChar = true;
            }
            if(!is_last(charIter, coordChar))
              replacement += ", ";
            if(forwardChar)
              ++charIter;
            if(forwardBound)
              std::advance(boundIter, 2);
          }
          replacement += "})";
          registerReplacement(token_.getLocation(), PP_.LookAhead(peekedTokens).getLocation(),
                              replacement);
        }
        consumeTokens(peekedTokens);
      }

      // Check for var ARG1 [ = RHS];
      if(token_.is(tok::identifier) && token_.getIdentifierInfo()->getName() == "var") {

        unsigned peekedTokens = 0;
        bool peekSuccess = false;
        // If we are in a stencil_function, var must be double
        if(stencilKind == SK_StencilFunction) {
          registerReplacement(token_.getLocation(), token_.getLocation(), "double");
          peekSuccess = peekUntil(tok::semi, peekedTokens);
        }
        // If we are in a stencil, we chose (temporary) storage as the default type and let the
        // passes promote them to local variables if needed
        // If we find [= RHS], we replace the statement with ARG1 = RHS
        // otherwise we remove the statement alltogether
        else {
          // Accumulate all identifiers up to `;`
          std::string storagesStr;
          peekSuccess = peekAndAccumulateUntil(tok::semi, peekedTokens, storagesStr);
          if(peekSuccess) {
            // Check for ill-defined statements
            if(storagesStr.find(",") != std::string::npos) {
              reportError(token_.getLocation(),
                          "only single declarations are currently supported: expected  ; got ,");
              return;
            }
            if((storagesStr.find("[") != std::string::npos) ||
               (storagesStr.find("]") != std::string::npos)) {
              reportError(token_.getLocation(),
                          "initialisation with offsets is not supported, used in:" + storagesStr);
            }

            // Split the string if we find an assignment
            if(storagesStr.find("=") != std::string::npos) {
              llvm::SmallVector<StringRef, 2> accumulatedDeclaration;
              StringRef(storagesStr).split(accumulatedDeclaration, '=');
              registerReplacement(token_.getLocation(), PP_.LookAhead(peekedTokens).getLocation(),
                                  accumulatedDeclaration[0].str() + " = " +
                                      accumulatedDeclaration[1].str() + ";");
              storagesAllocatedOnTheFly.emplace(accumulatedDeclaration[0].str());
            } else {
              storagesAllocatedOnTheFly.emplace(storagesStr);
              registerReplacement(token_.getLocation(), PP_.LookAhead(peekedTokens).getLocation(),
                                  "");
            }
            consumeTokens(peekedTokens);
          }
        }
      }
    }

    // We haven't found a `return` and if we parsed a `Do` before we can replace if with `void
    // Do`
    if(!replacementCandiates_.empty()) {
      registerReplacement(replacementCandiates_.top().first, replacementCandiates_.top().second,
                          "void Do");
      replacementCandiates_.pop();
    }

    if(curlyBracesNestingLevel != 0) {
      reportError(loc, dawn::format("unbalanced brace '%s' detected in Do-Method of %s '%s'",
                                    (curlyBracesNestingLevel > 0 ? "}" : "{"),
                                    toString(stencilKind), name));
    }
  }

  /// @brief Lex a Do-Method
  void lexStencilBody(StencilKind stencilKind, const std::string& name,
                      const clang::SourceLocation& loc) {
    using namespace clang;

    // We already lexed '{' of the struct
    int curlyBracesNestingLevel = 1;
    int numDoMethods = 0;

    std::unordered_set<std::string> storages;

    std::unordered_set<std::string> storagesAllocatedOnTheFly;

    // Wrapping the boundary conditions in a function call to have proper syntax requires storing
    // all the arguments found
    std::unordered_set<std::string> boundaryConditions;
    // and the location where we want to generate the function
    SourceLocation bcLocationStart;
    SourceLocation bcLocationEnd;

    bool bcLocationSet = false;

    while(lexNext()) {

      // Update brace counter
      if(token_.is(tok::l_brace) || token_.is(tok::r_brace)) {
        curlyBracesNestingLevel += token_.is(tok::l_brace);
        curlyBracesNestingLevel -= token_.is(tok::r_brace);
        if(curlyBracesNestingLevel == 0)
          break;
        continue;
      }

      // Lex `storage IDENTIFIER;` or `storage IDENTIFIER_1, ... IDENTIFIER_N;`
      if(token_.is(tok::identifier)) {
        unsigned peekedTokens = 0;

        // Consume gtclang::dsl:: of the storage if necessary
        peekNamespaces({"gridtools", "clang"}, peekedTokens);

        // Get the token which describes the `storage`
        const Token& curToken = peekedTokens == 0 ? token_ : PP_.LookAhead(peekedTokens++);

        if(curToken.is(tok::identifier) &&
           ((curToken.getIdentifierInfo()->getName().find("storage") != std::string::npos) ||
            curToken.getIdentifierInfo()->getName() == "var")) {

          if(stencilKind == SK_StencilFunction &&
             curToken.getIdentifierInfo()->getName() == "var") {
            reportError(token_.getLocation(), "declaration of temporary variable 'var' in stencil "
                                              "function " +
                                                  name + " is not allowed");
            break;
          }

          // Accumulate all identifiers up to `;`
          std::string storagesStr;
          if(peekAndAccumulateUntil(tok::semi, peekedTokens, storagesStr)) {

            // Split the comma separated string
            llvm::SmallVector<StringRef, 5> curStorages;
            StringRef(storagesStr).split(curStorages, ',');
            for(const auto& storage : curStorages)
              storages.emplace(storage.str());

            consumeTokens(peekedTokens);
          }
        }
        if(token_.is(tok::identifier) &&
           token_.getIdentifierInfo()->getName() == "boundary_condition" &&
           PP_.LookAhead(0).is(tok::l_paren)) {
          peekedTokens = 1;

          // Accumulate all identifiers up to `;`
          std::string storagesStr;
          if(peekAndAccumulateUntil(tok::semi, peekedTokens, storagesStr)) {
            boundaryConditions.emplace("boundary_condition(" + storagesStr + ";");
            // this is the first boundary_condition we encounter, store it's location
            if(!bcLocationSet) {
              bcLocationStart = token_.getLocation();
              bcLocationEnd = PP_.LookAhead(peekedTokens).getLocation();
              bcLocationSet = true;
            } else {
              // after saving the information, we clear its content
              registerReplacement(token_.getLocation(), PP_.LookAhead(peekedTokens).getLocation(),
                                  "");
            }
            consumeTokens(peekedTokens);
          }
        }
      }

      //
      // Lex argument list of Do-Method
      //
      bool DoWasLexed = false;
      SourceLocation DoMethodLoc;
      SourceLocation beforeFristDoMethod;

      // `void Do(`, `double Do(`
      if((token_.is(tok::kw_void) || token_.is(tok::kw_double) || token_.is(tok::kw_float)) &&
         PP_.LookAhead(0).is(tok::identifier) &&
         PP_.LookAhead(0).getIdentifierInfo()->getName() == "Do" &&
         PP_.LookAhead(1).is(tok::l_paren)) {

        DoMethodLoc = PP_.LookAhead(0).getLocation();
        beforeFristDoMethod = token_.getLocation();
        consumeTokens(2);
        DoWasLexed = true;
        lexDoMethodArgList(stencilKind, numDoMethods++, name, DoMethodLoc);
      }

      // `Do(`
      if(token_.is(tok::identifier) && token_.getIdentifierInfo()->getName() == "Do" &&
         PP_.LookAhead(0).is(tok::l_paren)) {

        // Replace with either `double Do` or `void Do`, this will be determined after we lexed
        // the Do-Method body and know if there was a return statement.
        replacementCandiates_.push(std::make_pair(token_.getLocation(), token_.getLocation()));

        DoMethodLoc = token_.getLocation();
        beforeFristDoMethod = token_.getLocation();
        consumeTokens(1);
        DoWasLexed = true;
        lexDoMethodArgList(stencilKind, numDoMethods++, name, DoMethodLoc);
      }

      //
      // Lex body of Do-Method
      //

      // ')' is consumed by `lexDoMethodArgList`, thus just check for the next '{'
      if(DoWasLexed && token_.is(tok::l_brace)) {
        lexDoMethodBody(stencilKind, name, DoMethodLoc, storages, storagesAllocatedOnTheFly);
        curlyBracesNestingLevel++;
      } else {
        // `Do {`
        if(token_.is(tok::identifier) && token_.getIdentifierInfo()->getName() == "Do" &&
           PP_.LookAhead(0).is(tok::l_brace)) {
          DoMethodLoc = token_.getLocation();

          // Replace with either `double Do` or `void Do`, this will be determined after we lexed
          // the Do-Method body and know if there was a return statement.
          replacementCandiates_.push(std::make_pair(token_.getLocation(), token_.getLocation()));

          // Consume '{'
          consumeTokens(1);
          curlyBracesNestingLevel++;

          // Replace the '{' with '(){' to fix the missing argument-list of the Do-Method
          registerReplacement(token_.getLocation(), token_.getLocation(), "() {");

          lexDoMethodBody(stencilKind, name, DoMethodLoc, storages, storagesAllocatedOnTheFly);
        }
      }

      if(DoWasLexed) {
        // Add the found temporaries into the storage part of the Do-Method
        if(!storagesAllocatedOnTheFly.empty()) {
          registerReplacement(beforeFristDoMethod, beforeFristDoMethod, "void ");
          std::string fulltemps = "";
          for(const auto& varName : storagesAllocatedOnTheFly) {
            fulltemps += "var " + varName + ";\n";
          }
          fulltemps += "void";
          registerReplacement(beforeFristDoMethod, beforeFristDoMethod, fulltemps);
          storagesAllocatedOnTheFly.clear();
        }
      }

      // Update brace counter
      if(token_.is(tok::l_brace) || token_.is(tok::r_brace)) {
        curlyBracesNestingLevel += token_.is(tok::l_brace);
        curlyBracesNestingLevel -= token_.is(tok::r_brace);
        if(curlyBracesNestingLevel == 0)
          break;
        continue;
      }
    }

    if(bcLocationSet) {
      std::string fullstring;
      fullstring = "void __boundary_condition__generated__() {\n";
      for(const auto& boundaryConditon : boundaryConditions) {
        fullstring += boundaryConditon + "\n";
      }
      fullstring += "}";
      registerReplacement(bcLocationStart, bcLocationEnd, fullstring);

      bcLocationSet = false;
      boundaryConditions.clear();
    }

    if(curlyBracesNestingLevel != 0) {
      reportError(loc, dawn::format("unbalanced brace '%s' detected in %s '%s'",
                                    (curlyBracesNestingLevel > 0 ? "}" : "{"),
                                    toString(stencilKind), name));
    }
  }

  /// @brief Try to lex stencil (or stencil function)
  void tryLexStencils() {
    using namespace clang;

    while(lexNext()) {

      // Check for the pattern `struct IDENTIFIER : public [gtclang::dsl::]stencil {`
      if(token_.is(tok::kw_struct) || token_.is(tok::kw_class)) {
        unsigned peekedTokens = 0;

        // Check for `Identifier`
        if(!PP_.LookAhead(peekedTokens).is(tok::identifier))
          continue;
        const Token& identifierToken = PP_.LookAhead(peekedTokens);
        peekedTokens++;

        // Check for `:`
        if(!PP_.LookAhead(peekedTokens++).is(tok::colon))
          continue;

        // Check for `public`
        if(!PP_.LookAhead(peekedTokens++).is(tok::kw_public))
          continue;

        // Consume namespace `gridtools` and namespace `clang`
        peekNamespaces({"gridtools", "clang"}, peekedTokens);

        // Check for `stencil` or `stencil_function`
        const Token& tokenStencilKind = PP_.LookAhead(peekedTokens++);
        StencilKind stencilKind = SK_Invalid;
        if(tokenStencilKind.is(tok::identifier)) {

          // Don't change order
          if(tokenStencilKind.getIdentifierInfo()->getName().find("stencil_function") !=
             StringRef::npos)
            stencilKind = SK_StencilFunction;
          else if(tokenStencilKind.getIdentifierInfo()->getName().find("stencil") !=
                  StringRef::npos)
            stencilKind = SK_Stencil;
        }
        if(stencilKind == SK_Invalid)
          continue;

        // Check for `{`
        if(!PP_.LookAhead(peekedTokens++).is(tok::l_brace))
          continue;

        consumeTokens(peekedTokens);
        lexStencilBody(stencilKind, identifierToken.getIdentifierInfo()->getName().str(),
                       identifierToken.getLocation());
      }

      // Check for `stencil IDENTIFIER {` or `stencil_function IDENTIFIER {` or `globals {`
      if(token_.is(tok::identifier)) {
        IdentifierInfo* identifierInfo = token_.getIdentifierInfo();
        if(!identifierInfo)
          continue;

        unsigned peekedTokens = 0;

        StencilKind stencilKind = SK_Invalid;
        if(identifierInfo->getName() == "stencil")
          stencilKind = SK_Stencil;
        else if(identifierInfo->getName() == "stencil_function")
          stencilKind = SK_StencilFunction;
        else if(identifierInfo->getName() == "globals") {

          // Check for `{`
          const Token& tokenLBrace = PP_.LookAhead(peekedTokens++);
          if(!tokenLBrace.is(tok::l_brace))
            continue;

          hasGlobals_ = true;

          // Create replacement `globals {`
          registerReplacement(token_.getLocation(), tokenLBrace.getLocation(), "struct globals {");
          consumeTokens(peekedTokens);
        }

        if(stencilKind != SK_Invalid) {
          // Check for `Identifier`
          const Token& tokenIdentifier = PP_.LookAhead(peekedTokens++);
          std::string identifier;
          if(tokenIdentifier.is(tok::identifier) && tokenIdentifier.getIdentifierInfo())
            identifier = tokenIdentifier.getIdentifierInfo()->getName();
          else
            continue;

          // Check for `{`
          const Token& tokenLBrace = PP_.LookAhead(peekedTokens++);
          if(!tokenLBrace.is(tok::l_brace))
            continue;

          // Create replacement for `stencil IDENTIFIER {` or `stencil_function IDENTIFIER {`
          const char* stencilKindStr = toString(stencilKind);
          registerReplacement(
              token_.getLocation(), tokenLBrace.getLocation(),
              dawn::format("struct %s : public gtclang::dsl::%s%s{ using gtclang::dsl::%s::%s;",
                           identifier, stencilKindStr, hasGlobals_ ? ", public globals " : " ",
                           stencilKindStr, stencilKindStr));

          consumeTokens(peekedTokens);
          lexStencilBody(stencilKind, identifier, tokenIdentifier.getLocation());
        }
      }
    }
  }

  /// @brief Dump the token for debugging
  /// @{
  void dumpToken() { dumpToken(token_); }
  void dumpToken(const clang::Token& token) {
    PP_.DumpToken(token);
    llvm::errs() << "\n";
  }
  /// @}

  /// @brief Performs a raw lexing of the source file and lexes
  ///
  /// `#pragma gtclang CLAUSE_1 [, ... CLAUSE_N]`
  ///
  /// Note that there might be cleaner ways of doing this but it is not clear if custom pragmas
  /// can be parsed in a proper way without hacking Clang. This is fairly efficient though.
  void tryLexPragmas() {
    using namespace clang;

    const llvm::MemoryBuffer* fromFile = SM_.getBuffer(SM_.getMainFileID());
    clang::Lexer rawLexer(SM_.getMainFileID(), fromFile, SM_, PP_.getLangOpts());

    // Raw lexing does not support lookAheads which makes the lexing slightly cumbersome. We use
    // state based approach here.
    enum LexStateKind {
      Unknown,                 // State is unknown
      LexedHash,               // Previously lexed a `#`
      LexedPragma,             // Previously lexed a `pragma
      LexedGTClang,            // Previously lexed a `gtclang`
      LexedValidGTClangPragma, // We lexed a valid `#pragma gtclang CLAUSE`, we now
                               // look for the next stencil or stencil_function it may apply
    };

    LexStateKind state = Unknown;
    dawn::sir::Attr curAttribute;
    SourceLocation curStartLoc;
    SourceLocation curEndLoc;
    std::string curClause;

    rawLexer.LexFromRawLexer(token_);
    while(token_.isNot(tok::eof)) {
      if(SM_.getFileID(token_.getLocation()) != SM_.getMainFileID())
        continue;

      switch(state) {
      case Unknown:
        // Check for `#`
        if(token_.is(tok::hash)) {
          state = LexedHash;
          curStartLoc = token_.getLocation();
          curAttribute.clear();
          curClause.clear();
        }
        break;
      case LexedHash:
        // Check for `pragma`
        if(token_.is(tok::raw_identifier) && token_.getRawIdentifier() == "pragma")
          state = LexedPragma;
        else
          state = Unknown;
        break;
      case LexedPragma:
        // Check for `gtclang`
        if(token_.is(tok::raw_identifier) && token_.getRawIdentifier() == "gtclang")
          state = LexedGTClang;
        else
          state = Unknown;
        break;
      case LexedGTClang:
        // Check for `CLAUSE`
        if(token_.is(tok::raw_identifier)) {
          curClause = token_.getRawIdentifier().str();

          if(curClause == "no_codegen")
            curAttribute.set(dawn::sir::Attr::Kind::NoCodeGen);
          else if(curClause == "merge_stages")
            curAttribute.set(dawn::sir::Attr::Kind::MergeStages);
          else if(curClause == "merge_do_methods")
            curAttribute.set(dawn::sir::Attr::Kind::MergeDoMethods);
          else if(curClause == "merge_temporaries")
            curAttribute.set(dawn::sir::Attr::Kind::MergeTemporaries);
          else if(curClause == "use_kcaches")
            curAttribute.set(dawn::sir::Attr::Kind::UseKCaches);
          else {
            // We don't know this pragma, issue a warning about unknown gtclang pragma
            Diagnostics::reportRaw(
                diag_, curStartLoc, clang::DiagnosticIDs::Warning,
                dawn::format("invalid clause '%s' for '#pragma gtclang'", curClause));
            state = Unknown;
            break;
          }

          curEndLoc = token_.getLocation();
          state = LexedValidGTClangPragma;

        } else
          state = Unknown;
        break;
      case LexedValidGTClangPragma:

        // The pragma is followed by a ',' we have to parse more clauses!
        if(token_.is(tok::comma)) {
          state = LexedGTClang;
          break;
        }

        // We need to know the stencil or stencil_function this pragma applies to, we thus check
        // for
        //   `stencil IDENTIFIER`
        //   `stencil_function IDENTIFIER`
        //   `struct IDENTIFIER`
        //   `class IDENTIFIER`
        if((token_.is(tok::raw_identifier) && (token_.getRawIdentifier() == "stencil" ||
                                               token_.getRawIdentifier() == "stencil_function")) ||
           token_.is(tok::kw_class) || token_.is(tok::kw_struct)) {

          // Lex next token to get the `IDENTIFIER`
          rawLexer.LexFromRawLexer(token_);
          if(token_.is(tok::eof))
            continue;

          if(token_.is(tok::raw_identifier)) {
            std::string curIdentifer = token_.getRawIdentifier().str();

            // Register the attribute and remove the pragma
            attributeMap_.emplace(curIdentifer, curAttribute);
            registerReplacement(curStartLoc, curEndLoc, "");
          }
        } else {
          Diagnostics::reportRaw(
              diag_, token_.getLocation(), clang::DiagnosticIDs::Error,
              dawn::format("statement after '#pragma gtclang %s' must be a stencil declaration",
                           curClause));
        }

        // We finished or failed to lex the pragma, start over again
        state = Unknown;
      };

      rawLexer.LexFromRawLexer(token_);
    }
  }

public:
  GTClangLexer(clang::CompilerInstance& compiler)
      : compiler_(compiler), PP_(compiler_.getPreprocessor()), diag_(compiler_.getDiagnostics()),
        SM_(compiler_.getSourceManager()), done_(false), hasError_(false), hasGlobals_(false) {}

  /// @brief Compute the vector of replacements
  void computeReplacements() {
    done_ = false;
    attributeMap_.clear();
    hasGlobals_ = false;
    tryLexPragmas();
    tryLexStencils();
  }

  /// @brief Get the attribute map for the stencil and stencil functions
  const std::unordered_map<std::string, dawn::sir::Attr>& getAttributeMap() const {
    return attributeMap_;
  }

  /// @brief Get the replacements
  const std::vector<clang::tooling::Replacement>& getReplacements() const { return replacements_; }
};

} // namespace

GTClangPreprocessorAction::GTClangPreprocessorAction(GTClangContext* context) : context_(context) {}

void GTClangPreprocessorAction::ExecuteAction() {
  using namespace clang;

  DAWN_LOG(INFO) << "Start preprocessing ...";

  CompilerInstance& compiler = getCompilerInstance();
  SourceManager& SM = compiler.getSourceManager();
  Preprocessor& PP = compiler.getPreprocessor();

  PP.EnterMainSourceFile();
  compiler.getDiagnosticClient().BeginSourceFile(compiler.getLangOpts(), &PP);

  GTClangLexer lexer(compiler);
  lexer.computeReplacements();

  compiler.getDiagnosticClient().EndSourceFile();

  if(compiler.getDiagnostics().hasErrorOccurred())
    return;

  bool ReportPassPreprocessor = context_->getOptions().ReportPassPreprocessor;
  bool DumpPP = context_->getOptions().DumpPP;

  std::string PPCode;
  if(!lexer.getReplacements().empty()) {
    // Apply the replacements
    clang::Rewriter rewriter(SM, compiler.getLangOpts());
    for(const tooling::Replacement& replacement : lexer.getReplacements())
      replacement.apply(rewriter);

    // Rewrite the code
    llvm::raw_string_ostream os(PPCode);
    rewriter.getEditBuffer(SM.getMainFileID()).write(os);
    os.flush();

    // Replace the code of the main-file
    SM.overrideFileContents(SM.getFileEntryForID(SM.getMainFileID()),
                            llvm::MemoryBuffer::getMemBufferCopy(PPCode.data()));

    // Set the attributes
    for(const auto& nameAttrPair : lexer.getAttributeMap())
      context_->setStencilAttribute(nameAttrPair.first, nameAttrPair.second);

  } else if(DumpPP || ReportPassPreprocessor) {
    PPCode = SM.getBufferData(SM.getMainFileID()).str();
  }

  if(DumpPP)
    std::cout << PPCode << std::endl;

  if(ReportPassPreprocessor) {
    StringRef PPCodeRef(PPCode);

    llvm::SmallVector<clang::StringRef, 100> PPCodeLines;
    PPCodeRef.split(PPCodeLines, '\n');
    for(int i = 0; i < PPCodeLines.size(); ++i) {
      StringRef line = PPCodeLines[i];

      // Drop indentation
      line = line.substr(line.find_first_not_of(' '));

      // Drop comments
      line = line.substr(0, line.find("//"));
      line = line.substr(0, line.find("/*"));

      // Drop trailing whitespaces
      line = line.rtrim();

      // Line numbering starts at 1
      std::cout << (i + 1) << ": " << line.str() << "\n\n";
    }
  }

  DAWN_LOG(INFO) << "Done preprocessing";
}

} // namespace gtclang
