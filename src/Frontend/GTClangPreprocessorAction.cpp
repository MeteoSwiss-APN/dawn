//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _     _ _              _            _
//                        (_)   | | |            | |          | |
//               __ _ _ __ _  __| | |_ ___   ___ | |___    ___| | __ _ _ __   __ _
//              / _` | '__| |/ _` | __/ _ \ / _ \| / __|  / __| |/ _` | '_ \ / _` |
//             | (_| | |  | | (_| | || (_) | (_) | \__ \ | (__| | (_| | | | | (_| |
//              \__, |_|  |_|\__,_|\__\___/ \___/|_|___/  \___|_|\__,_|_| |_|\__, |
//               __/ |                                                        __/ |
//              |___/                                                        |___/
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gtclang/Frontend/GTClangPreprocessorAction.h"
#include "gsl/Support/Assert.h"
#include "gsl/Support/Format.h"
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
  std::unordered_map<std::string, gsl::sir::Attr> attributeMap_;

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
    GSL_ASSERT(stencilKind != SK_Invalid);
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
          loc, gsl::format(
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
                       const std::unordered_set<std::string>& storages) {
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
         storages.count(token_.getIdentifierInfo()->getName().str())) {
        SourceLocation lSquareLoc = PP_.LookAhead(0).getLocation();
        unsigned peekedTokens = 1;

        // If we do not find a matching `]` there is def. something fishy.
        if(!peekUntil(tok::r_square, peekedTokens)) {
          reportError(
              lSquareLoc,
              gsl::format(
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
                            gsl::format("for(auto __k_loopvar__ : {%s, %s})", Arg1, Arg2));

        consumeTokens(peekedTokens);
      }
    }

    // We haven't found a `return` and if we parsed a `Do` before we can replace if with `void Do`
    if(!replacementCandiates_.empty()) {
      registerReplacement(replacementCandiates_.top().first, replacementCandiates_.top().second,
                          "void Do");
      replacementCandiates_.pop();
    }

    if(curlyBracesNestingLevel != 0) {
      reportError(loc, gsl::format("unbalanced brace '%s' detected in Do-Method of %s '%s'",
                                   (curlyBracesNestingLevel > 0 ? "}" : "{"), toString(stencilKind),
                                   name));
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

        // Consume gridtools::clang:: of the storage if necessary
        peekNamespaces({"gridtools", "clang"}, peekedTokens);

        // Get the token which describes the `storage`
        const Token& curToken = peekedTokens == 0 ? token_ : PP_.LookAhead(peekedTokens++);

        if(curToken.is(tok::identifier) &&
           (curToken.getIdentifierInfo()->getName() == "storage" ||
            curToken.getIdentifierInfo()->getName() == "temporary_storage")) {

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
      }

      //
      // Lex argument list of Do-Method
      //
      bool DoWasLexed = false;
      SourceLocation DoMethodLoc;

      // `void Do(`, `double Do(`
      if((token_.is(tok::kw_void) || token_.is(tok::kw_double) || token_.is(tok::kw_float)) &&
         PP_.LookAhead(0).is(tok::identifier) &&
         PP_.LookAhead(0).getIdentifierInfo()->getName() == "Do" &&
         PP_.LookAhead(1).is(tok::l_paren)) {

        DoMethodLoc = PP_.LookAhead(0).getLocation();
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
        consumeTokens(1);
        DoWasLexed = true;
        lexDoMethodArgList(stencilKind, numDoMethods++, name, DoMethodLoc);
      }

      //
      // Lex body of Do-Method
      //

      // ')' is consumed by `lexDoMethodArgList`, thus just check for the next '{'
      if(DoWasLexed && token_.is(tok::l_brace)) {
        lexDoMethodBody(stencilKind, name, DoMethodLoc, storages);
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

          lexDoMethodBody(stencilKind, name, DoMethodLoc, storages);
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

    if(curlyBracesNestingLevel != 0) {
      reportError(loc, gsl::format("unbalanced brace '%s' detected in %s '%s'",
                                   (curlyBracesNestingLevel > 0 ? "}" : "{"), toString(stencilKind),
                                   name));
    }
  }

  /// @brief Try to lex stencil (or stencil function)
  void tryLexStencils() {
    using namespace clang;

    while(lexNext()) {

      // Check for the pattern `struct IDENTIFIER : public [gridtools::clang::]stencil {`
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
          registerReplacement(token_.getLocation(), tokenLBrace.getLocation(),
                              "struct globals : public gridtools::clang::globals_impl<globals> {");
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
              gsl::format(
                  "struct %s : public gridtools::clang::%s%s{ using gridtools::clang::%s::%s;",
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
  /// Note that there might be cleaner ways of doing this but it is not clear if custom pragmas can
  /// be parsed in a proper way without hacking Clang. This is fairly efficient though.
  void tryLexPragmas() {
    using namespace clang;

    const llvm::MemoryBuffer* fromFile = SM_.getBuffer(SM_.getMainFileID());
    clang::Lexer rawLexer(SM_.getMainFileID(), fromFile, SM_, PP_.getLangOpts());

    // Raw lexing does not support lookAheads which makes the lexing slightly cumbersome. We use
    // state based approach here.
    enum LexStateKind {
      LK_Unknown,                 // State is unknown
      LK_LexedHash,               // Previously lexed a `#`
      LK_LexedPragma,             // Previously lexed a `pragma
      LK_LexedGTClang,            // Previously lexed a `gtclang`
      LK_LexedValidGTClangPragma, // We lexed a valid `#pragma gtclang CLAUSE`, we now
                                  // look for the next stencil or stencil_function it may apply
    };

    LexStateKind state = LK_Unknown;
    gsl::sir::Attr curAttribute;
    SourceLocation curStartLoc;
    SourceLocation curEndLoc;
    std::string curClause;

    rawLexer.LexFromRawLexer(token_);
    while(token_.isNot(tok::eof)) {
      if(SM_.getFileID(token_.getLocation()) != SM_.getMainFileID())
        continue;

      switch(state) {
      case LK_Unknown:
        // Check for `#`
        if(token_.is(tok::hash)) {
          state = LK_LexedHash;
          curStartLoc = token_.getLocation();
          curAttribute.clear();
          curClause.clear();
        }
        break;
      case LK_LexedHash:
        // Check for `pragma`
        if(token_.is(tok::raw_identifier) && token_.getRawIdentifier() == "pragma")
          state = LK_LexedPragma;
        else
          state = LK_Unknown;
        break;
      case LK_LexedPragma:
        // Check for `gtclang`
        if(token_.is(tok::raw_identifier) && token_.getRawIdentifier() == "gtclang")
          state = LK_LexedGTClang;
        else
          state = LK_Unknown;
        break;
      case LK_LexedGTClang:
        // Check for `CLAUSE`
        if(token_.is(tok::raw_identifier)) {
          curClause = token_.getRawIdentifier().str();

          if(curClause == "no_codegen")
            curAttribute.set(gsl::sir::Attr::AK_NoCodeGen);
          else if(curClause == "merge_stages")
            curAttribute.set(gsl::sir::Attr::AK_MergeStages);
          else if(curClause == "merge_do_methods")
            curAttribute.set(gsl::sir::Attr::AK_MergeDoMethods);
          else if(curClause == "merge_temporaries")
            curAttribute.set(gsl::sir::Attr::AK_MergeTemporaries);
          else if(curClause == "use_kcaches")
            curAttribute.set(gsl::sir::Attr::AK_UseKCaches);
          else {
            // We don't know this pragma, issue a warning about unknown gtclang pragma
            Diagnostics::reportRaw(
                diag_, curStartLoc, clang::DiagnosticIDs::Warning,
                gsl::format("invalid clause '%s' for '#pragma gtclang'", curClause));
            state = LK_Unknown;
            break;
          }

          curEndLoc = token_.getLocation();
          state = LK_LexedValidGTClangPragma;

        } else
          state = LK_Unknown;
        break;
      case LK_LexedValidGTClangPragma:

        // The pragma is followed by a ',' we have to parse more clauses!
        if(token_.is(tok::comma)) {
          state = LK_LexedGTClang;
          break;
        }

        // We need to know the stencil or stencil_function this pragma applies to, we thus check for
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
              gsl::format("statement after '#pragma gtclang %s' must be a stencil declaration",
                          curClause));
        }

        // We finished or failed to lex the pragma, start over again
        state = LK_Unknown;
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
  const std::unordered_map<std::string, gsl::sir::Attr>& getAttributeMap() const {
    return attributeMap_;
  }

  /// @brief Get the replacements
  const std::vector<clang::tooling::Replacement>& getReplacements() const { return replacements_; }
};

} // anonymous namespace

GTClangPreprocessorAction::GTClangPreprocessorAction(GTClangContext* context) : context_(context) {}

void GTClangPreprocessorAction::ExecuteAction() {
  using namespace clang;

  GSL_LOG(INFO) << "Start preprocessing ...";

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

  GSL_LOG(INFO) << "Done preprocessing";
}

} // namespace gtclang
