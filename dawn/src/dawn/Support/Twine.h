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
//
// This file is a port of LLVM's Twine.
//
//===------------------------------------------------------------------------------------------===//

#ifndef DAWN_SUPPORT_TWINE_H
#define DAWN_SUPPORT_TWINE_H

#include "dawn/Support/SmallVector.h"
#include "dawn/Support/StringRef.h"
#include "dawn/Support/Unreachable.h"
#include <cassert>
#include <iosfwd>
#include <string>

namespace dawn {

/// Twine - A lightweight data structure for efficiently representing the
/// concatenation of temporary values as strings.
///
/// A Twine is a kind of rope, it represents a concatenated string using a
/// binary-tree, where the string is the preorder of the nodes. Since the
/// Twine can be efficiently rendered into a buffer when its result is used,
/// it avoids the cost of generating temporary values for intermediate string
/// results -- particularly in cases when the Twine result is never
/// required. By explicitly tracking the type of leaf nodes, we can also avoid
/// the creation of temporary strings for conversions operations (such as
/// appending an integer to a string).
///
/// A Twine is not intended for use directly and should not be stored, its
/// implementation relies on the ability to store pointers to temporary stack
/// objects which may be deallocated at the end of a statement. Twines should
/// only be used accepted as const references in arguments, when an API wishes
/// to accept possibly-concatenated strings.
///
/// Twines support a special 'null' value, which always concatenates to form
/// itself, and renders as an empty string. This can be returned from APIs to
/// effectively nullify any concatenations performed on the result.
///
/// @b Implementation
///
/// Given the nature of a Twine, it is not possible for the Twine's
/// concatenation method to construct interior nodes; the result must be
/// represented inside the returned value. For this reason a Twine object
/// actually holds two values, the left- and right-hand sides of a
/// concatenation. We also have nullary Twine objects, which are effectively
/// sentinel values that represent empty strings.
///
/// Thus, a Twine can effectively have zero, one, or two children. The @see
/// isNullary(), @see isUnary(), and @see isBinary() predicates exist for
/// testing the number of children.
///
/// We maintain a number of invariants on Twine objects (FIXME: Why):
///  - Nullary twines are always represented with their Kind on the left-hand
///    side, and the Empty kind on the right-hand side.
///  - Unary twines are always represented with the value on the left-hand
///    side, and the Empty kind on the right-hand side.
///  - If a Twine has another Twine as a child, that child should always be
///    binary (otherwise it could have been folded into the parent).
///
/// These invariants are check by @see isValid().
///
/// @b Efficiency Considerations
///
/// The Twine is designed to yield efficient and small code for common
/// situations. For this reason, the concat() method is inlined so that
/// concatenations of leaf nodes can be optimized into stores directly into a
/// single stack allocated object.
///
/// In practice, not all compilers can be trusted to optimize concat() fully,
/// so we provide two additional methods (and accompanying operator+
/// overloads) to guarantee that particularly important cases (cstring plus
/// StringRef) codegen as desired.
///
/// @ingroup support
class Twine {
  /// NodeKind - Represent the type of an argument.
  enum class NodeKind : unsigned char {
    /// An empty string; the result of concatenating anything with it is also
    /// empty.
    Null,

    /// The empty string.
    Empty,

    /// A pointer to a Twine instance.
    Twine,

    /// A pointer to a C string instance.
    CString,

    /// A pointer to an std::string instance.
    StdString,

    /// A pointer to a StringRef instance.
    StringRef,

    /// A pointer to a SmallString instance.
    SmallString,

    /// A char value, to render as a character.
    Char,

    /// An unsigned int value, to render as an unsigned decimal integer.
    DecUI,

    /// An int value, to render as a signed decimal integer.
    DecI,

    /// A pointer to an unsigned long value, to render as an unsigned decimal
    /// integer.
    DecUL,

    /// A pointer to a long value, to render as a signed decimal integer.
    DecL,

    /// A pointer to an unsigned long long value, to render as an unsigned
    /// decimal integer.
    DecULL,

    /// A pointer to a long long value, to render as a signed decimal integer.
    DecLL,

    /// A pointer to a uint64_t value, to render as an unsigned hexadecimal
    /// integer.
    UHex
  };

  union Child {
    const Twine* twine;
    const char* cString;
    const std::string* stdString;
    const StringRef* stringRef;
    const SmallVectorImpl<char>* smallString;
    char character;
    unsigned int decUI;
    int decI;
    const unsigned long* decUL;
    const long* decL;
    const unsigned long long* decULL;
    const long long* decLL;
    const uint64_t* uHex;
  };

private:
  /// LHS - The prefix in the concatenation, which may be uninitialized for
  /// Null or Empty kinds.
  Child lhs_;
  /// RHS - The suffix in the concatenation, which may be uninitialized for
  /// Null or Empty kinds.
  Child rhs_;
  /// LHSKind - The NodeKind of the left hand side, \see getLHSKind().
  NodeKind lhsKind_;
  /// RHSKind - The NodeKind of the right hand side, \see getRHSKind().
  NodeKind rhsKind_;

private:
  /// Construct a nullary twine; the kind must be NodeKind::Null or NodeKind::Empty.
  explicit Twine(NodeKind Kind) : lhsKind_(Kind), rhsKind_(NodeKind::Empty) {
    assert(isNullary() && "Invalid kind!");
  }

  /// Construct a binary twine.
  explicit Twine(const Twine& LHS, const Twine& RHS)
      : lhsKind_(NodeKind::Twine), rhsKind_(NodeKind::Twine) {
    this->lhs_.twine = &LHS;
    this->rhs_.twine = &RHS;
    assert(isValid() && "Invalid twine!");
  }

  /// Construct a twine from explicit values.
  explicit Twine(Child LHS, NodeKind LHSKind, Child RHS, NodeKind RHSKind)
      : lhs_(LHS), rhs_(RHS), lhsKind_(LHSKind), rhsKind_(RHSKind) {
    assert(isValid() && "Invalid twine!");
  }

  /// Since the intended use of twines is as temporary objects, assignments
  /// when concatenating might cause undefined behavior or stack corruptions
  Twine& operator=(const Twine& Other) = delete;

  /// Check for the null twine.
  bool isNull() const { return getLHSKind() == NodeKind::Null; }

  /// Check for the empty twine.
  bool isEmpty() const { return getLHSKind() == NodeKind::Empty; }

  /// Check if this is a nullary twine (null or empty).
  bool isNullary() const { return isNull() || isEmpty(); }

  /// Check if this is a unary twine.
  bool isUnary() const { return getRHSKind() == NodeKind::Empty && !isNullary(); }

  /// Check if this is a binary twine.
  bool isBinary() const {
    return getLHSKind() != NodeKind::Null && getRHSKind() != NodeKind::Empty;
  }

  /// Check if this is a valid twine (satisfying the invariants on
  /// order and number of arguments).
  bool isValid() const {
    // Nullary twines always have Empty on the RHS.
    if(isNullary() && getRHSKind() != NodeKind::Empty)
      return false;

    // Null should never appear on the RHS.
    if(getRHSKind() == NodeKind::Null)
      return false;

    // The RHS cannot be non-empty if the LHS is empty.
    if(getRHSKind() != NodeKind::Empty && getLHSKind() == NodeKind::Empty)
      return false;

    // A twine child should always be binary.
    if(getLHSKind() == NodeKind::Twine && !lhs_.twine->isBinary())
      return false;
    if(getRHSKind() == NodeKind::Twine && !rhs_.twine->isBinary())
      return false;

    return true;
  }

  /// Get the NodeKind of the left-hand side.
  NodeKind getLHSKind() const { return lhsKind_; }

  /// Get the NodeKind of the right-hand side.
  NodeKind getRHSKind() const { return rhsKind_; }

  /// Print one child from a twine.
  void printOneChild(std::ostream& OS, Child Ptr, NodeKind Kind) const;

  /// Print the representation of one child from a twine.
  void printOneChildRepr(std::ostream& OS, Child Ptr, NodeKind Kind) const;

public:
  /// @name Constructors
  /// @{

  /// Construct from an empty string.
  /*implicit*/ Twine() : lhsKind_(NodeKind::Empty), rhsKind_(NodeKind::Empty) {
    assert(isValid() && "Invalid twine!");
  }

  Twine(const Twine&) = default;

  /// Construct from a C string.
  ///
  /// We take care here to optimize "" into the empty twine -- this will be
  /// optimized out for string constants. This allows Twine arguments have
  /// default "" values, without introducing unnecessary string constants.
  /*implicit*/ Twine(const char* Str) : rhsKind_(NodeKind::Empty) {
    if(Str[0] != '\0') {
      lhs_.cString = Str;
      lhsKind_ = NodeKind::CString;
    } else
      lhsKind_ = NodeKind::Empty;

    assert(isValid() && "Invalid twine!");
  }

  /// Construct from an std::string.
  /*implicit*/ Twine(const std::string& Str)
      : lhsKind_(NodeKind::StdString), rhsKind_(NodeKind::Empty) {
    lhs_.stdString = &Str;
    assert(isValid() && "Invalid twine!");
  }

  /// Construct from a StringRef.
  /*implicit*/ Twine(const StringRef& Str)
      : lhsKind_(NodeKind::StringRef), rhsKind_(NodeKind::Empty) {
    lhs_.stringRef = &Str;
    assert(isValid() && "Invalid twine!");
  }

  /// Construct from a SmallString.
  /*implicit*/ Twine(const SmallVectorImpl<char>& Str)
      : lhsKind_(NodeKind::SmallString), rhsKind_(NodeKind::Empty) {
    lhs_.smallString = &Str;
    assert(isValid() && "Invalid twine!");
  }

  /// Construct from a char.
  explicit Twine(char Val) : lhsKind_(NodeKind::Char), rhsKind_(NodeKind::Empty) {
    lhs_.character = Val;
  }

  /// Construct from a signed char.
  explicit Twine(signed char Val) : lhsKind_(NodeKind::Char), rhsKind_(NodeKind::Empty) {
    lhs_.character = static_cast<char>(Val);
  }

  /// Construct from an unsigned char.
  explicit Twine(unsigned char Val) : lhsKind_(NodeKind::Char), rhsKind_(NodeKind::Empty) {
    lhs_.character = static_cast<char>(Val);
  }

  /// Construct a twine to print \p Val as an unsigned decimal integer.
  explicit Twine(unsigned Val) : lhsKind_(NodeKind::DecUI), rhsKind_(NodeKind::Empty) {
    lhs_.decUI = Val;
  }

  /// Construct a twine to print \p Val as a signed decimal integer.
  explicit Twine(int Val) : lhsKind_(NodeKind::DecI), rhsKind_(NodeKind::Empty) { lhs_.decI = Val; }

  /// Construct a twine to print \p Val as an unsigned decimal integer.
  explicit Twine(const unsigned long& Val) : lhsKind_(NodeKind::DecUL), rhsKind_(NodeKind::Empty) {
    lhs_.decUL = &Val;
  }

  /// Construct a twine to print \p Val as a signed decimal integer.
  explicit Twine(const long& Val) : lhsKind_(NodeKind::DecL), rhsKind_(NodeKind::Empty) {
    lhs_.decL = &Val;
  }

  /// Construct a twine to print \p Val as an unsigned decimal integer.
  explicit Twine(const unsigned long long& Val)
      : lhsKind_(NodeKind::DecULL), rhsKind_(NodeKind::Empty) {
    lhs_.decULL = &Val;
  }

  /// Construct a twine to print \p Val as a signed decimal integer.
  explicit Twine(const long long& Val) : lhsKind_(NodeKind::DecLL), rhsKind_(NodeKind::Empty) {
    lhs_.decLL = &Val;
  }

  // FIXME: Unfortunately, to make sure this is as efficient as possible we
  // need extra binary constructors from particular types. We can't rely on
  // the compiler to be smart enough to fold operator+()/concat() down to the
  // right thing. Yet.

  /// Construct as the concatenation of a C string and a StringRef.
  /*implicit*/ Twine(const char* LHS, const StringRef& RHS)
      : lhsKind_(NodeKind::CString), rhsKind_(NodeKind::StringRef) {
    this->lhs_.cString = LHS;
    this->rhs_.stringRef = &RHS;
    assert(isValid() && "Invalid twine!");
  }

  /// Construct as the concatenation of a StringRef and a C string.
  /*implicit*/ Twine(const StringRef& LHS, const char* RHS)
      : lhsKind_(NodeKind::StringRef), rhsKind_(NodeKind::CString) {
    this->lhs_.stringRef = &LHS;
    this->rhs_.cString = RHS;
    assert(isValid() && "Invalid twine!");
  }

  /// Create a 'null' string, which is an empty string that always
  /// concatenates to form another empty string.
  static Twine createNull() { return Twine(NodeKind::Null); }

  /// @}
  /// @name Numeric Conversions
  /// @{

  // Construct a twine to print \p Val as an unsigned hexadecimal integer.
  static Twine utohexstr(const uint64_t& Val) {
    Child LHS, RHS;
    LHS.uHex = &Val;
    RHS.twine = nullptr;
    return Twine(LHS, NodeKind::UHex, RHS, NodeKind::Empty);
  }

  /// @}
  /// @name Predicate Operations
  /// @{

  /// Check if this twine is trivially empty; a false return value does not
  /// necessarily mean the twine is empty.
  bool isTriviallyEmpty() const { return isNullary(); }

  /// Return true if this twine can be dynamically accessed as a single
  /// StringRef value with getSingleStringRef().
  bool isSingleStringRef() const {
    if(getRHSKind() != NodeKind::Empty)
      return false;

    switch(getLHSKind()) {
    case NodeKind::Empty:
    case NodeKind::CString:
    case NodeKind::StdString:
    case NodeKind::StringRef:
    case NodeKind::SmallString:
      return true;
    default:
      return false;
    }
  }

  /// @}
  /// @name String Operations
  /// @{

  Twine concat(const Twine& Suffix) const;

  /// @}
  /// @name Output & Conversion.
  /// @{

  /// Return the twine contents as a std::string.
  std::string str() const;

  /// Append the concatenated string into the given SmallString or SmallVector.
  void toVector(SmallVectorImpl<char>& Out) const;

  /// This returns the twine as a single StringRef.  This method is only valid
  /// if isSingleStringRef() is true.
  StringRef getSingleStringRef() const {
    assert(isSingleStringRef() && "This cannot be had as a single stringref!");
    switch(getLHSKind()) {
    default:
      dawn_unreachable("Out of sync with isSingleStringRef");
    case NodeKind::Empty:
      return StringRef();
    case NodeKind::CString:
      return StringRef(lhs_.cString);
    case NodeKind::StdString:
      return StringRef(*lhs_.stdString);
    case NodeKind::StringRef:
      return *lhs_.stringRef;
    case NodeKind::SmallString:
      return StringRef(lhs_.smallString->data(), lhs_.smallString->size());
    }
  }

  /// This returns the twine as a single StringRef if it can be
  /// represented as such. Otherwise the twine is written into the given
  /// SmallVector and a StringRef to the SmallVector's data is returned.
  StringRef toStringRef(SmallVectorImpl<char>& Out) const {
    if(isSingleStringRef())
      return getSingleStringRef();
    toVector(Out);
    return StringRef(Out.data(), Out.size());
  }

  /// This returns the twine as a single null terminated StringRef if it
  /// can be represented as such. Otherwise the twine is written into the
  /// given SmallVector and a StringRef to the SmallVector's data is returned.
  ///
  /// The returned StringRef's size does not include the null terminator.
  StringRef toNullTerminatedStringRef(SmallVectorImpl<char>& Out) const;

  /// Write the concatenated string represented by this twine to the
  /// stream \p OS.
  void print(std::ostream& OS) const;

  /// Dump the concatenated string represented by this twine to stderr.
  void dump() const;

  /// Write the representation of this twine to the stream \p OS.
  void printRepr(std::ostream& OS) const;

  /// Dump the representation of this twine to stderr.
  void dumpRepr() const;

  /// @}
};

/// @name Twine Inline Implementations
/// @{

inline Twine Twine::concat(const Twine& Suffix) const {
  // Concatenation with null is null.
  if(isNull() || Suffix.isNull())
    return Twine(NodeKind::Null);

  // Concatenation with empty yields the other side.
  if(isEmpty())
    return Suffix;
  if(Suffix.isEmpty())
    return *this;

  // Otherwise we need to create a new node, taking care to fold in unary
  // twines.
  Child NewLHS, NewRHS;
  NewLHS.twine = this;
  NewRHS.twine = &Suffix;
  NodeKind NewLHSKind = NodeKind::Twine, NewRHSKind = NodeKind::Twine;
  if(isUnary()) {
    NewLHS = lhs_;
    NewLHSKind = getLHSKind();
  }
  if(Suffix.isUnary()) {
    NewRHS = Suffix.lhs_;
    NewRHSKind = Suffix.getLHSKind();
  }

  return Twine(NewLHS, NewLHSKind, NewRHS, NewRHSKind);
}

inline Twine operator+(const Twine& LHS, const Twine& RHS) { return LHS.concat(RHS); }

// Additional overload to guarantee simplified codegen; this is equivalent to concat()
inline Twine operator+(const char* LHS, const StringRef& RHS) { return Twine(LHS, RHS); }

// Additional overload to guarantee simplified codegen; this is equivalent to concat()
inline Twine operator+(const StringRef& LHS, const char* RHS) { return Twine(LHS, RHS); }

inline std::ostream& operator<<(std::ostream& OS, const Twine& RHS) {
  RHS.print(OS);
  return OS;
}

/// @}

} // namespace dawn

#endif