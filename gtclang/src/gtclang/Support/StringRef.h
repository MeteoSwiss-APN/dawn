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
//
// This file is a port of LLVM's StringRef.
//
//===------------------------------------------------------------------------------------------===//

#pragma once

#include "dawn/Support/Assert.h"
#include "dawn/Support/Compiler.h"
#include "dawn/Support/STLExtras.h"
#include <functional>
#include <algorithm>
#include <cstring>
#include <iosfwd>
#include <limits>
#include <string>
#include <utility>

namespace gtclang {

template <typename T>
class SmallVectorImpl;
class StringRef;

/// @ingroup support
/// @{

bool getAsUnsignedInteger(StringRef Str, unsigned Radix, unsigned long long& Result);
bool getAsSignedInteger(StringRef Str, unsigned Radix, long long& Result);
bool consumeUnsignedInteger(StringRef& Str, unsigned Radix, unsigned long long& Result);
bool consumeSignedInteger(StringRef& Str, unsigned Radix, long long& Result);

/// @brief Represent a constant reference to a string, i.e. a character array and a length, which
/// need not be null terminated.
///
/// This class does not own the string data, it is expected to be used in situations where the
/// character data resides in some other buffer, whose lifetime extends past that of the StringRef.
/// For this reason, it is not in general safe to store a StringRef.
class StringRef {
public:
  typedef const char* iterator;
  typedef const char* const_iterator;
  static const size_t npos = ~size_t(0);
  typedef size_t size_type;

private:
  /// The start of the string, in an external buffer.
  const char* Data = nullptr;

  /// The length of the string.
  size_t Length = 0;

  // Workaround memcmp issue with null pointers (undefined behavior)
  // by providing a specialized version
  inline static int compareMemory(const char* Lhs, const char* Rhs, size_t Length) {
    if(Length == 0) {
      return 0;
    }
    return ::memcmp(Lhs, Rhs, Length);
  }

public:
  /// @name Constructors
  /// @{

  /// @brief Construct an empty string ref
  StringRef() = default;

  /// @brief Disable conversion from nullptr.  This prevents things like if (S == nullptr)
  StringRef(std::nullptr_t) = delete;

  /// @brief Construct a string ref from a cstring
  inline StringRef(const char* Str) : Data(Str), Length(Str ? ::strlen(Str) : 0) {}

  /// @brief Construct a string ref from a pointer and length
  inline constexpr StringRef(const char* data, size_t length) : Data(data), Length(length) {}

  /// @brief Construct a string ref from an std::string
  inline StringRef(const std::string& Str) : Data(Str.data()), Length(Str.length()) {}

  static StringRef withNullAsEmpty(const char* data) { return StringRef(data ? data : ""); }

  /// @}
  /// @name Iterators
  /// @{

  iterator begin() const { return Data; }

  iterator end() const { return Data + Length; }

  const unsigned char* bytes_begin() const {
    return reinterpret_cast<const unsigned char*>(begin());
  }
  const unsigned char* bytes_end() const { return reinterpret_cast<const unsigned char*>(end()); }

  /// @}
  /// @name String Operations
  /// @{

  /// @brief  Get a pointer to the start of the string (which may not be null terminated)
  inline const char* data() const { return Data; }

  /// @brief Check if the string is empty
  inline bool empty() const { return Length == 0; }

  /// @brief Get the string size.
  inline size_t size() const { return Length; }

  /// @brief Get the first character in the string
  char front() const {
    DAWN_ASSERT(!empty());
    return Data[0];
  }

  /// @brief Get the last character in the string
  char back() const {
    DAWN_ASSERT(!empty());
    return Data[Length - 1];
  }

  /// @brief Check for string equality, this is more efficient than compare() when the relative
  /// ordering of inequal strings isn't needed
  inline bool equals(StringRef RHS) const {
    return (Length == RHS.Length && compareMemory(Data, RHS.Data, RHS.Length) == 0);
  }

  /// @brief Check for string equality, ignoring case.
  bool equals_lower(StringRef RHS) const { return Length == RHS.Length && compare_lower(RHS) == 0; }

  /// @brief Compare two strings; the result is -1, 0, or 1 if this string is lexicographically
  /// less than, equal to, or greater than the \p RHS
  inline int compare(StringRef RHS) const {
    // Check the prefix for a mismatch.
    if(int Res = compareMemory(Data, RHS.Data, std::min(Length, RHS.Length)))
      return Res < 0 ? -1 : 1;

    // Otherwise the prefixes match, so we only need to check the lengths.
    if(Length == RHS.Length)
      return 0;
    return Length < RHS.Length ? -1 : 1;
  }

  /// @brief Compare two strings, ignoring case
  int compare_lower(StringRef RHS) const;

  /// @brief Compare two strings, treating sequences of digits as numbers
  int compare_numeric(StringRef RHS) const;

  /// @brief Determine the edit distance between this string and another string
  ///
  /// @param Other              The string to compare this string against.
  /// @param AllowReplacements  Whether to allow character replacements (change one character into
  ///                           another) as a single operation, rather than as two operations
  ///                           (an insertion and a removal).
  /// @param MaxEditDistance    If non-zero, the maximum edit distance that this routine is allowed
  ///                           to compute. If the edit distance will exceed that maximum, returns
  ///                           `MaxEditDistance+1`.
  ///
  /// @returns the minimum number of character insertions, removals, or (if @p AllowReplacements
  /// is `true`) replacements needed to transform one of the given strings into the other. If zero,
  /// the strings are identical.
  ///
  unsigned edit_distance(StringRef Other, bool AllowReplacements = true,
                         unsigned MaxEditDistance = 0) const;

  /// @brief Get the contents as an std::string.
  std::string str() const {
    if(!Data)
      return std::string();
    return std::string(Data, Length);
  }

  /// @}
  /// @name Operator Overloads
  /// @{

  char operator[](size_t Index) const {
    DAWN_ASSERT_MSG(Index < Length, "Invalid index!");
    return Data[Index];
  }

  /// @brief Disallow accidental assignment from a temporary std::string
  ///
  /// The declaration here is extra complicated so that `stringRef = {}` and `stringRef = "abc"`
  /// continue to select the move assignment operator.
  template <typename T>
  typename std::enable_if<std::is_same<T, std::string>::value, StringRef>::type&
  operator=(T&& Str) = delete;

  /// @}
  /// @name Type Conversions
  /// @{

  operator std::string() const { return str(); }

  /// @}
  /// @name String Predicates
  /// @{

  /// @brief Check if this string starts with the given @p Prefix
  inline bool startswith(StringRef Prefix) const {
    return Length >= Prefix.Length && compareMemory(Data, Prefix.Data, Prefix.Length) == 0;
  }

  /// @brief Check if this string starts with the given @p Prefix, ignoring case.
  bool startswith_lower(StringRef Prefix) const;

  /// @brief Check if this string ends with the given @p Suffix
  inline bool endswith(StringRef Suffix) const {
    return Length >= Suffix.Length &&
           compareMemory(end() - Suffix.Length, Suffix.Data, Suffix.Length) == 0;
  }

  /// @brief Check if this string ends with the given @p Suffix, ignoring case.
  bool endswith_lower(StringRef Suffix) const;

  /// @}
  /// @name String Searching
  /// @{

  /// @brief Search for the first character @p C in the string
  ///
  /// @returns The index of the first occurrence of @p C, or npos if not found.
  inline size_t find(char C, size_t From = 0) const {
    size_t FindBegin = std::min(From, Length);
    if(FindBegin < Length) { // Avoid calling memchr with nullptr.
      // Just forward to memchr, which is faster than a hand-rolled loop.
      if(const void* P = ::memchr(Data + FindBegin, C, Length - FindBegin))
        return static_cast<const char*>(P) - Data;
    }
    return npos;
  }

  /// @brief Search for the first character @p C in the string, ignoring case
  ///
  /// @returns The index of the first occurrence of @p C, or npos if not found.
  size_t find_lower(char C, size_t From = 0) const;

  /// @brief Search for the first character satisfying the predicate @p F
  ///
  /// @returns The index of the first character satisfying @p F starting from @p From, or npos if
  /// not found.
  inline size_t find_if(std::function<bool(char)> F, size_t From = 0) const {
    StringRef S = drop_front(From);
    while(!S.empty()) {
      if(F(S.front()))
        return size() - S.size();
      S = S.drop_front();
    }
    return npos;
  }

  /// @brief Search for the first character not satisfying the predicate \p F
  ///
  /// @returns The index of the first character not satisfying @p F starting from @p From, or npos
  /// if not found.
  inline size_t find_if_not(std::function<bool(char)> F, size_t From = 0) const {
    return find_if([F](char c) { return !F(c); }, From);
  }

  /// @brief Search for the first string @p Str in the string
  ///
  /// @returns The index of the first occurrence of @p Str, or npos if not found.
  size_t find(StringRef Str, size_t From = 0) const;

  /// @brief Search for the first string @p Str in the string, ignoring case
  ///
  /// @returns The index of the first occurrence of @p Str, or npos if not found.
  size_t find_lower(StringRef Str, size_t From = 0) const;

  /// @brief Search for the last character \p C in the string
  ///
  /// @returns The index of the last occurrence of \p C, or npos if not found.
  size_t rfind(char C, size_t From = npos) const {
    From = std::min(From, Length);
    size_t i = From;
    while(i != 0) {
      --i;
      if(Data[i] == C)
        return i;
    }
    return npos;
  }

  /// @brief Search for the last character @p C in the string, ignoring case
  ///
  /// @returns The index of the last occurrence of @p C, or npos if not found.
  size_t rfind_lower(char C, size_t From = npos) const;

  /// @brief Search for the last string @p Str in the string
  ///
  /// @returns The index of the last occurrence of @p Str, or npos if not found.
  size_t rfind(StringRef Str) const;

  /// @brief Search for the last string @p Str in the string, ignoring case.
  ///
  /// @returns The index of the last occurrence of @p Str, or npos if not found.
  size_t rfind_lower(StringRef Str) const;

  /// @brief Find the first character in the string that is @p C, or npos if not found.
  /// Same as find.
  size_t find_first_of(char C, size_t From = 0) const { return find(C, From); }

  /// @brief Find the first character in the string that is in @p Chars, or npos if not found
  ///
  /// @b Complexity: O(size() + Chars.size())
  size_t find_first_of(StringRef Chars, size_t From = 0) const;

  /// @brief Find the first character in the string that is not @p C or npos if not found
  size_t find_first_not_of(char C, size_t From = 0) const;

  /// @brief Find the first character in the string that is not in the string @p Chars, or npos if
  /// not found.
  ///
  /// @b Complexity: O(size() + Chars.size())
  size_t find_first_not_of(StringRef Chars, size_t From = 0) const;

  /// @brief Find the last character in the string that is @p C, or npos if not found
  size_t find_last_of(char C, size_t From = npos) const { return rfind(C, From); }

  /// @brief Find the last character in the string that is in @p C, or npos if not found
  ///
  /// @b Complexity: O(size() + Chars.size())
  size_t find_last_of(StringRef Chars, size_t From = npos) const;

  /// @brief Find the last character in the string that is not @p C, or npos if not found
  size_t find_last_not_of(char C, size_t From = npos) const;

  /// @brief Find the last character in the string that is not in @p Chars, or npos if not found
  ///
  /// @b Complexity: O(size() + Chars.size())
  size_t find_last_not_of(StringRef Chars, size_t From = npos) const;

  /// @brief Return true if the given string is a substring of *this, and false otherwise
  inline bool contains(StringRef Other) const { return find(Other) != npos; }

  /// @brief Return true if the given character is contained in *this, and false otherwise
  inline bool contains(char C) const { return find_first_of(C) != npos; }

  /// @brief Return true if the given string is a substring of *this, and false otherwise
  inline bool contains_lower(StringRef Other) const { return find_lower(Other) != npos; }

  /// @brief Return true if the given character is contained in *this, and false otherwise
  inline bool contains_lower(char C) const { return find_lower(C) != npos; }

  /// @}
  /// @name Helpful Algorithms
  /// @{

  /// @brief Return the number of occurrences of @p C in the string
  size_t count(char C) const {
    size_t Count = 0;
    for(size_t i = 0, e = Length; i != e; ++i)
      if(Data[i] == C)
        ++Count;
    return Count;
  }

  /// @brief Return the number of non-overlapped occurrences of \p Str in the string
  size_t count(StringRef Str) const;

  /// @brief Parse the current string as an integer of the specified radix.
  ///
  /// If @p Radix is specified as zero, this does radix autosensing using  extended C rules: 0 is
  /// octal, 0x is hex, 0b is binary.
  ///
  /// If the string is invalid or if only a subset of the string is valid, this returns true to
  /// signify the error. The string is considered erroneous if empty or if it overflows T.
  template <typename T>
  typename std::enable_if<std::numeric_limits<T>::is_signed, bool>::type
  getAsInteger(unsigned Radix, T& Result) const {
    long long LLVal;
    if(getAsSignedInteger(*this, Radix, LLVal) || static_cast<T>(LLVal) != LLVal)
      return true;
    Result = LLVal;
    return false;
  }

  template <typename T>
  typename std::enable_if<!std::numeric_limits<T>::is_signed, bool>::type
  getAsInteger(unsigned Radix, T& Result) const {
    unsigned long long ULLVal;
    // The additional cast to unsigned long long is required to avoid the
    // Visual C++ warning C4805: '!=' : unsafe mix of type 'bool' and type
    // 'unsigned __int64' when instantiating getAsInteger with T = bool.
    if(getAsUnsignedInteger(*this, Radix, ULLVal) ||
       static_cast<unsigned long long>(static_cast<T>(ULLVal)) != ULLVal)
      return true;
    Result = ULLVal;
    return false;
  }

  /// @brief Parse the current string as an integer of the specified radix.
  ///
  /// If @p Radix is specified as zero, this does radix autosensing using extended C rules: 0 is
  /// octal, 0x is hex, 0b is binary.
  ///
  /// If the string does not begin with a number of the specified radix, this returns true to
  /// signify the error. The string is considered erroneous if empty or if it overflows T.
  /// The portion of the string representing the discovered numeric value is removed from the
  /// beginning of the string.
  template <typename T>
  typename std::enable_if<std::numeric_limits<T>::is_signed, bool>::type
  consumeInteger(unsigned Radix, T& Result) {
    long long LLVal;
    if(consumeSignedInteger(*this, Radix, LLVal) ||
       static_cast<long long>(static_cast<T>(LLVal)) != LLVal)
      return true;
    Result = LLVal;
    return false;
  }

  template <typename T>
  typename std::enable_if<!std::numeric_limits<T>::is_signed, bool>::type
  consumeInteger(unsigned Radix, T& Result) {
    unsigned long long ULLVal;
    if(consumeUnsignedInteger(*this, Radix, ULLVal) ||
       static_cast<unsigned long long>(static_cast<T>(ULLVal)) != ULLVal)
      return true;
    Result = ULLVal;
    return false;
  }

  /// @}
  /// @name String Operations
  /// @{

  /// @brief Convert the given ASCII string to lowercase.
  std::string lower() const;

  /// @brief Convert the given ASCII string to uppercase.
  std::string upper() const;

  /// @}
  /// @name Substring Operations
  /// @{

  /// @brief Return a reference to the substring from [Start, Start + N).
  ///
  /// @param Start        The index of the starting character in the substring; if
  ///                     the index is npos or greater than the length of the string then the
  ///                     empty substring will be returned.
  /// @param N            The number of characters to included in the substring. If N
  ///                     exceeds the number of characters remaining in the string, the string
  ///                     suffix (starting with \p Start) will be returned.
  inline StringRef substr(size_t Start, size_t N = npos) const {
    Start = std::min(Start, Length);
    return StringRef(Data + Start, std::min(N, Length - Start));
  }

  /// @brief Return a StringRef equal to 'this' but with only the first @p N elements remaining.
  /// If @p N is greater than the length of the string, the entire string is returned.
  inline StringRef take_front(size_t N = 1) const {
    if(N >= size())
      return *this;
    return drop_back(size() - N);
  }

  /// @brief Return a StringRef equal to 'this' but with only the first @p N elements remaining.
  ///
  /// If @p N is greater than the length of the string, the entire string is returned.
  inline StringRef take_back(size_t N = 1) const {
    if(N >= size())
      return *this;
    return drop_front(size() - N);
  }

  /// @brief Return the longest prefix of 'this' such that every character in the prefix satisfies
  /// the given predicate.
  inline StringRef take_while(std::function<bool(char)> F) const {
    return substr(0, find_if_not(F));
  }

  /// @brief Return the longest prefix of 'this' such that no character in the prefix satisfies
  /// the given predicate.
  inline StringRef take_until(std::function<bool(char)> F) const { return substr(0, find_if(F)); }

  /// @brief Return a StringRef equal to 'this' but with the first @p N elements dropped
  inline StringRef drop_front(size_t N = 1) const {
    DAWN_ASSERT_MSG(size() >= N, "Dropping more elements than exist");
    return substr(N);
  }

  /// @brief Return a StringRef equal to 'this' but with the last @p N elements dropped
  inline StringRef drop_back(size_t N = 1) const {
    DAWN_ASSERT_MSG(size() >= N, "Dropping more elements than exist");
    return substr(0, size() - N);
  }

  /// @brief Return a StringRef equal to 'this', but with all characters satisfying the given
  /// predicate dropped from the beginning of the string
  inline StringRef drop_while(std::function<bool(char)> F) const { return substr(find_if_not(F)); }

  /// @brief Return a StringRef equal to 'this', but with all characters not satisfying the given
  /// predicate dropped from the beginning of the string
  inline StringRef drop_until(std::function<bool(char)> F) const { return substr(find_if(F)); }

  /// @brief Returns true if this StringRef has the given prefix and removes that prefix
  inline bool consume_front(StringRef Prefix) {
    if(!startswith(Prefix))
      return false;

    *this = drop_front(Prefix.size());
    return true;
  }

  /// @brief Returns true if this StringRef has the given suffix and removes that suffix
  inline bool consume_back(StringRef Suffix) {
    if(!endswith(Suffix))
      return false;

    *this = drop_back(Suffix.size());
    return true;
  }

  /// @brief Return a reference to the substring from [Start, End).
  ///
  /// @param Start        The index of the starting character in the substring; if
  ///                     the index is npos or greater than the length of the string then the
  ///                     empty substring will be returned.
  /// @param End          The index following the last character to include in the
  ///                     substring. If this is npos or exceeds the number of characters
  ///                     remaining in the string, the string suffix (starting with \p Start)
  ///                     will be returned. If this is less than \p Start, an empty string will
  ///                     be returned.
  inline StringRef slice(size_t Start, size_t End) const {
    Start = std::min(Start, Length);
    End = std::min(std::max(Start, End), Length);
    return StringRef(Data + Start, End - Start);
  }

  /// @brief Split into two substrings around the first occurrence of a separator character.
  ///
  /// If \p Separator is in the string, then the result is a pair (LHS, RHS) such that
  /// (*this == LHS + Separator + RHS) is true and RHS is maximal. If @p Separator is not in the
  /// string, then the result is a pair (LHS, RHS) where (*this == LHS) and (RHS == "").
  ///
  /// @param Separator  The character to split on.
  /// @returns The split substrings.
  std::pair<StringRef, StringRef> split(char Separator) const {
    size_t Idx = find(Separator);
    if(Idx == npos)
      return std::make_pair(*this, StringRef());
    return std::make_pair(slice(0, Idx), slice(Idx + 1, npos));
  }

  /// @brief Split into two substrings around the first occurrence of a separator string
  ///
  /// If @p Separator is in the string, then the result is a pair (LHS, RHS) such that
  /// (*this == LHS + Separator + RHS) is true and RHS is maximal. If @p Separator is not in the
  /// string, then the result is a pair (LHS, RHS) where (*this == LHS) and (RHS == "").
  ///
  /// @param Separator - The string to split on.
  /// @return - The split substrings.
  std::pair<StringRef, StringRef> split(StringRef Separator) const {
    size_t Idx = find(Separator);
    if(Idx == npos)
      return std::make_pair(*this, StringRef());
    return std::make_pair(slice(0, Idx), slice(Idx + Separator.size(), npos));
  }

  /// @brief Split into substrings around the occurrences of a separator string
  ///
  /// Each substring is stored in @p A. If @p MaxSplit is >= 0, at most @p MaxSplit splits are done
  /// and consequently <= @p MaxSplit + 1 elements are added to A. If @p KeepEmpty is false, empty
  /// strings are not added to @p A. They still count when considering @p MaxSplit. An useful
  /// invariant is that Separator.join(A) == *this if MaxSplit == -1 and KeepEmpty == true.
  ///
  /// @param A            Where to put the substrings.
  /// @param Separator    The string to split on.
  /// @param MaxSplit     The maximum number of times the string is split.
  /// @param KeepEmpty    True if empty substring should be added.
  void split(SmallVectorImpl<StringRef>& A, StringRef Separator, int MaxSplit = -1,
             bool KeepEmpty = true) const;

  /// @brief Split into substrings around the occurrences of a separator character.
  ///
  /// Each substring is stored in @p A. If @p MaxSplit is >= 0, at most @p MaxSplit splits are done
  /// and consequently <= @p MaxSplit + 1 elements are added to A. If @p KeepEmpty is false, empty
  /// strings are not added to @p A. They still count when considering @p MaxSplit. An useful
  /// invariant is that Separator.join(A) == *this if MaxSplit == -1 and KeepEmpty == true.
  ///
  /// \param A            Where to put the substrings.
  /// \param Separator    The string to split on.
  /// \param MaxSplit     The maximum number of times the string is split.
  /// \param KeepEmpty    True if empty substring should be added.
  void split(SmallVectorImpl<StringRef>& A, char Separator, int MaxSplit = -1,
             bool KeepEmpty = true) const;

  /// @brief Split into two substrings around the last occurrence of a separator character.
  ///
  /// If @p Separator is in the string, then the result is a pair (LHS, RHS) such that
  /// (*this == LHS + Separator + RHS) is true and RHS is minimal. If @p Separator is not in the
  /// string, then the result is a pair (LHS, RHS) where (*this == LHS) and (RHS == "").
  ///
  /// @param Separator - The character to split on.
  /// @return - The split substrings.
  std::pair<StringRef, StringRef> rsplit(char Separator) const {
    size_t Idx = rfind(Separator);
    if(Idx == npos)
      return std::make_pair(*this, StringRef());
    return std::make_pair(slice(0, Idx), slice(Idx + 1, npos));
  }

  /// @brief Return string with consecutive @p Char characters starting from the the left removed
  StringRef ltrim(char Char) const { return drop_front(std::min(Length, find_first_not_of(Char))); }

  /// @brief Return string with consecutive characters in @p Chars starting from the left removed.
  StringRef ltrim(StringRef Chars = " \t\n\v\f\r") const {
    return drop_front(std::min(Length, find_first_not_of(Chars)));
  }

  /// @brief Return string with consecutive @p Char characters starting from the right removed
  StringRef rtrim(char Char) const {
    return drop_back(Length - std::min(Length, find_last_not_of(Char) + 1));
  }

  /// @brief Return string with consecutive characters in @p Chars starting from the right removed.
  StringRef rtrim(StringRef Chars = " \t\n\v\f\r") const {
    return drop_back(Length - std::min(Length, find_last_not_of(Chars) + 1));
  }

  /// @brief Return string with consecutive @p Char characters starting from the left and right
  /// removed.
  StringRef trim(char Char) const { return ltrim(Char).rtrim(Char); }

  /// @brief Return string with consecutive characters in @p Chars starting from the left and right
  /// removed.
  StringRef trim(StringRef Chars = " \t\n\v\f\r") const { return ltrim(Chars).rtrim(Chars); }

  /// @}

  /// @brief Convert to stream
  friend std::ostream& operator<<(std::ostream& stream, const StringRef& str);
};

/// @name StringRef Comparison Operators
/// @{
inline bool operator==(StringRef LHS, StringRef RHS) { return LHS.equals(RHS); }
inline bool operator!=(StringRef LHS, StringRef RHS) { return !(LHS == RHS); }
inline bool operator<(StringRef LHS, StringRef RHS) { return LHS.compare(RHS) == -1; }
inline bool operator<=(StringRef LHS, StringRef RHS) { return LHS.compare(RHS) != 1; }
inline bool operator>(StringRef LHS, StringRef RHS) { return LHS.compare(RHS) == 1; }
inline bool operator>=(StringRef LHS, StringRef RHS) { return LHS.compare(RHS) != -1; }

inline std::string& operator+=(std::string& buffer, StringRef string) {
  return buffer.append(string.data(), string.size());
}
/// @}

// StringRefs can be treated like a POD type.
template <typename T>
struct isPodLike;
template <>
struct isPodLike<StringRef> {
  static const bool value = true;
};

/// @}

} // namespace gtclang
