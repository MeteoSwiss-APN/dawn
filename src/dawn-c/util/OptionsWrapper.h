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

#ifndef DAWN_C_UTIL_OPTIONSWRAPPER_H
#define DAWN_C_UTIL_OPTIONSWRAPPER_H

#include "dawn-c/ErrorHandling.h"
#include "dawn-c/Types.h"
#include "dawn/Compiler/Options.h"
#include "dawn/Support/Assert.h"
#include <cstring>
#include <string>
#include <unordered_map>

namespace dawn {

namespace util {

namespace internal {

template <class T>
struct OptionsInfo {};

template <>
struct OptionsInfo<bool> {
  static constexpr DawnTypeKind TypeKind = DT_Integer;
  using Type = int;

  static std::size_t getSizeInBytes(const bool& value) { return sizeof(Type); }
  static void* allocateCopy(const bool& value) {
    auto ptr = (Type*)std::malloc(getSizeInBytes(value));
    *ptr = value;
    return ptr;
  }
};

template <>
struct OptionsInfo<int> {
  static constexpr DawnTypeKind TypeKind = DT_Integer;
  using Type = int;

  static std::size_t getSizeInBytes(const int& value) { return sizeof(Type); }
  static void* allocateCopy(const int& value) {
    auto ptr = (Type*)std::malloc(getSizeInBytes(value));
    *ptr = value;
    return ptr;
  }
};

template <>
struct OptionsInfo<float> {
  static constexpr DawnTypeKind TypeKind = DT_Double;
  using Type = double;

  static std::size_t getSizeInBytes(const float& value) { return sizeof(Type); }
  static void* allocateCopy(const float& value) {
    auto ptr = (Type*)std::malloc(getSizeInBytes(value));
    *ptr = value;
    return ptr;
  }
};

template <>
struct OptionsInfo<double> {
  static constexpr DawnTypeKind TypeKind = DT_Double;
  using Type = double;

  static std::size_t getSizeInBytes(const double& value) { return sizeof(Type); }
  static void* allocateCopy(const double& value) {
    auto ptr = (Type*)std::malloc(getSizeInBytes(value));
    *ptr = value;
    return ptr;
  }
};

template <>
struct OptionsInfo<const char*> {
  static constexpr DawnTypeKind TypeKind = DT_Char;
  using Type = char;

  static std::size_t getSizeInBytes(const char* value) { return std::strlen(value) + 1; }
  static void* allocateCopy(const char* value) {
    std::size_t size = getSizeInBytes(value);
    auto ptr = (Type*)std::malloc(size);
    std::memcpy(ptr, value, size);
    return ptr;
  }
};

template <>
struct OptionsInfo<std::string> {
  static constexpr DawnTypeKind TypeKind = DT_Char;
  using Type = char;

  static std::size_t getSizeInBytes(const std::string& value) { return value.size() + 1; }
  static void* allocateCopy(const std::string& value) {
    std::size_t size = getSizeInBytes(value);
    auto ptr = (Type*)std::malloc(size);
    std::memcpy(ptr, value.c_str(), size);
    return ptr;
  }
};

template <class T>
struct ValueGetter {};

template <>
struct ValueGetter<bool> {
  static bool get(const dawnOptionsEntry_t* entry) {
    DAWN_ASSERT_MSG(entry->Type == DT_Integer, "invalid type, expected int");
    return *(int*)entry->Value;
  }
};

template <>
struct ValueGetter<int> {
  static int get(const dawnOptionsEntry_t* entry) {
    DAWN_ASSERT_MSG(entry->Type == DT_Integer, "invalid type, expected int");
    return *(int*)entry->Value;
  }
};

template <>
struct ValueGetter<double> {
  static double get(const dawnOptionsEntry_t* entry) {
    DAWN_ASSERT_MSG(entry->Type == DT_Double, "invalid type, expected double");
    return *(double*)entry->Value;
  }
};

template <>
struct ValueGetter<std::string> {
  static std::string get(const dawnOptionsEntry_t* entry) {
    DAWN_ASSERT_MSG(entry->Type == DT_Char, "invalid type, expected string");
    return std::string((const char*)entry->Value);
  }
};

} // namespace internal

/// @brief Warpper for dawnOptionsEntry_t`
/// @ingroup dawn_c_util
struct OptionsEntryWrapper {
  OptionsEntryWrapper() = delete;

  /// @brief Construct `dawnOptionsEntry_t` from `value`
  template <class T>
  static dawnOptionsEntry_t* construct(const T& value) {
    dawnOptionsEntry_t* entry = (dawnOptionsEntry_t*)std::malloc(sizeof(dawnOptionsEntry_t));
    using OptionInfoT = internal::OptionsInfo<T>;
    entry->Type = OptionInfoT::TypeKind;
    entry->SizeInBytes = OptionInfoT::getSizeInBytes(value);
    entry->Value = OptionInfoT::allocateCopy(value);
    return entry;
  }

  /// @brief Deallocate `dawnOptionsEntry_t`
  static void destroy(dawnOptionsEntry_t* entry) {
    std::free(entry->Value);
    std::free(entry);
  }

  /// @brief Copy construct `dawnOptionsEntry_t`
  static dawnOptionsEntry_t* copy(const dawnOptionsEntry_t* entry) {
    dawnOptionsEntry_t* entryCopy = (dawnOptionsEntry_t*)std::malloc(sizeof(dawnOptionsEntry_t));
    entryCopy->Type = entry->Type;
    entryCopy->SizeInBytes = entry->SizeInBytes;
    entryCopy->Value = std::malloc(entry->SizeInBytes);
    std::memcpy(entryCopy->Value, entry->Value, entry->SizeInBytes);
    return entryCopy;
  }

  /// @brief Extract the value from an entry of type `T`
  template <class T>
  static T getValue(const dawnOptionsEntry_t* entry) {
    return internal::ValueGetter<T>::get(entry);
  }
};

/// @brief Warpper for `dawn::Options` which provides types erased string based access
/// @ingroup dawn_c_util
class OptionsWrapper {
public:
  OptionsWrapper() {
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  this->setOption<TYPE>(#NAME, DEFAULT_VALUE);
#include "dawn/Compiler/Options.inc"
#undef OPT
  }

  ~OptionsWrapper();

  /// @brief Set option `name` to `value`
  template <class T>
  void setOption(std::string name, const T& value) noexcept {
    options_[name] = OptionsEntryWrapper::construct(value);
  }

  /// @brief Insert a copy of the options entry `name`
  void setOptionEntry(std::string name, const dawnOptionsEntry_t* value) noexcept {
    options_[name] = OptionsEntryWrapper::copy(value);
  }

  /// @brief Check if option `name` exists
  bool hasOption(std::string name) const noexcept;

  /// @brief Returns a copy of the option `name` or `NULL` if option does not exist
  const dawnOptionsEntry_t* getOption(std::string name) const noexcept;

  /// @brief Set the `dawn::Options` to the content of the wrapper
  void setDawnOptions(dawn::Options* options) const noexcept;

  /// @brief Convert to string
  char* toString() const;

private:
  std::unordered_map<std::string, dawnOptionsEntry_t*> options_;
};

/// @brief Convert `dawnOptions_t` to `OptionsWrapper`
/// @ingroup dawn_c_util
/// @{
inline const OptionsWrapper* toConstOptionsWrapper(const dawnOptions_t* options) {
  if(!options->Impl)
    dawnFatalError("uninitialized Options");
  return reinterpret_cast<const OptionsWrapper*>(options->Impl);
}

inline OptionsWrapper* toOptionsWrapper(dawnOptions_t* options) {
  if(!options->Impl)
    dawnFatalError("uninitialized Options");
  return reinterpret_cast<OptionsWrapper*>(options->Impl);
}
/// @}

} // namespace util

} // namespace dawn

#endif
