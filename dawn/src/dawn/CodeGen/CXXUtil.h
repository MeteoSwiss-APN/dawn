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

#pragma once

#include "dawn/Support/Assert.h"
#include "dawn/Support/Printing.h"
#include "dawn/Support/StringUtil.h"
#include <functional>
#include <sstream>

namespace dawn {

/// @namespace codegen
/// @brief This namespace contains a simple DSL for writing C++ code
namespace codegen {

namespace internal {

inline std::ostream& indent(int level, std::stringstream& ss) {
  switch(level) {
  case 0:
    return (ss << MakeIndent<0>::value);
  case 1:
    return (ss << MakeIndent<1>::value);
  case 2:
    return (ss << MakeIndent<2>::value);
  case 3:
    return (ss << MakeIndent<3>::value);
  case 4:
    return (ss << MakeIndent<4>::value);
  default:
    return (ss << std::string(DAWN_PRINT_INDENT * level, ' '));
  }
}

template <typename T, typename Sig>
struct hasCommitImplDetail {
  template <typename U, U>
  struct TypeCheck;
  template <typename V>
  static char (&chk(TypeCheck<Sig, &V::commitImpl>*))[1];
  template <typename>
  static char (&chk(...))[2];
  static constexpr bool value = (sizeof(chk<T>(0)) == 1);
};

template <class T>
struct hasCommitImpl {
  static constexpr bool value = hasCommitImplDetail<T, void (T::*)()>::value;
};

} // namespace internal

/// @brief Clear the string stream and return it again
inline std::stringstream& clear(std::stringstream& ss) {
  ss.clear();
  ss.str("");
  return ss;
}

//===------------------------------------------------------------------------------------------===//
//     Streamable
//===------------------------------------------------------------------------------------------===//

/// @brief Streamable: Wrapper of a string stream
/// @ingroup codegen
class Streamable {
protected:
  bool isCommitted_;
  std::reference_wrapper<std::stringstream> ss_;

public:
  /// @brief Construct the streamable object with a stringstream and the indent
  /// level `il`
  Streamable(std::stringstream& s, int il = 0) : isCommitted_(false), ss_(s) {
    internal::indent(il, ss());
  }

  /// @brief Stream data to the underlying string stream
  template <class T>
  Streamable& operator<<(T&& data) {
    ss() << data;
    return *this;
  }

  ~Streamable() { isCommitted_ = true; }

  /// @brief Indent to level
  std::ostream& indentImpl(int level, bool nl = false) {
    if(nl)
      newlineImpl();
    return internal::indent(level, ss());
  }

  /// @brief Skip to new line
  std::ostream& newlineImpl() { return (ss() << "\n"); }

  /// @brief Manually flush to to the stream
  void commit() { isCommitted_ = true; }

  /// @brief Check if we already committed the end to the stream
  bool isCommitted() const { return isCommitted_; }

  /// @brief Get a reference to the string stream
  std::stringstream& ss() { return ss_.get(); }

  /// @brief Get the underlying str of the stream
  std::string str() {
    commit();
    return ss().str();
  }
};

#define DAWN_DECL_COMMIT(ClassName, Base)                                                          \
  ~ClassName() {                                                                                   \
    if(!isCommitted())                                                                             \
      commitImpl();                                                                                \
  }                                                                                                \
  void commit() {                                                                                  \
    if(!isCommitted()) {                                                                           \
      commitImpl();                                                                                \
      Base::commit();                                                                              \
    }                                                                                              \
  }                                                                                                \
  std::string str() {                                                                              \
    commit();                                                                                      \
    return Base::str();                                                                            \
  }                                                                                                \
  static_assert(internal::hasCommitImpl<ClassName>::value,                                         \
                "Missing `void commitImpl()` function in class " #ClassName);

//===------------------------------------------------------------------------------------------===//
//     NewLine
//===------------------------------------------------------------------------------------------===//

/// @brief NewLine: String accompanied by a new line escape
/// @ingroup codegen
struct NewLine : public Streamable {
  NewLine(std::stringstream& s, int il = 0, bool initialNewLine = false) : Streamable(s, il) {
    if(initialNewLine)
      ss() << "\n";
  }

  void commitImpl() { ss() << "\n"; }
  DAWN_DECL_COMMIT(NewLine, Streamable)
};

//===------------------------------------------------------------------------------------------===//
//     Statement
//===------------------------------------------------------------------------------------------===//

/// @brief Statement: String accompanied by a semicolon and NewLine
/// @ingroup codegen
struct Statement : public NewLine {
  Statement(std::stringstream& s, int il = 0, bool initialNewLine = false)
      : NewLine(s, il, initialNewLine) {}

  void commitImpl() { ss() << ";"; }
  DAWN_DECL_COMMIT(Statement, NewLine)
};

//===------------------------------------------------------------------------------------------===//
//     Type
//===------------------------------------------------------------------------------------------===//

/// @brief Type: Defintion of (templated) type
/// @ingroup codegen
struct Type : public Streamable {
  int hasTemplate = false;

  Type(const std::string& name, std::stringstream& s, int il = 0) : Streamable(s, il) {
    ss() << name;
  }
  Type(Type&&) = default;

  /// @brief Add a template to the Type `type<name>`
  /// @{
  template <typename T>
  Type& addTemplate(const T& name) {
    if(!hasTemplate) {
      hasTemplate = true;
      ss() << "<" << name;
    } else
      ss() << ", " << name;
    return *this;
  }

  Type& addTemplate(Type& type) {
    addTemplate(type.str());
    return *this;
  }
  /// @}

  /// @brief Add a sequence of templates to the Type `type<name1, name2, ..., nameN>`
  /// @{
  template <class Sequence, class StrinfigyFunctor,
            class ValueType = typename std::decay<Sequence>::type::value_type>
  Type& addTemplates(Sequence&& sequence, StrinfigyFunctor&& stringifier) {
    return addTemplate(RangeToString(", ", "", "")(
        sequence, [&](const ValueType& value) -> std::string { return stringifier(value); }));
  }
  template <class Sequence>
  Type& addTemplates(Sequence&& sequence) {
    return addTemplates(std::forward<Sequence>(sequence), RangeToString::DefaultStringifyFunctor());
  }
  /// @}

  void commitImpl() { ss() << (hasTemplate ? ">" : ""); }
  DAWN_DECL_COMMIT(Type, Streamable)
};

//===------------------------------------------------------------------------------------------===//
//     Using
//===------------------------------------------------------------------------------------------===//

/// @brief Definition of a `using T1 = T2`
/// @ingroup codegen
struct Using : public Statement {
  bool RHSDeclared = false;

  /// @brief Add typedef `using name = ...`
  Using(const std::string& name, std::stringstream& s, int il = 0) : Statement(s, il) {
    ss() << "using " << name;
  }

  /// @brief Add a Type to the right-hand side of the using
  Type addType(const std::string& name) {
    ss() << " = ";
    return Type(name, ss());
  }

  void commitImpl() {}
  DAWN_DECL_COMMIT(Using, Statement)
};

//===------------------------------------------------------------------------------------------===//
//     Namespace
//===------------------------------------------------------------------------------------------===//
/// @brief Definition of a `namespace`
/// @ingroup codegen
struct Namespace {
  const std::string name_;
  std::stringstream& s_;

  ~Namespace() {}
  /// @brief Add `namespace`
  Namespace(const std::string& name, std::stringstream& s) : name_(name), s_(s) {
    s_ << "namespace " << name_ << "{" << std::endl;
  }

  void commit() { s_ << "} // namespace " << name_ << std::endl; }
};

//===------------------------------------------------------------------------------------------===//
//     Function
//===------------------------------------------------------------------------------------------===//

/// @brief Definition of a member function
/// @ingroup codegen
struct MemberFunction : public NewLine {
  int IndentLevel = 0;
  int NumArgs = 0;
  int NumInits = 0;
  bool CanHaveInit = false;
  bool CanHaveBody = true;
  bool IsBodyDeclared = false;
  bool AreArgsFinished = false;
  bool IsConst = false;

  /// @brief Declare function with return type (possibly empty) and the name of the function
  MemberFunction(const std::string& returnType, const std::string& name, std::stringstream& s,
                 int il = 0)
      : NewLine(s, il), IndentLevel(il) {
    if(!returnType.empty()) {
      ss() << returnType << " ";
    }
    ss() << name;
  }

  /// @brief Add an argument to the function
  ///
  /// This function can only be called *before* the body is added.
  MemberFunction& addArg(const std::string& name) {
    DAWN_ASSERT(!IsBodyDeclared);
    ss() << (NumArgs++ == 0 ? "(" : ", ") << name;
    return *this;
  }

  /// @brief Add a an initializer (only used by constructors)
  MemberFunction& addInit(const std::string& name) {
    DAWN_ASSERT(CanHaveInit && !IsBodyDeclared);
    finishArgs();
    ss() << (NumInits++ == 0 ? ": " : ", ") << name;
    return *this;
  }

  /// @brief Mark the function as const
  MemberFunction& isConst(bool value) {
    IsConst = value;
    return *this;
  }

  /// @brief Add a sequence of arguments of type `typeName`
  ///
  /// This function can only be called *before* the body is added.
  /// @{
  template <class Sequence, class StrinfigyFunctor,
            class ValueType = typename std::decay<Sequence>::type::value_type>
  MemberFunction& addArgs(const std::string& typeName, Sequence&& sequence,
                          StrinfigyFunctor&& stringifier) {
    DAWN_ASSERT(!IsBodyDeclared);
    ss() << RangeToString(", ", (NumArgs++ == 0 ? "(" : ", "), "")(
        sequence,
        [&](const ValueType& value) -> std::string { return typeName + stringifier(value); });
    return *this;
  }

  template <class Sequence>
  MemberFunction& addArgs(const std::string& typeName, Sequence&& sequence) {
    return addArgs(typeName, std::forward<Sequence>(sequence),
                   RangeToString::DefaultStringifyFunctor());
  }
  /// @}

  /// @brief Start definition of the body
  MemberFunction& startBody() {
    if(!IsBodyDeclared) {
      finishArgs();
      ss() << "{\n";
      IsBodyDeclared = true;
    }
    return *this;
  }

  /// @brief Finish declaration of arguments
  MemberFunction& finishArgs() {
    DAWN_ASSERT(!IsBodyDeclared);
    if(!AreArgsFinished) {
      ss() << (NumArgs == 0 ? "() " : ") ");
      if(IsConst)
        ss() << " const ";
      AreArgsFinished = true;
    }
    return *this;
  }

  /// @brief Add a statement to the function body
  ///
  /// This function can only be called *after* the body was added.
  MemberFunction& addStatement(const std::string& arg) {
    startBody();
    Statement stmt(ss(), IndentLevel + 1);
    stmt << arg;
    return *this;
  }

  /// @brief Add a statement block to the function body (a statement block is sourrounded by
  /// '{ ... }'
  ///
  /// The `blockBodyFun` is executed between the opening '{' and the closing '}'. Note that the
  /// `blockBodyFun` can access the `MemberFunction` and add further statements which will be
  /// correctly indented.
  ///
  /// @b Signature:
  /// @code
  ///   init {
  ///     ...
  ///   }
  /// @endcode
  ///
  /// This function can only be called *after* the body was added.
  template <class BlockBodyFunType>
  MemberFunction& addBlockStatement(const std::string& init, BlockBodyFunType&& blockBodyFun) {
    startBody();
    indentImpl(IndentLevel) << init << " {\n";
    IndentLevel++;
    blockBodyFun();
    IndentLevel--;
    indentImpl(IndentLevel) << "}";
    return *this;
  }

  /// @brief Add a typedef
  Using addTypeDef(const std::string& typeName) {
    indentImpl(IndentLevel + 1);
    return Using(typeName, ss());
  }

  /// @brief Add a comment
  NewLine addComment(const std::string& comment) {
    indentImpl(IndentLevel + 1, true);
    NewLine nl(ss());
    nl << ("// " + comment);
    return nl;
  }

  /// @brief Indent the stream
  void indentStatment() { indentImpl(IndentLevel + 1); }

  /// @brief Get the indent
  int getIndent() const { return DAWN_PRINT_INDENT * (IndentLevel + 1); }

  void commitImpl() {
    // If there is a body, check if we added one or add an empty one `{}`
    if(CanHaveBody) {
      if(!IsBodyDeclared) {
        if(NumInits == 0)
          ss() << ") ";
        ss() << "{}";
      } else
        indentImpl(IndentLevel) << "}";
    }
  }
  DAWN_DECL_COMMIT(MemberFunction, NewLine)
};

//===------------------------------------------------------------------------------------------===//
//     Structure
//===------------------------------------------------------------------------------------------===//

/// @brief Structure: Declaration of a struct or class
/// @ingroup codegen
struct Structure : public Statement {
  enum class ConstructorDefaultKind { Custom, Default, Deleted };

  int IndentLevel = 0;
  std::string StructureName;
  std::string SuffixMember;

  Structure(const char* identifier, const std::string& name, std::stringstream& s,
            const std::string& templateName = "", const std::string& derived = "", int il = 0)
      : Statement(s), IndentLevel(il) {
    StructureName = name;
    if(!templateName.empty())
      indentImpl(IndentLevel) << "template<" << templateName << ">";
    indentImpl(IndentLevel, true) << identifier << " " << StructureName
                                  << (derived.empty() ? std::string() : " : public " + derived)
                                  << " {\n";
  }

  /// @brief Add a suffix member which will be printed between the last '}' and ';'
  ///
  /// @b Signature:
  /// @code
  ///   StructureName name {
  ///
  ///   } SuffixMember;
  /// @endcode
  void addSuffixMember(const std::string& suffixMember) { SuffixMember = suffixMember; }

  /// @brief Change the accessibility of the following members and methods
  void changeAccessibility(const std::string& specifier) {
    NewLine nl(ss(), IndentLevel);
    nl << specifier << ":";
  }

  /// @brief Get the name of the class
  const std::string& getName() const { return StructureName; }

  /// @brief Add a copy constructor
  ///
  /// @b Signature:
  /// @code
  ///   Structure(const Structure&) {...}
  /// @endcode
  MemberFunction
  addCopyConstructor(ConstructorDefaultKind constructorKind = ConstructorDefaultKind::Custom) {
    return addBuiltinConstructor("const " + StructureName + "&", constructorKind);
  }

  /// @brief Add a move constructor
  ///
  /// @b Signature:
  /// @code
  ///   Structure(Structure&&) {...}
  /// @endcode
  MemberFunction
  addMoveConstructor(ConstructorDefaultKind constructorKind = ConstructorDefaultKind::Custom) {
    return addBuiltinConstructor(StructureName + "&&", constructorKind);
  }

  /// @brief Add a constructor
  ///
  /// @b Signature:
  /// @code
  ///   template< templateName >
  ///   Structure(...) {...}
  /// @endcode
  MemberFunction addConstructor(const std::string& templateName = "") {
    MemberFunction memFun = addMemberFunction("", StructureName, templateName);
    memFun.CanHaveInit = true;
    return memFun;
  }

  /// @brief Add a destructor
  ///
  /// @b Signature:
  /// @code
  ///   template< templateName >
  ///   Structure(...) {...}
  /// @endcode
  MemberFunction addDestructor(bool isVirtual) {
    return addMemberFunction("", (isVirtual ? "virtual ~" : "~") + StructureName);
  }

  /// @brief Add a member function (method)
  ///
  /// @b Signature:
  /// @code
  ///   template< templateName >
  ///   returnType funcName(...) {...}
  /// @endcode
  MemberFunction addMemberFunction(const std::string& returnType, const std::string& funcName,
                                   const std::string& templateName = "") {
    newlineImpl();
    if(!templateName.empty())
      indentImpl(IndentLevel + 1) << "template<" << templateName << ">\n";
    return MemberFunction(returnType, funcName, ss(), IndentLevel + 1);
  }

  /// @brief Add a typedef
  ///
  /// @b Signature:
  /// @code
  ///   using typeName = ...
  /// @endcode
  Using addTypeDef(const std::string& typeName) {
    indentImpl(IndentLevel + 1);
    return Using(typeName, ss());
  }

  /// @brief Add a member
  ///
  /// @b Signature:
  /// @code
  ///   typeName memberName;
  /// @endcode
  Statement addMember(const std::string& typeName, const std::string& memberName) {
    Statement member(ss(), IndentLevel + 1);
    member << typeName << " " << memberName;
    return member;
  }

  /// @brief Add a comment
  ///
  /// @b Signature:
  /// @code
  ///   // comment
  /// @endcode
  NewLine addComment(const std::string& comment) {
    indentImpl(IndentLevel + 1, true);
    NewLine nl(ss());
    nl << "// " + comment;
    return nl;
  }

  /// @brief Add a single statement
  ///
  /// @b Signature
  /// @code
  ///   statment;
  /// @endcode
  Statement addStatement(const std::string& statment) {
    Statement stmt(ss(), IndentLevel + 1);
    stmt << statment;
    return stmt;
  }

  /// @brief Add a sub struct
  ///
  /// @b Signature:
  /// @code
  ///   struct name {
  ///    ...
  ///   };
  /// @endcode
  Structure addStruct(const std::string& name, const std::string& templateName = "",
                      const std::string& derived = "") {
    return Structure("struct", name, ss(), templateName, derived, IndentLevel + 1);
  }

  /// @brief Add inline struct member
  ///
  /// @b Signature:
  /// @code
  ///   struct name : public derived {
  ///    ...
  ///   } member;
  /// @endcode
  Structure addStructMember(const std::string& name, const std::string& member,
                            const std::string& derived = "") {
    Structure s("struct", name, ss(), "", derived, IndentLevel + 1);
    s.addSuffixMember(member);
    return s;
  }

  /// @brief Add a sub class
  ///
  /// @b Signature:
  /// @code
  ///   class name {
  ///    ...
  ///   };
  /// @endcode
  Structure addClass(const std::string& name, const std::string& templateName = "") {
    return Structure("class", name, ss(), templateName, "", IndentLevel + 1);
  }

  void commitImpl() {
    indentImpl(IndentLevel) << "}" << (SuffixMember.empty() ? std::string() : (" " + SuffixMember));
  }
  DAWN_DECL_COMMIT(Structure, Statement)

protected:
  MemberFunction
  addBuiltinConstructor(const std::string& arg,
                        ConstructorDefaultKind constructorKind = ConstructorDefaultKind::Custom) {
    newlineImpl();
    MemberFunction ctor("", StructureName, ss(), IndentLevel + 1);
    ctor.addArg(arg);
    switch(constructorKind) {
    case ConstructorDefaultKind::Default:
      ctor.CanHaveBody = false;
      ctor << ") = default;";
      return ctor;
    case ConstructorDefaultKind::Deleted:
      ctor.CanHaveBody = false;
      ctor << ") = delete;";
      return ctor;
    default:
      return ctor;
    }
  }
};

/// @brief Class: Declaration of a class
/// @ingroup codegen
struct Class : public Structure {
  using Structure::Structure;
  Class(const std::string& name, std::stringstream& s, const std::string& templateName = "")
      : Structure("class", name, s, templateName) {}
};

/// @brief Struct: Declaration of a class
/// @ingroup codegen
struct Struct : public Structure {
  using Structure::Structure;
  Struct(const std::string& name, std::stringstream& s, const std::string& templateName = "")
      : Structure("struct", name, s, templateName) {}
};

auto c_gt = []() { return std::string("gridtools::"); };
auto c_dgt = []() { return std::string("gridtools::dawn::"); };
auto c_gt_enum = []() { return std::string("gridtools::enumtype::"); };
auto c_gt_intent = []() { return std::string("gridtools::intent::"); };

} // namespace codegen

} // namespace dawn
