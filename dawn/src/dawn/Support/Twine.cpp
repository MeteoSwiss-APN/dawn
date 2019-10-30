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

#include "dawn/Support/Twine.h"
#include "dawn/Support/SmallString.h"
#include <iostream>
#include <sstream>

namespace dawn {

std::string Twine::str() const {
  // If we're storing only a std::string, just return it.
  if(lhsKind_ == NodeKind::StdString && rhsKind_ == NodeKind::Empty)
    return *lhs_.stdString;

  // Otherwise, flatten and copy the contents first.
  SmallString<256> Vec;
  return toStringRef(Vec).str();
}

void Twine::toVector(SmallVectorImpl<char>& Out) const {
  std::stringstream OS;
  print(OS);
  std::string outStr = OS.str();
  Out.append(outStr.begin(), outStr.end());
}

StringRef Twine::toNullTerminatedStringRef(SmallVectorImpl<char>& Out) const {
  if(isUnary()) {
    switch(getLHSKind()) {
    case NodeKind::CString:
      // Already null terminated, yay!
      return StringRef(lhs_.cString);
    case NodeKind::StdString: {
      const std::string* str = lhs_.stdString;
      return StringRef(str->c_str(), str->size());
    }
    default:
      break;
    }
  }
  toVector(Out);
  Out.push_back(0);
  Out.pop_back();
  return StringRef(Out.data(), Out.size());
}

void Twine::printOneChild(std::ostream& OS, Child Ptr, NodeKind Kind) const {
  switch(Kind) {
  case Twine::NodeKind::Null:
    break;
  case Twine::NodeKind::Empty:
    break;
  case Twine::NodeKind::Twine:
    Ptr.twine->print(OS);
    break;
  case Twine::NodeKind::CString:
    OS << Ptr.cString;
    break;
  case Twine::NodeKind::StdString:
    OS << *Ptr.stdString;
    break;
  case Twine::NodeKind::StringRef:
    OS << Ptr.stringRef->str();
    break;
  case Twine::NodeKind::SmallString:
    OS << StringRef(Ptr.smallString->data(), Ptr.smallString->size()).str();
    break;
  case Twine::NodeKind::Char:
    OS << Ptr.character;
    break;
  case Twine::NodeKind::DecUI:
    OS << Ptr.decUI;
    break;
  case Twine::NodeKind::DecI:
    OS << Ptr.decI;
    break;
  case Twine::NodeKind::DecUL:
    OS << *Ptr.decUL;
    break;
  case Twine::NodeKind::DecL:
    OS << *Ptr.decL;
    break;
  case Twine::NodeKind::DecULL:
    OS << *Ptr.decULL;
    break;
  case Twine::NodeKind::DecLL:
    OS << *Ptr.decLL;
    break;
  case Twine::NodeKind::UHex:
    OS << *Ptr.uHex;
    break;
  }
}

void Twine::printOneChildRepr(std::ostream& OS, Child Ptr, NodeKind Kind) const {
  switch(Kind) {
  case Twine::NodeKind::Null:
    OS << "null";
    break;
  case Twine::NodeKind::Empty:
    OS << "empty";
    break;
  case Twine::NodeKind::Twine:
    OS << "rope:";
    Ptr.twine->printRepr(OS);
    break;
  case Twine::NodeKind::CString:
    OS << "cstring:\"" << Ptr.cString << "\"";
    break;
  case Twine::NodeKind::StdString:
    OS << "std::string:\"" << Ptr.stdString << "\"";
    break;
  case Twine::NodeKind::StringRef:
    OS << "stringref:\"" << Ptr.stringRef->str() << "\"";
    break;
  case Twine::NodeKind::SmallString:
    OS << "smallstring:\"" << StringRef(Ptr.smallString->data(), Ptr.smallString->size()).str()
       << "\"";
    break;
  case Twine::NodeKind::Char:
    OS << "char:\"" << Ptr.character << "\"";
    break;
  case Twine::NodeKind::DecUI:
    OS << "decUI:\"" << Ptr.decUI << "\"";
    break;
  case Twine::NodeKind::DecI:
    OS << "decI:\"" << Ptr.decI << "\"";
    break;
  case Twine::NodeKind::DecUL:
    OS << "decUL:\"" << *Ptr.decUL << "\"";
    break;
  case Twine::NodeKind::DecL:
    OS << "decL:\"" << *Ptr.decL << "\"";
    break;
  case Twine::NodeKind::DecULL:
    OS << "decULL:\"" << *Ptr.decULL << "\"";
    break;
  case Twine::NodeKind::DecLL:
    OS << "decLL:\"" << *Ptr.decLL << "\"";
    break;
  case Twine::NodeKind::UHex:
    OS << "uhex:\"" << Ptr.uHex << "\"";
    break;
  }
}

void Twine::print(std::ostream& OS) const {
  printOneChild(OS, lhs_, getLHSKind());
  printOneChild(OS, rhs_, getRHSKind());
}

void Twine::printRepr(std::ostream& OS) const {
  OS << "(Twine ";
  printOneChildRepr(OS, lhs_, getLHSKind());
  OS << " ";
  printOneChildRepr(OS, rhs_, getRHSKind());
  OS << ")";
}

void Twine::dump() const { print(std::cerr); }

void Twine::dumpRepr() const { printRepr(std::cerr); }

} // namespace dawn
