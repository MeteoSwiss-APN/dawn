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
#ifndef DAWN_IIR_PRINTER_H
#define DAWN_IIR_PRINTER_H

#include "dawn/Support/Format.h"
#include <sstream>

namespace dawn {
namespace iir {

class StencilInstantiation;

template <typename Arg, typename... Args>
void doPrint(std::stringstream& ss, Arg&& arg, Args&&... args) {
  ss << std::forward<Arg>(arg);
  using expander = int[];
  (void)expander{0, (void(ss << std::forward<Args>(args)), 0)...};
}

struct IIRPrinter {
  int level_;
  std::stringstream& ss_;
  const StencilInstantiation* instantiation_;

  IIRPrinter(const int level, std::stringstream& ss, const StencilInstantiation* instantiation)
      : level_(level), ss_(ss), instantiation_(instantiation) {}
  void close() const;

  IIRPrinter& operator++() {
    ++level_;
    return *this;
  }

  IIRPrinter operator++(int) {
    IIRPrinter p(*this);
    ++(*this);
    return p;
  }

  void dumpHeader(const std::string& name) const;
  template <typename... Str>
  void dump(Str... name) const {
    doPrint(ss_, std::string(level_ * 2 + 1, ' '), name..., '\n');
  }

  const StencilInstantiation* getStencilInstantiation() const { return instantiation_; }
  std::string str() const { return ss_.str(); }
};

} // namespaceiir
} // namespace dawn
#endif
