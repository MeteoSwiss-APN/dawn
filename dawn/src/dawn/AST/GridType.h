#pragma once

#include "dawn/Support/Assert.h"
#include <iosfwd>

namespace dawn {
namespace iir {
class IIR;
}
} // namespace dawn

namespace dawn {
struct SIR;
} // namespace dawn

namespace dawn {

namespace ast {
enum class GridType { Cartesian, Triangular };

class GlobalGridType {
  friend class dawn::iir::IIR;
  friend struct dawn::SIR;

public:
  static auto& instance() {
    static GlobalGridType gridType;
    return gridType;
  }

  bool valueSet() const { return GlobalGridType::instance().valueSet_; }
  GridType getGridType() const { return GlobalGridType::instance().type_; }
  GlobalGridType(const GlobalGridType&) = delete;
  GlobalGridType& operator=(const GlobalGridType&) = delete;
  GlobalGridType(GlobalGridType&&) = delete;
  GlobalGridType& operator=(GlobalGridType&&) = delete;

private:
  bool valueSet_ = false;
  GridType type_;
  void setGridType(GridType type) {
    GlobalGridType::instance().valueSet_ = true;
    GlobalGridType::instance().type_ = type;
  }
  GlobalGridType() {}
};
} // namespace ast

std::ostream& operator<<(std::ostream& os, const ast::GridType& gridType);
} // namespace dawn
