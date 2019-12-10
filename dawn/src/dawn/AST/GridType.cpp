#include "dawn/AST/GridType.h"
#include <iostream>

namespace dawn {

std::ostream& operator<<(std::ostream& os, const ast::GridType& gridType) {
  switch(gridType) {
  case ast::GridType::Structured:
    os << "structured";
    break;
  case ast::GridType::Unstructured:
    os << "unstructured";
    break;
  }
  return os;
}

} // namespace dawn
