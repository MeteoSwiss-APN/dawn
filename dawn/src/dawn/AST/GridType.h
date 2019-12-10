#pragma once

#include <iosfwd>

namespace dawn {
namespace ast {
enum class GridType { Structured, Unstructured };
}
std::ostream& operator<<(std::ostream& os, const ast::GridType& gridType);
} // namespace dawn
