#include "dawn/IIR/NodeUpdateType.h"

namespace dawn {
namespace iir {

namespace impl {
bool updateLevel(NodeUpdateType updateType) {
  return static_cast<int>(updateType) < 2 && static_cast<int>(updateType) > -2;
}
bool updateTreeAbove(NodeUpdateType updateType) { return static_cast<int>(updateType) > 0; }
bool updateTreeBelow(NodeUpdateType updateType) { return static_cast<int>(updateType) < 0; }
} // namespace impl
} // namespace iir
} // namespace dawn
