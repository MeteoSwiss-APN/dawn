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

#include "dawn/IIR/Accesses.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/IIR/StencilMetaInformation.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/StringUtil.h"
#include <sstream>
#include <utility>

namespace dawn {
namespace iir {

namespace {

template <class MapType, class AccessIDToStringFunctionType>
std::string reportAccessesImpl(AccessIDToStringFunctionType&& accessIDToStringFunction,
                               const MapType& writeAccessMap, const MapType& readAccessMap) {
  std::stringstream ss;

  auto printMap = [&](const MapType& map, const char* intent) {
    for(auto it = map.begin(), end = map.end(); it != end; ++it) {
      ss << (it != map.begin() ? " " : "") << intent << ":";
      auto const& [AccessID, extent] = *it;
      ss << accessIDToStringFunction(AccessID);
      ss << ":<" << to_string(extent) << ">";
    }
  };

  printMap(writeAccessMap, "W");
  ss << " ";
  printMap(readAccessMap, "R");
  return ss.str();
}

} // anonymous namespace

bool Accesses::operator==(const Accesses& rhs) const {
  return readAccesses_ == rhs.readAccesses_ && writeAccesses_ == rhs.writeAccesses_;
}

bool Accesses::operator!=(const Accesses& rhs) const { return !(*this == rhs); }

void Accesses::mergeReadOffset(int AccessID, const ast::Offsets& offset) {
  auto it = readAccesses_.find(AccessID);
  if(it != readAccesses_.end()) {
    it->second.merge(offset);
  } else {
    readAccesses_.emplace(AccessID, Extents(offset));
  }
}

void Accesses::mergeReadExtent(int AccessID, const Extents& extent) {
  auto it = readAccesses_.find(AccessID);
  if(it != readAccesses_.end())
    it->second.merge(extent);
  else
    readAccesses_.emplace(AccessID, extent);
}

void Accesses::mergeWriteOffset(int AccessID, const ast::Offsets& offset) {
  auto it = writeAccesses_.find(AccessID);
  if(it != writeAccesses_.end())
    it->second.merge(offset);
  else
    writeAccesses_.emplace(AccessID, Extents(offset));
}

void Accesses::mergeWriteExtent(int AccessID, const Extents& extent) {
  auto it = writeAccesses_.find(AccessID);
  if(it != writeAccesses_.end())
    it->second.merge(extent);
  else
    writeAccesses_.emplace(AccessID, extent);
}

void Accesses::addReadExtent(int AccessID, const Extents& extent) {
  auto it = readAccesses_.find(AccessID);
  if(it != readAccesses_.end())
    it->second += extent;
  else
    readAccesses_.emplace(AccessID, extent);
}

void Accesses::addWriteExtent(int AccessID, const Extents& extent) {
  auto it = writeAccesses_.find(AccessID);
  if(it != writeAccesses_.end())
    it->second += extent;
  else
    writeAccesses_.emplace(AccessID, extent);
}

bool Accesses::hasReadAccess(int accessID) const { return readAccesses_.count(accessID); }

bool Accesses::hasWriteAccess(int accessID) const { return writeAccesses_.count(accessID); }

bool Accesses::hasAccess(int accessID) const {
  return hasReadAccess(accessID) || hasWriteAccess(accessID);
}

Extents const& Accesses::getReadAccess(int AccessID) const {
  DAWN_ASSERT(readAccesses_.count(AccessID));
  return readAccesses_.at(AccessID);
}

const Extents& Accesses::getWriteAccess(int AccessID) const {
  DAWN_ASSERT(writeAccesses_.count(AccessID));
  return writeAccesses_.at(AccessID);
}

// Yes.. this is an abomination down here. One would need to factor the common fanctionality from
// StencilInstantiation and StencilFunctionInstantiation into a common base class to clean this
// mess up.

std::string Accesses::reportAccesses(const StencilMetaInformation& metadata) const {
  return reportAccessesImpl([&](int AccessID) { return metadata.getNameFromAccessID(AccessID); },
                            writeAccesses_, readAccesses_);
}

std::string Accesses::reportAccesses(const StencilFunctionInstantiation* stencilFunc) const {
  return reportAccessesImpl(
      [&stencilFunc](int AccessID) { return stencilFunc->getNameFromAccessID(AccessID); },
      writeAccesses_, readAccesses_);
}

std::string Accesses::toString(std::function<std::string(int)>&& accessIDToStringFunction,
                               std::size_t initialIndent) const {
  std::stringstream ss;
  std::string indent(initialIndent, ' ');

  auto printMap = [&](const std::unordered_map<int, Extents>& map) {
    for(auto const& [accessID, access] : map)
      ss << indent << "  " << accessIDToStringFunction(accessID) << " : " << access << "\n";
  };

  ss << indent << "Write Accesses:\n";
  printMap(writeAccesses_);

  ss << indent << "Read Accesses:\n";
  printMap(readAccesses_);

  return ss.str();
}

} // namespace iir
} // namespace dawn
