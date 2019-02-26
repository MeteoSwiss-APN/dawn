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
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/StringUtil.h"
#include <iostream>
#include <utility>

namespace dawn {
namespace iir {

namespace {

template <class MapType, class AccessIDToStringFunctionType>
std::string toStringImpl(AccessIDToStringFunctionType&& accessIDToStringFunction,
                         std::size_t initialIndent, const MapType& writeAccessMap,
                         const MapType& readAccessMap) {
  std::stringstream ss;
  std::string indent(initialIndent, ' ');

  auto printMap = [&](const MapType& map) {
    for(auto it = map.begin(), end = map.end(); it != end; ++it)
      ss << indent << "  " << accessIDToStringFunction(it->first) << " : " << it->second << "\n";
  };

  ss << indent << "Write Accesses:\n";
  printMap(writeAccessMap);

  ss << indent << "Read Accesses:\n";
  printMap(readAccessMap);

  return ss.str();
}

template <class MapType, class AccessIDToStringFunctionType>
std::string reportAccessesImpl(AccessIDToStringFunctionType&& accessIDToStringFunction,
                               const MapType& writeAccessMap, const MapType& readAccessMap) {
  std::stringstream ss;

  auto printMap = [&](const MapType& map, const char* intent) {
    for(auto it = map.begin(), end = map.end(); it != end; ++it) {
      ss << (it != map.begin() ? " " : "") << intent << ":";
      int AccessID = it->first;
      ss << accessIDToStringFunction(AccessID);
      ss << ":<";
      const auto& extents = it->second.getExtents();
      for(std::size_t i = 0; i < extents.size(); ++i)
        ss << extents[i].Minus << "," << extents[i].Plus << (i != extents.size() - 1 ? "," : ">");
    }
  };

  printMap(writeAccessMap, "W");
  ss << " ";
  printMap(readAccessMap, "R");
  return ss.str();
}

} // anonymous namespace

void Accesses::mergeReadOffset(int AccessID, const Array3i& offset) {
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

void Accesses::mergeWriteOffset(int AccessID, const Array3i& offset) {
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
    it->second.add(extent);
  else
    readAccesses_.emplace(AccessID, extent);
}

void Accesses::addWriteExtent(int AccessID, const Extents& extent) {
  auto it = writeAccesses_.find(AccessID);
  if(it != writeAccesses_.end())
    it->second.add(extent);
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

std::string Accesses::reportAccesses(const StencilInstantiation* instantiation) const {
  return reportAccessesImpl(
      [&instantiation](int AccessID) { return instantiation->getNameFromAccessID(AccessID); },
      writeAccesses_, readAccesses_);
}

std::string Accesses::reportAccesses(const StencilFunctionInstantiation* stencilFunc) const {
  return reportAccessesImpl(
      [&stencilFunc](int AccessID) { return stencilFunc->getNameFromAccessID(AccessID); },
      writeAccesses_, readAccesses_);
}

std::string Accesses::toString(const StencilInstantiation* instantiation,
                               std::size_t initialIndent) const {
  return toStringImpl(
      [&instantiation](int AccessID) { return instantiation->getNameFromAccessID(AccessID); },
      initialIndent, writeAccesses_, readAccesses_);
}

std::string Accesses::toString(const StencilFunctionInstantiation* stencilFunc,
                               std::size_t initialIndent) const {
  return toStringImpl(
      [&stencilFunc](int AccessID) { return stencilFunc->getNameFromAccessID(AccessID); },
      initialIndent, writeAccesses_, readAccesses_);
}

} // namespace iir
} // namespace dawn
