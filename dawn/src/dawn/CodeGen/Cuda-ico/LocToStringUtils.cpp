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

#include "LocToStringUtils.h"

#include "dawn/Support/Unreachable.h"

#include <sstream>

namespace dawn {
namespace codegen {
namespace cudaico {

std::string chainToTableString(const ast::UnstructuredIterationSpace iterSpace) {
  std::stringstream ss;
  for(auto loc : iterSpace.Chain) {
    switch(loc) {
    case dawn::ast::LocationType::Cells:
      ss << "c";
      break;
    case dawn::ast::LocationType::Edges:
      ss << "e";
      break;
    case dawn::ast::LocationType::Vertices:
      ss << "v";
      break;
    }
  }
  if(iterSpace.IncludeCenter) {
    ss << "ITable";
  } else {
    ss << "Table";
  }
  return ss.str();
}

std::string chainToSparseSizeString(const ast::UnstructuredIterationSpace iterSpace) {
  std::stringstream ss;
  for(auto loc : iterSpace.Chain) {
    switch(loc) {
    case dawn::ast::LocationType::Cells:
      ss << "C_";
      break;
    case dawn::ast::LocationType::Edges:
      ss << "E_";
      break;
    case dawn::ast::LocationType::Vertices:
      ss << "V_";
      break;
    }
  }
  if(iterSpace.IncludeCenter) {
    ss << "ISIZE";
  } else {
    ss << "SIZE";
  }
  return ss.str();
}

std::string chainToDenseSizeStringHostMesh(std::vector<dawn::ast::LocationType> locs) {
  std::stringstream ss;
  switch(locs[0]) {
  case dawn::ast::LocationType::Cells:
    ss << "mesh.cells().size()";
    break;
  case dawn::ast::LocationType::Edges:
    ss << "mesh.edges().size()";
    break;
  case dawn::ast::LocationType::Vertices:
    ss << "mesh.nodes().size()";
    break;
  }
  return ss.str();
}

std::string chainToVectorString(std::vector<dawn::ast::LocationType> locs) {
  std::stringstream ss;
  ss << "{";
  bool first = true;
  for(auto loc : locs) {
    if(!first) {
      ss << ", ";
    }
    switch(loc) {
    case dawn::ast::LocationType::Cells:
      ss << "dawn::LocationType::Cells";
      break;
    case dawn::ast::LocationType::Edges:
      ss << "dawn::LocationType::Edges";
      break;
    case dawn::ast::LocationType::Vertices:
      ss << "dawn::LocationType::Vertices";
      break;
    }
    first = false;
  }
  ss << "}";
  return ss.str();
}
std::string locToDenseSizeStringGpuMesh(dawn::ast::LocationType loc, std::optional<Padding> padding,
                                        bool addParens) {
  std::string ret;
  switch(loc) {
  case dawn::ast::LocationType::Cells:
    ret = padding.has_value() ? "NumCells + " + std::to_string(padding->Cells()) : "NumCells";
    break;
  case dawn::ast::LocationType::Edges:
    ret = padding.has_value() ? "NumEdges + " + std::to_string(padding->Edges()) : "NumEdges";
    break;
  case dawn::ast::LocationType::Vertices:
    ret = padding.has_value() ? "NumVertices + " + std::to_string(padding->Vertices())
                              : "NumVertices";
    break;
  default:
    dawn_unreachable("");
  }
  if(addParens) {
    return "(" + ret + ")";
  } else {
    return ret;
  }
}
std::string locToStrideString(dawn::ast::LocationType loc) {
  switch(loc) {
  case dawn::ast::LocationType::Cells:
    return "CellStride";
    break;
  case dawn::ast::LocationType::Edges:
    return "EdgeStride";
    break;
  case dawn::ast::LocationType::Vertices:
    return "VertexStride";
    break;
  default:
    dawn_unreachable("");
  }  
}
std::string locToDenseTypeString(dawn::ast::LocationType loc) {
  switch(loc) {
  case dawn::ast::LocationType::Cells:
    return "dawn::cell_field_t<LibTag, ::dawn::float_type>";
    break;
  case dawn::ast::LocationType::Edges:
    return "dawn::edge_field_t<LibTag, ::dawn::float_type>";
    break;
  case dawn::ast::LocationType::Vertices:
    return "dawn::vertex_field_t<LibTag, ::dawn::float_type>";
    break;
  default:
    dawn_unreachable("");
  }
}
std::string locToSparseTypeString(dawn::ast::LocationType loc) {
  switch(loc) {
  case dawn::ast::LocationType::Cells:
    return "dawn::sparse_cell_field_t<LibTag, ::dawn::float_type>";
    break;
  case dawn::ast::LocationType::Edges:
    return "dawn::sparse_edge_field_t<LibTag, ::dawn::float_type>";
    break;
  case dawn::ast::LocationType::Vertices:
    return "dawn::sparse_vertex_field_t<LibTag, ::dawn::float_type>";
    break;
  default:
    dawn_unreachable("");
  }
}

std::string locToStringPlural(dawn::ast::LocationType loc) {
  switch(loc) {
  case dawn::ast::LocationType::Cells:
    return "Cells";
    break;
  case dawn::ast::LocationType::Edges:
    return "Edges";
    break;
  case dawn::ast::LocationType::Vertices:
    return "Vertices";
    break;
  default:
    dawn_unreachable("");
  }
}

} // namespace cudaico
} // namespace codegen
} // namespace dawn
