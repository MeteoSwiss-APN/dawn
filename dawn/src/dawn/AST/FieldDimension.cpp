#include "FieldDimension.h"

#include "dawn/Support/Casting.h"
#include "dawn/Support/Format.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {
namespace ast {

UnstructuredFieldDimension::UnstructuredFieldDimension(ast::NeighborChain neighborChain,
                                                       bool includeCenter)
    : iterSpace_(std::move(neighborChain), includeCenter) {}

const ast::NeighborChain& UnstructuredFieldDimension::getNeighborChain() const {
  DAWN_ASSERT(isSparse());
  return iterSpace_.Chain;
}

std::string UnstructuredFieldDimension::toString() const {
  auto getLocationTypeString = [](const ast::LocationType type) {
    switch(type) {
    case ast::LocationType::Cells:
      return std::string("cell");
      break;
    case ast::LocationType::Vertices:
      return std::string("vertex");
      break;
    case ast::LocationType::Edges:
      return std::string("edge");
      break;
    default:
      dawn_unreachable("unexpected type");
    }
  };

  std::string output = "", separator = "";
  for(const auto elem : iterSpace_.Chain) {
    if(iterSpace_.IncludeCenter && separator == "") {
      output += separator + "[" + getLocationTypeString(elem) + "]";
    } else {
      output += separator + getLocationTypeString(elem);
    }
    separator = "->";
  }
  return output;
}

ast::GridType HorizontalFieldDimension::getType() const {
  if(dimension_isa<CartesianFieldDimension>(*this)) {
    return ast::GridType::Cartesian;
  } else {
    return ast::GridType::Unstructured;
  }
}

std::string FieldDimensions::toString() const {
  if(dimension_isa<CartesianFieldDimension>(getHorizontalFieldDimension())) {
    const auto& cartesianDimensions =
    dimension_cast<CartesianFieldDimension const&>(getHorizontalFieldDimension());
    return format("[%i,%i,%i]", cartesianDimensions.I(), cartesianDimensions.J(), K());

  } else if(dimension_isa<UnstructuredFieldDimension>(getHorizontalFieldDimension())) {
    const auto& unstructuredDimension =
    dimension_cast<UnstructuredFieldDimension const&>(getHorizontalFieldDimension());
    return format("[%s,%i]", unstructuredDimension.toString(), K());

  } else {
    dawn_unreachable("Invalid horizontal field dimension");
  }
}

int FieldDimensions::numSpatialDimensions() const {
  if(!horizontalFieldDimension_) {
    return 1;
  }
  if(dimension_isa<CartesianFieldDimension>(getHorizontalFieldDimension())) {
    const auto& cartesianDimensions =
    dimension_cast<CartesianFieldDimension const&>(getHorizontalFieldDimension());
    return int(cartesianDimensions.I()) + int(cartesianDimensions.J()) + int(K());
  } else if(dimension_isa<UnstructuredFieldDimension>(getHorizontalFieldDimension())) {
    return 2 + int(K());
  } else {
    dawn_unreachable("Invalid horizontal field dimension");
  }
}

int FieldDimensions::rank() const {
  const int spatialDims = numSpatialDimensions();
  if(isVertical()) {
    return 1;
  }
  int rank;
  if(dimension_isa<UnstructuredFieldDimension>(getHorizontalFieldDimension())) {
    rank = spatialDims > 1 ? spatialDims - 1 // The horizontal counts as 1 dimension (dense)
                           : spatialDims;
    // Need to account for sparse dimension, if present
    if(dimension_cast<UnstructuredFieldDimension const&>(getHorizontalFieldDimension())
        .isSparse()) {
      ++rank;
    }
  } else { // Cartesian
    rank = spatialDims;
  }
  return rank;
}

} // namespace ast
} // namespace dawn
