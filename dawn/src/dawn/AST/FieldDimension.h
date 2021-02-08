#pragma once

#include "GridType.h"
#include "IterationSpace.h"
#include "Tags.h"

#include <memory>
#include <optional>

namespace dawn {
namespace ast {

class FieldDimensionImpl {
public:
  std::unique_ptr<FieldDimensionImpl> clone() const { return cloneImpl(); }
  virtual ~FieldDimensionImpl() = default;
  bool operator==(const FieldDimensionImpl& other) { return equalityImpl(other); }

private:
  virtual std::unique_ptr<FieldDimensionImpl> cloneImpl() const = 0;
  virtual bool equalityImpl(const FieldDimensionImpl& other) const = 0;
};

/// @brief In the cartesian case, the horizontal dimension is an IJ-mask describing if the
/// field is allowed to have extents in I and/or J: [1,0] is a storage_i and cannot be
/// accessed with field[j+1]
///
/// @ingroup sir
class CartesianFieldDimension : public FieldDimensionImpl {
  const std::array<bool, 2> mask_;
  std::unique_ptr<FieldDimensionImpl> cloneImpl() const override {
    return std::make_unique<CartesianFieldDimension>(mask_);
  }
  virtual bool equalityImpl(const FieldDimensionImpl& other) const override {
    auto const& otherCartesian = dynamic_cast<CartesianFieldDimension const&>(other);
    return otherCartesian.I() == I() && otherCartesian.J() == J();
  }

public:
  bool I() const { return mask_[0]; }
  bool J() const { return mask_[1]; }
  explicit CartesianFieldDimension(std::array<bool, 2> mask) : mask_(mask) {}
};

/// @brief In the unstructured case, the horizontal dimension can be either dense or sparse.
/// A field on dense corresponds to 1 value for each location of type defined by the "dense location
/// type". A field on sparse corresponds to as many values as indirect neighbors defined through a
/// neighbor chain.
///
/// Construct with a neighbor chain. If it is of size = 1, then the dimension is dense (with
/// location type = single element of chain), otherwise sparse (with dense location type being the
/// first element of the chain).
///
/// @ingroup sir
class UnstructuredFieldDimension : public FieldDimensionImpl {
  std::unique_ptr<FieldDimensionImpl> cloneImpl() const override {
    return std::make_unique<UnstructuredFieldDimension>(iterSpace_.Chain, iterSpace_.IncludeCenter);
  }
  virtual bool equalityImpl(const FieldDimensionImpl& other) const override {
    auto const& otherUnstructured = dynamic_cast<UnstructuredFieldDimension const&>(other);
    return iterSpace_ == otherUnstructured.iterSpace_;
  }
  const ast::UnstructuredIterationSpace iterSpace_;

public:
  explicit UnstructuredFieldDimension(ast::NeighborChain neighborChain, bool includeCenter = false);
  /// @brief Returns the neighbor chain encoding the sparse part (isSparse() must be true!).
  const ast::NeighborChain& getNeighborChain() const;
  /// @brief Returns the dense location (always present)
  ast::LocationType getDenseLocationType() const { return iterSpace_.Chain[0]; }
  /// @brief Returns the last sparse location type if there is a sparse part, otherwise returns the
  /// dense part.
  ast::LocationType getLastSparseLocationType() const { return iterSpace_.Chain.back(); }
  bool isSparse() const { return iterSpace_.Chain.size() > 1; }
  bool isDense() const { return !isSparse(); }
  bool getIncludeCenter() const { return iterSpace_.IncludeCenter; }
  ast::UnstructuredIterationSpace getIterSpace() const { return iterSpace_; }
  std::string toString() const;
};

class HorizontalFieldDimension {
  std::unique_ptr<FieldDimensionImpl> impl_;

public:
  // Construct a Cartesian horizontal field dimension with specified ij mask.
  HorizontalFieldDimension(dawn::ast::cartesian_, std::array<bool, 2> mask)
      : impl_(std::make_unique<CartesianFieldDimension>(mask)) {}

  // Construct a Unstructured horizontal field sparse dimension with specified neighbor chain
  // (sparse part). Dense part is the first element of the chain.
  HorizontalFieldDimension(dawn::ast::unstructured_, ast::NeighborChain neighborChain,
                           bool includeCenter = false)
      : impl_(std::make_unique<UnstructuredFieldDimension>(neighborChain, includeCenter)) {}
  // Construct a Unstructured horizontal field dense dimension with specified (dense) location type.
  HorizontalFieldDimension(dawn::ast::unstructured_, ast::LocationType locationType,
                           bool includeCenter = false)
      : impl_(std::make_unique<UnstructuredFieldDimension>(ast::NeighborChain{locationType},
                                                           includeCenter)) {}

  HorizontalFieldDimension(const HorizontalFieldDimension& other) { *this = other; }
  HorizontalFieldDimension(HorizontalFieldDimension&& other) { *this = other; };

  HorizontalFieldDimension& operator=(const HorizontalFieldDimension& other) {
    impl_ = other.impl_->clone();
    return *this;
  }
  HorizontalFieldDimension& operator=(HorizontalFieldDimension&& other) {
    impl_ = std::move(other.impl_);
    return *this;
  }

  bool operator==(const HorizontalFieldDimension& other) const { return *impl_ == *other.impl_; }

  ast::GridType getType() const;

  template <typename T>
  friend T dimension_cast(HorizontalFieldDimension const& dimension);
  template <typename T>
  friend bool dimension_isa(HorizontalFieldDimension const& dimension);
};

template <typename T>
T dimension_cast(HorizontalFieldDimension const& dimension) {
  using PlainT = std::remove_reference_t<T>;
  static_assert(std::is_base_of_v<FieldDimensionImpl, PlainT>,
                "Can only be casted to a valid field dimension implementation");
  static_assert(std::is_const_v<PlainT>, "Can only be casted to const");
  return *dynamic_cast<std::add_pointer_t<T>>(dimension.impl_.get());
}

template <typename T>
bool dimension_isa(HorizontalFieldDimension const& dimension) {
  using PlainT = std::remove_pointer_t<std::remove_reference_t<T>>;
  static_assert(std::is_base_of_v<FieldDimensionImpl, PlainT>,
                "Can only be casted to a valid field dimension implementation");
  return static_cast<bool>(dynamic_cast<PlainT*>(dimension.impl_.get()));
}

class FieldDimensions {
public:
  FieldDimensions(HorizontalFieldDimension&& horizontalFieldDimension, bool maskK)
      : horizontalFieldDimension_(horizontalFieldDimension), maskK_(maskK) {
    if(!maskK && dimension_isa<CartesianFieldDimension>(*horizontalFieldDimension_)) {
      auto cartDims = dimension_cast<const CartesianFieldDimension&>(*horizontalFieldDimension_);
      DAWN_ASSERT_MSG(cartDims.I() || cartDims.J(),
                      "a field cant' have all dimensions masked out!");
    }
  }
  FieldDimensions(bool maskK) : maskK_(maskK) {
    DAWN_ASSERT_MSG(
        maskK_, "a field can't have null horizontal dimensions as well as masked out k dimension!");
  }
  FieldDimensions(const FieldDimensions&) = default;
  FieldDimensions(FieldDimensions&&) = default;

  FieldDimensions& operator=(const FieldDimensions&) = default;
  FieldDimensions& operator=(FieldDimensions&&) = default;

  bool operator==(const FieldDimensions& other) const {
    return (maskK_ == other.maskK_ && horizontalFieldDimension_ == other.horizontalFieldDimension_);
  }

  bool K() const { return maskK_; }
  const HorizontalFieldDimension& getHorizontalFieldDimension() const {
    DAWN_ASSERT_MSG(!isVertical(), "attempted to get horizontal dimension of a vertical field!");
    return horizontalFieldDimension_.value();
  }
  bool isVertical() const { return !horizontalFieldDimension_.has_value(); }
  std::string toString() const;

  // returns number of dimensions (1-3)
  int numSpatialDimensions() const;

  // returns the rank of the corresponding storage (multidimensional array)
  int rank() const;

private:
  std::optional<HorizontalFieldDimension> horizontalFieldDimension_;
  bool maskK_;
};

} // namespace ast
} // namespace dawn
