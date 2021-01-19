namespace dawn {

namespace ast {

/// @brief Attributes attached to various SIR objects which allow to change the behavior on per
/// stencil basis
/// @ingroup sir
class Attr {
  unsigned attrBits_;

public:
  Attr() : attrBits_(0) {}

  /// @brief Attribute bit-mask
  enum class Kind : unsigned {
    NoCodeGen = 1 << 0,        ///< Don't generate code for this stencil
    MergeStages = 1 << 1,      ///< Merge the Stages of this stencil
    MergeDoMethods = 1 << 2,   ///< Merge the Do-Methods of this stencil
    MergeTemporaries = 1 << 3, ///< Merge the temporaries of this stencil
    UseKCaches = 1 << 4        ///< Use K-Caches
  };

  /// @brief Check if `attr` bit is set
  bool has(Kind attr) const { return (attrBits_ >> static_cast<unsigned>(attr)) & 1; }

  /// @brief Check if any of the `attrs` bits is set
  /// @{
  bool hasOneOf(Kind attr1, Kind attr2) const { return has(attr1) || has(attr2); }

  template <typename... AttrTypes>
  bool hasOneOf(Kind attr1, Kind attr2, AttrTypes... attrs) const {
    return has(attr1) || hasOneOf(attr2, attrs...);
  }
  /// @}

  ///@brief getting the Bits
  unsigned getBits() const { return attrBits_; }
  /// @brief Set `attr`bit
  void set(Kind attr) { attrBits_ |= 1 << static_cast<unsigned>(attr); }

  /// @brief Unset `attr` bit
  void unset(Kind attr) { attrBits_ &= ~(1 << static_cast<unsigned>(attr)); }

  /// @brief Clear all attributes
  void clear() { attrBits_ = 0; }

  bool operator==(const Attr& rhs) const { return getBits() == rhs.getBits(); }
  bool operator!=(const Attr& rhs) const { return getBits() != rhs.getBits(); }
};

} // namespace ast
} // namespace dawn
