#pragma once
#include "atlas_interface.hpp"
#include <type_traits>

// quick verifiyer class, very close in behaviour to gridtools::dawn::verifyer
class atlasVerifyer {
private:
  bool use_default_precision_ = false;
  double precision_;

  template <typename Value>
  void setDefaultPrecision() {
    if(std::is_same<Value, double>::value) {
      precision_ = 1e-10;
    } else if(std::is_same<Value, float>::value) {
      precision_ = 1e-6;
    } else if(std::is_same<Value, int>::value) {
      precision_ = 0;
    } else {
      assert(false);
    }
  }

  template <typename value_type>
  bool compare_below_threashold(value_type expected, value_type actual,
                                value_type precision) const {
    if(precision == 0) {
      return expected == actual;
    }
    if(std::fabs(expected) < 1e-3 && std::fabs(actual) < 1e-3) {
      if(std::fabs(expected - actual) < precision) {
        return true;
      }
    } else {
      if(std::fabs((expected - actual) / (precision * expected)) < 1.0) {
        return true;
      }
    }
    return false;
  }

public:
  atlasVerifyer() : use_default_precision_(true) {}
  atlasVerifyer(double precision) : use_default_precision_(false), precision_(precision) {}

  template <typename Value, int RANK, atlas::array::Intent AccessMode>
  bool compareArrayView(const atlas::array::ArrayView<Value, RANK, AccessMode>& lhs,
                        const atlas::array::ArrayView<Value, RANK, AccessMode>& rhs,
                        int max_erros = 10) {
    // it's far from trivial to compare two _general_ array views, since they can have arbitrary
    // rank, with dimensions unknown at compile time since we currently only consider scalar fields
    // in 3 spatial dimensions I decided to force this to two
    //      NOTE: f(i,k) is rank 2, with i being the element index on a level, and k the given level
    //      NOTE: even atlas seems to limit the used dimensions to rank 4, which would be a 2D
    //      tensor on each element at each level
    //            (2 additional ranks to address the tensor)
    static_assert(RANK == 2, "");

    if(use_default_precision_) {
      setDefaultPrecision<Value>();
    }

    // first compare geometry
    if((lhs.shape(0) != rhs.shape(0)) || (lhs.shape(1) != rhs.shape(1))) {
      return false;
    }

    // then the values
    bool verified = true;
    for(int i = 0; i < lhs.shape(0); i++) {
      for(int k = 0; k < lhs.shape(1); k++) {
        Value valueLhs = lhs(i, k);
        Value valueRhs = lhs(i, k);
        if(!compare_below_threashold(valueLhs, valueRhs, Value(precision_))) {
          if(--max_erros >= 0) {
            std::cerr << "( " << i << " " << k << " ) : "
                      << "  error: " << std::fabs((valueRhs - valueLhs) / (valueRhs)) << std::endl;
          }
          verified = false;
        }
      }
    }

    return verified;
  }
};
