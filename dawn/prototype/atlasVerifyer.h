#pragma once
#include "atlas_interface.hpp"
#include <type_traits>

// quick verifiyer class, very close in behaviour to gridtools::clang::verifyer
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
  struct ComparisonResult {
    bool outcome;
    value_type error;
    operator bool() { return outcome; }
  };

  template <typename value_type>
  ComparisonResult<value_type> compare_below_threshold(value_type expected, value_type actual,
                                                       value_type precision) const {
    ComparisonResult<value_type> result;
    result.outcome = false;
    if(precision == 0) {
      result.outcome = expected == actual;
      result.error = 0;
    } else if(std::fabs(expected) < 1e-3 && std::fabs(actual) < 1e-3) {
      if(std::fabs(expected - actual) < precision) {
        result.outcome = true;
      }
      result.error = std::fabs(expected - actual);
    } else {
      if(std::fabs((expected - actual) / (precision * expected)) < 1.0) {
        result.outcome = true;
      }
      result.error = std::fabs((expected - actual) / expected);
    }
    return result;
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
        Value valueRhs = rhs(i, k);
        ComparisonResult<Value> comparisonResult =
            compare_below_threshold(valueLhs, valueRhs, Value(precision_));
        if(!comparisonResult) {
          if(--max_erros >= 0) {
            std::cerr << "( " << i << " " << k << " ) : "
                      << "  error: " << comparisonResult.error << std::endl;
          }
          verified = false;
        }
      }
    }

    return verified;
  }
};