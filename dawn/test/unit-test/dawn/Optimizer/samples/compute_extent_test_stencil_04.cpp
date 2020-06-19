#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
  storage in, out, mid, mid2;
  void Do() {
    vertical_region(k_start, k_end) {
      mid = in;
      mid2 = mid[i + 1];
      out = mid2[j - 1];
    }
  }
};
