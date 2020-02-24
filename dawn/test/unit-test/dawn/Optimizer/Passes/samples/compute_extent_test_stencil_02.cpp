#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
  storage in, out;
  void Do() {
    vertical_region(k_start, k_end) {
      mid = in[i - 1];
      out = in[j + 1];
    }
  }
};
