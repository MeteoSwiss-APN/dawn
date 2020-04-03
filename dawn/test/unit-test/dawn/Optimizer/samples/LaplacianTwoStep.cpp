#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;
// halo size = 1
stencil Test {
  storage in, out;
  void Do() {
    vertical_region(k_start, k_end) {
      tmp= lap(in);
      out = lap(tmp);
    }
  }
};

