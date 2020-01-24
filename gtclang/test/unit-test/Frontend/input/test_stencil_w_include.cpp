#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil stencil_test {
  storage in, out;
  void Do() {
    vertical_region(k_start, k_end) { out = in + 1; }
  }
};
