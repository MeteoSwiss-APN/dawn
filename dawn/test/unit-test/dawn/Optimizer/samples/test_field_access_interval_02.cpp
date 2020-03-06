#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
  storage in;
  void Do() {
    vertical_region(k_start, k_start) { in = 12; }
    vertical_region(k_end, k_end) { in = 10; }
  }
};
