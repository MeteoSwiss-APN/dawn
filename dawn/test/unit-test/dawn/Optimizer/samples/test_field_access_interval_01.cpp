#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

interval k_flat = k_start + 12;

stencil Test {
  storage in, out;
  void Do() {
    vertical_region(k_start, k_flat) { in = 12; }
    vertical_region(k_start, k_end) { out = 10; }
  }
};
