#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
  storage in;
  void Do() {
    vertical_region(k_start, k_start + 2) { in = 12; }
  }
};
