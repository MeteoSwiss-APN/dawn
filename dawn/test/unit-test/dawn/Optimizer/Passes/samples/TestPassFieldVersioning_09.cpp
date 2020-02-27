#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
  storage field, tmp;

  Do {
    vertical_region(k_start, k_end) {
      tmp = field(i + 1);
      field = tmp;

      tmp = field(i + 1);
      field = tmp;
    }
  }
};