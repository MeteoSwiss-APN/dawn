#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
    storage foo;
    var tmp;

    Do {
      vertical_region(k_start, k_end) {
        tmp = 5.0;
        foo = tmp;
      }
    }
};