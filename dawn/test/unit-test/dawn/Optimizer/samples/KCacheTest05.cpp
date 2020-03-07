#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test1 {
    storage in, out;

    Do {
      vertical_region(k_start, k_end) {
        out = in + in(k + 1);
      }
    }
};

stencil Test2 {
    storage in, out;

    Do {
      vertical_region(k_start, k_end) {
        out += in + in(k + 1);
      }
    }
};