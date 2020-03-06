#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
    storage in, out;
    var tmp;

    Do {
      vertical_region(k_start, k_end) {
        tmp = in;
      }
      vertical_region(k_start, k_end) {
        out = tmp(i + 1);
      }
    }
};