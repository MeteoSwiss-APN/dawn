#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
    storage in, out;
    var tmp;

    Do {
      vertical_region(k_end, k_end) {
        tmp = in;
      }
      vertical_region(k_end - 1, k_start) {
        tmp = in * 2;
        out = tmp(k + 1);
      }
    }
};