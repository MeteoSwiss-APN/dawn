#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

#pragma gtclang merge_stages
stencil interval_partition_test {
  storage in, out, out2, out3, out4;
  var tmp;
  void Do() {
    vertical_region(k_start, k_end) { out = in + 1; }
    vertical_region(k_start, k_start + 2) { out2 = in + 2; }
    vertical_region(k_end - 3, k_end) { out3 = in + 3; }
    vertical_region(k_start + 1, k_end - 2) { out4 = in + 4; }
  }
};
