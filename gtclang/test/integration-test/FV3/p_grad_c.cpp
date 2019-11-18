#include "gridtools/clang_dsl.hpp"
using namespace gridtools::clang;

globals {
  double dt2;
  bool hydrostatic;
};

stencil p_grad_c {
  storage_ij delpc;
  storage pkc, gz, uc, vc, rdxc, rdyc;
  var wk; // should be ij-only

  Do {
    vertical_region(k_start, k_end) {
      if(hydrostatic) {
        wk = pkc[k + 1] - pkc;
      } else {
        wk = delpc;
      }

      // //    do j=js,je
      // //       do i=is,ie+1
      uc += dt2 * rdxc / (wk[i - 1] + wk) *
            ((gz[i - 1, k + 1] - gz) *
                 (pkc[k + 1] - pkc[i - 1]) +
             (gz[i - 1] - gz[k + 1]) *
                 (pkc[i - 1, k + 1] - pkc));

      // //    do j=js,je+1
      // //       do i=is,ie
      vc += dt2 * rdyc / (wk[j - 1] + wk) *
            ((gz[j - 1, k + 1] - gz) *
                 (pkc[k + 1] - pkc[j - 1]) +
             (gz[j - 1] - gz[k + 1]) *
                 (pkc[j - 1, k + 1] - pkc));
    }
  }
};
