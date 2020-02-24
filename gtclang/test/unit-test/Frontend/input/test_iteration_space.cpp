stencil Test1 {
  storage out;
  Do {
    iteration_space(i_start, i_end - 1) { out = 0; }
    iteration_space(j_start, j_end - 2) { out = 0; }
    iteration_space(i_start, i_start + 1, j_start, j_start + 1, k_start, k_end - 1) { out = 0; }
  }
};
