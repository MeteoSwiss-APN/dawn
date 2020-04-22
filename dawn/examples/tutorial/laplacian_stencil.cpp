globals { double dx; };

stencil laplacian_stencil {
  storage_ij out_field;
  storage_ij in_field;
  Do() {
    vertical_region(k_start, k_end) {
      out_field[i, j] = (-4 * in_field[i, j] + in_field[i + 1, j] + in_field[i - 1, j] +
                         in_field[i, j - 1] + in_field[i, j + 1]) /
                        (dx * dx);
    }
  }
};
