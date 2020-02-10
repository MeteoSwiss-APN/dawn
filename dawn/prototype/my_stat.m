function my_stat(in)
  min_in = min(in(:,3));
  max_in = max(in(:,3));
  fprintf('min: %f, max: %f\n', min_in, max_in);
endfunction
