function plot_mesh(m)
  n = max(size(m));
  hold on;
  for i=1:n
    plot([m(i,1),m(i,3)], [m(i,2),m(i,4)], 'k');
  endfor
endfunction
