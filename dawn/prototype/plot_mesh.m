function plot_mesh(m, lw, col)
  if (nargin == 1) 
    lw = 1;
    col = 'k';  
  endif
  
  if (nargin == 2) 
    col = 'k'
  endif
  
  n = max(size(m));
  hold on;
  for i=1:n
    plot([m(i,1),m(i,3)], [m(i,2),m(i,4)], 'LineWidth', lw, col);
  endfor
endfunction
