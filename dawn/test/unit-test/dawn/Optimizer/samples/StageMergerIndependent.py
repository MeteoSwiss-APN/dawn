from dusk.script import *

@stencil
def independent(
    eout0: Field[Edge],    
    ein0: Field[Edge],    
    cout0: Field[Cell],    
    cin0: Field[Cell],    
    eout1: Field[Edge],    
    ein1: Field[Edge],    
):
    with levels_upward:        
        eout0 = sum_over(Edge > Cell > Edge, ein0[Edge > Cell > Edge])
        cout0 = sum_over(Cell > Edge > Cell, cin0[Cell > Edge > Cell])
        eout1 = sum_over(Edge > Cell > Edge, ein1[Edge > Cell > Edge])
