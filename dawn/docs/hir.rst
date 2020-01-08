.. _SIR:

High-Level Intermediate Representation
########################################

Developing a single DSL that is able to support all numerical methods and computational patterns present in GFD models present a serious challenge due to the wide domain it needs to cover. In this project, we accept the reality that multiple scientific communities desire to develop their own DSL language, tailored for their model and needs. The High-level Intermediate Representation (HIR) allows to define multiple high level DSLs in a lightweight manner by reusing most of a complex toolchain, i.e. domain specific optimizers, safety checkers and code generators.
In addition, a standardized HIR has another major advantage: It allows to easily interact with third-party developers and hardware manufacturers. For instance, the HIR of the `COSMO <http://www.cosmo-model.org/>`_ atmospheric model, serialized to a mark-up Language like `JSON <https://en.wikipedia.org/wiki/JSON>`_, can be distributed to hardware vendors, which in turn have their proprietary, in-house compilers based on the HIR and can return plain C/C++ or CUDA code. This frees third-party developers from compiling the models and hopefully improves collaboration.

In this section we define a first draft of a specification of the HIR for structured grid weathe and climate computations.


.. glossary::

  `Program`

    the main program

     .. admonition:: It has
       :class: tip

       - A :term:`GlobalVariableMap`.
       - A sequence of :term:`ScopedProgram` or :term:`ExternalKernel`

  `GlobalVariableMap`

    a map of global variables to values used through the whole :term:`Program`

  `ExternalKernel`

    describes the call to a external kernels.

    .. admonition:: It has
       :class: tip

       - a sequence of input and output :term:`FieldDecl` 's.

  `ScopedProgram`

    Provides the control flow of a component of the full model.

    .. admonition:: It has
       :class: tip

       - a list<:term:`FieldDecl`> used within the program
       - a :term:`ControlFlowGraph`

    .. admonition:: Example:
       :class: attention

       The following pseudo-code represents a :term:`ScopedProgram` where `conjugate_gradient_solver` is the actual :term:`VerticalRegionComputationDecl`
       that contains the grid points calculations that operates on fields `f` and `g`

       .. code:: c++

         for(iter: range) {
           field = f,g;
           res = conjugate_gradient_solver(f,g);
           if(res) break;
         }

  `ControlFlowGraph`

    a sequence of statements of a control flow

    .. admonition:: It has
       :class: tip

       - a list<:term:`ControlFlowGraphStmt`>

    .. admonition:: Example:
       :class: attention

       The following pseudo-code within a :term:`ScopedProgram`

       .. code:: c++

         if(tstep==1) {
           call hdiff1(u, v)
         }
         else {
           call hdiff2(u,v)
         }
         u -= u_ref

       is represented by the following :term:`ControlFlowGraph`

       .. image:: figures/controlflowexample.png

  `ControlFlowGraphStmt`

    A statement node of a :term:`ControlFlowGraph`.

    .. admonition:: It is any of
       :class: tip

       - :term:`VerticalRegionComputationDecl`, :term:`IfStmt`, :term:`BlockStmt`, :term:`CallStmt`, :term:`VarDecl`

  `IfStmt`

    an if/then/else block statement

    .. admonition:: It has
       :class: tip

       - a condition of type :term:`ControlFlowGraphStmt` || :term:`ComputationStmt`
       - a `then` statement of type :term:`ControlFlowGraphStmt` || :term:`ComputationStmt`
       - an `else` of type :term:`ControlFlowGraphStmt` || :term:`ComputationStmt`

    .. admonition:: It can be used in
       :class: note

       - :term:`ControlFlowGraphStmt`
       - :term:`ComputationStmt`

  `BlockStmt`

    a block of any number of statements.

    .. admonition:: It has
       :class: tip

       - a list<:term:`ControlFlowGraphStmt` || :term:`ComputationStmt` >

    .. admonition:: It can be used in
       :class: note

       - :term:`ControlFlowGraphStmt`
       - :term:`ComputationStmt`

  `CallStmt`

    a statement call to another :term:`ScopedProgram`, :term:`ExternalKernel` or :term:`BoundaryConditionDecl`

    .. admonition:: It has
       :class: tip

       - A list of argument :term:`FieldDecl` 's
       - One of :term:`ScopedProgram`, :term:`ExternalKernel` or :term:`BoundaryConditionDecl`

    .. admonition:: It can be used in
       :class: note

       - :term:`ControlFlowGraphStmt`

    .. admonition:: Example:
       :class: attention

       .. code:: Fortran

          Program radiation {
            Field pca1, pca2
            call boundary_condition(zero_gradient, pca1) // -> This is a CallStmt to a BC
            call lw_solver(pca1, pca2)  // -> This is a CallStmt to a ScopedProgram
          }

  `BoundaryConditionDecl`

    a declaration of a boundary condition computation. It is an alias of a :term:`StencilFunctionDecl`

    .. admonition:: It has
       :class: tip

       - a :term:`StencilFunctionDecl`

  `Interval`

    describes a vertical interval, determined by the lower and the upper bound.

  `FieldDecl`

    storage in a N-D gridded space that is referenced by in the HIR by its name.
    All :term:`FieldDecl` 's, with is_temporary==false,  will be arguments of a call to a `:term:`ScopedProgram` or :term:`StencilFunctionDecl`

    .. admonition:: It has
       :class: tip

       - string: name
       - bool: is_temporary

    .. admonition:: It can be used in
       :class: note

       - a :term:`ScopedProgram`
       - a :term:`StencilFunctionDecl`

  `VarDecl`

    a variable that represents a N-dimensional scalar.

    .. admonition:: It has
       :class: tip

       - :term:`Type`: type
       - string: name
       - int: dimension of the variable (0 for gridded :term:`FieldDecl`)
       - string: operation to initialize the variable

    .. admonition:: It can be used in
       :class: note

       - :term:`ControlFlowGraph`
       - :term:`ComputationAST`

    .. admonition:: Example:
       :class: attention

       .. code:: c++

          Program model {
            storage f1, f2;
            var v1;                 //  -> This is a gridded storage variable declaration
            var w[3] = {0.,1.,0.};  //  -> This is a 3 dimensional scalar variable with initialization
          }

  `VerticalRegionComputationDecl`
    declaration of the computations within a vertical region (determined by an :term:`Interval`), that are executed with a certain vertical loop order

    .. admonition:: It has
       :class: tip

       - a :term:`LoopOrder`
       - an :term:`ComputationAST`
       - an :term:`Interval` where the statements of the `ComputationAST` are computed

    .. admonition:: It can be used in
       :class: note

       - a :term:`ScopedProgram`

    .. admonition:: Example:
       :class: attention

       The backward substitution of the thomas algorithm

       .. math::

         x_{n} &= d'_{n} \\
         x_{i} &= d'_{i} -c'_{i}x_{i+1} \;\;\; ; i=n-1, n-2,...,1


       can be coded as two `VerticalRegionComputationDecl` 's, with different :term:`ComputationAST`, one for the update of the boundary level `n` and another one with a backward loop in the interval [n-1,1]

  `LoopOrder`

    a loop order.

    .. admonition:: It can be any of
       :class: tip

       - enum: increment, enum: decrement

  `ComputationAST`
    a sequence of statement nodes that describe the grid-point computations of a :term:`VerticalRegionComputationDecl` or a :term:`StencilFunctionDecl`

    .. admonition:: It has
       :class: tip

       - one or more than one :term:`ComputationStmt`

    .. admonition:: It can be used in
       :class: note

       - :term:`VerticalRegionComputationDecl`
       - :term:`StencilFunctionDecl`

    .. admonition:: Example:
       :class: attention

       The following pseudocode

       .. code:: c++

          if(tstep == 1) {
            field[] = avg(k, field[i+1]) * 0.5;
          }
          return field*coeff;

       will be represented as with the following AST

       .. image:: figures/astexample.png


  `ComputationStmt`
    a computation statement, i.e. a node of the :term:`ComputationAST`.

    .. admonition:: It can be any of
       :class: tip

       - :term:`BlockStmt`, :term:`ExprStmt`, :term:`ReturnStmt` (only if the :term:`ComputationAST` belongs to a :term:`StencilFunctionDecl`), :term:`StencilFunCallExpr`, :term:`VarDecl`, :term:`IfStmt`

    .. admonition:: It can be used in
       :class: note

       - :term:`ComputationAST`

  `ReturnStmt`

    a return statement

    .. admonition:: It has
       :class: tip

       - a :term:`Expr` to return

  `ExprStmt`

    a statement that encloses a :term:`Expr`

  `Expr`

    a expression, i.e. anything that contains a :term:`Identifier`, :term:`LiteralExpr`, :term:`OperatorExpr`

  `Identifier`

    .. admonition:: It can be any of
       :class: tip

       - :term:`VarAccessExpr`, :term:`FieldAccessExpr`

  `VarAccessExpr`
    an expression that represents an access to a variable

    .. admonition:: It has
       :class: tip

       - :term:`VarDecl`
       - :term:`LiteralExpr`: access index of the var has more than 1 dimensions

    .. admonition:: Example:
       :class: attention

       .. code:: c++

          globals {
            int tstep = 10;
          }
          Program model {
            var wght[2];
            if(tstep == 11)  // -> tstep is a VarAccessExpr (to global variable)
               wght[1] ++;   // -> wght is a VarAccessExpr with an (index == 1)
          }


  `FieldAccessExpr`
    an expression that represents an access to a field

    .. admonition:: It has
       :class: tip

       - :term:`FieldDecl`
       - list<:term:`Offset`>


  `LiteralExpr`

    expression that represents a literal

    .. admonition:: It has
       :class: tip

       - string: value
       - Type: type

  `Type`

    a type representation


  `OperatorExpr`
    an expression that represents an operator

    .. admonition:: It can be any of
       :class: tip

       - :term:`BinaryOpExpr`, :term:`UnaryOpExpr`, :term:`AssignmentOpExpr`, :term:`TernaryOpExpr`, :term:`StencilFunCallExpr`

  `BinaryOpExpr`
    a binary operator expression

    .. admonition:: It has
       :class: tip

       - :term:`Expr`: left
       - string: operator
       - :term:`Expr`: right

  `UnaryOpExpr`
    a unary operator expression

    .. admonition:: It has
       :class: tip

       - string: operator
       - :term:`Expr`: operand

  `AssignmentOpExpr`
    an assignment operator expression

    .. admonition:: It has
       :class: tip

       - :term:`Expr`: left
       - string: operator
       - :term:`Expr`: right

  `TernaryOpExpr`
    an ternary operator expression

    .. admonition:: It has
       :class: tip

       - :term:`Expr`: condition
       - :term:`Expr`: left
       - :term:`Expr`: right

  `StencilFunCallExpr`
    a stencil function call expression

    .. admonition:: It has
       :class: tip

       - :term:`StencilFunctionDecl`
       - list<:term:`StencilFunctionArg`>: arguments


  `Offset`

    Relative distance in a given `Direction` to a neighbor grid point.

    .. admonition:: It can be used in
       :class: note

       - :term:`FieldDecl` accesses
       - As argument of :term:`StencilFunctionDecl` 's.

    .. admonition:: Example:
       :class: attention

       `i+1`, [1,0,0]

  `Direction`
    Identifies a dimension. A direction is also treated as an :term:`Offset` with distance 0.
    It is mainly used to parameterize the direction of numerical operators in stencil functions.

    .. admonition:: It can be used in
       :class: note

       - :term:`FieldDecl` accesses
       - as argument of :term:`StencilFunctionDecl` 's.

    .. admonition:: Example:
       :class: attention

        `avg(j, field)`

  `StencilFunctionArg`
    a stencil function argument

    .. admonition:: It is any of
       :class: tip

       - :term:`Offset`, :term:`Direction`, :term:`FieldDecl`, :term:`StencilFunCallExpr`

  `StencilFunctionDecl`
    A parameterized function applied to a grid point that contains stencil operations.

    .. admonition:: It has
       :class: tip

       - list<:term:`Interval`>: intervals for which there are different specializations of the computation.
       - list<:term:`ComputationAST`>: a computation AST for each :term:`Interval`.
       - list<:term:`StencilFunctionArg`>: list of arguments that are common to the specializations of all :term:`Interval` 's.

    .. admonition:: It can be used in
       :class: note

       - a :term:`ComputationAST`
