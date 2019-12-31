
Formal Definition of the Dawn Parallel Model
============================================

Contract to the user from HIR
-----------------------------

Dawn will generate code based on the operations passed from high-level
intermediate representation. It's specification can be found `here`_.

Dependencies
~~~~~~~~~~~~

Generally, stencil computation in dawn differentiates between horizontal
and the vertical dimension. The assumption is that individual statements
of a stencil are embarrassingly parallel in the horizontal. For that
reason, there is no guarantee that any horizontal point's computation
will be finished before any other in each user statement.

Vertical Dependencies
^^^^^^^^^^^^^^^^^^^^^

Each vertical region of a stencil specifies a loop order of execution.
Dawn will uphold the following rules:

-  If the loop order is forward, any statement with a vertical
   dependency on other statements executed on a vertical level with
   index ``k`` will be executed after each statement that it depends on
   that appears previously in the current vertical region or in any
   previous vertical region was executed on every vertical level with
   indices ``k-1`` and smaller.
-  If the loop order is backward, any statement with a vertical
   dependency on another statement executed on a vertical level with
   index ``k`` will be executed after each statement that it depends on
   that appears previously in the current vertical region or in any
   previous vertical region was executed on every vertical level with
   indices ``k+1`` and bigger.

An example of this would be the following

.. code:: c++

   1 vertical_region(k_start, k_end)
   2   one = 1
   3 vertical_region(k_start, k_end)
   4   two = one[k - 1]

Here, for every level with index ``k``, line two will be executed before
line 4.

Horizontal Dependencies
^^^^^^^^^^^^^^^^^^^^^^^

Irrespective of the loop order every statement with a horizontal
dependency on other statements will be executed after all statements it
depends on are executed on the full horizontal plane in the same
vertical level.

An example of this is the following

.. code:: c++

   1 vertical_region(k_start, k_end)
   2   one = 1
   3   two = one[i - 1]

For every vertical level, we guarantee that line two will be executed on
the full horizontal plane before line three will be executed.

--------------

   It is currently unclear how we should handle this in one corner case:
   Non-temporary fields can cause problem in the following case: write
   center, read offset, write center. This is currently translated to
   one multistage with three stages. The problem is that on input /
   output fields we duplicate the writes which can have 2 side-effects
   if neighboring blocks are out of sync: Block 1 writes in its extended
   domain (line 2), block 2 writes (line 2) (same value), then reads
   offset (line 3), then writes in the center (line 4). Now Block 1
   reads (line 3) and gets the wrong value. This means that if we want
   to solve the problem we need a global sync in that case (meaning
   breaking the kernel call).

.. code:: c++

   1 vertical_region(k_start, k_end){
   2   one = 1
   3   two = one[i - 1]
   4   one = 2
   5 }

translates to

::

   MultiStage_0 [parallel]
       {
         Stage_0
         {
           Do_0 { Start : End }
           {
             one[<no_horizontal_offset>,0] = int_type 1;
           }
           Extents: [(-1,0),(0,0),(0,0)]
         }
         Stage_1
         {
           Do_0 { Start : End }
           {
             two[<no_horizontal_offset>,0] = one[-1,0,0];
           }
           Extents: [<no_horizontal_extent>,(0,0)]
         }
         Stage_2
         {
           Do_0 { Start : End }
           {
             one[<no_horizontal_offset>,0] = int_type 2;
           }
           Extents: [<no_horizontal_extent>,(0,0)]
         }
       }

--------------

Compute Domain
~~~~~~~~~~~~~~

If the vertical region does not specify any horizontal extent, dawn
ensures that the compute-domain of statements that others depend on is
extended to fill all the intermediate fields as required. If a
horizontal compute-domain is specified, it is never extended.

Internal Contract of Data Structures
------------------------------------

StencilInstantiation
~~~~~~~~~~~~~~~~~~~~

StencilInstantiations (SI) are user defined and are never changed during
any optimization. Each SI will be globally split from every other one
and there is no optimization across SIs. For every SI, a callable API
will be generated.

Stencil
~~~~~~~

Stencils serve no purpose right now

--------------

   It might be useful to remove them altogether since even if they
   existed, no concept can't be captured by the other structures

--------------

Multistage
~~~~~~~~~~

A Multistage (MS) is the equivalent of a global synchronization between
all processes on the full domain. each Multistage can only have one loop
order

-  in the CUDA back end they are the equivalent of a kernel call
-  in the GridTools back end they are the equivalent of a multistage
-  in the naive back end every storage will be synchronized on the host
   at the beginning of each multistage

--------------

   Can we discuss why we still have the loop order on this level? Would
   it potentially be interesting to promote this to the Do level and
   change the fusion strategy if we use the GridTools back end? Or do we
   know that we are losing performance in the CUDA back end if we mix
   forward and backward loops versus new kernel calls?

--------------

Stage
~~~~~

A Stage (St) specifies a specific compute-domain for different
computations. These can be either global computation bounds or extended
compute domains due to larger required input data.

Each stage has the option to require (block-wide) synchronization.

-  in the CUDA back end they are equivalent to conditionals that check
   if the block / thread combination is inside the compute domain and a
   potential ``__syncthreads()`` afterwards.
-  in the GridTools back end they are the equivalent of a
   ``stage(_with_extents)``
-  in the naive back end they are represented a the innermost ij loops

--------------

   currently the ij loop of the stage is innermost, would it be
   interesting to move it to be the outermost one to be consistent with
   the execution model?

--------------

DoMethod
~~~~~~~~
A DoMethod (Do) specifies a vertical iteration space in which
computation happens. Dos in a St can't be overlapping.

-  in the CUDA back end each Do is represented with a for loop inside a
   kernel call
-  in the GridTools back end each Do is translated to an
   ``apply``\ function that has it's own interval specified
-  in the naive back end each Do is represented with an (outermost) ij
   loop

--------------

   Is the constraint that two Dos can't be overlapping in a Stage still
   useful for general back ends? Should the GT back end run its own pass
   to enforce this the same way the CUDA one does?

--------------

Execution Model from Internal Structures
----------------------------------------
The above assumptions require the following execution order:

.. code:: c++

   1 for MultiStage
   2  for k
   3   for Stage
   4    for ij
   5     for DoMethod
   6      execute_statements()

Since MS can be dependent on each other they have to be a sequential loop.
The assumption that each stage will have all it's dependent statements executed
on each lower k-level can be translated into the k-loop happening before the
Stage loop.

--------------

   The biggest question we have to ask ourselves is if we want to
   support multiple stages in a sequential k-setting. If so we need to
   be very precise here which we are not (yet). Depending on required
   performance, we want the ij loop to be as far outside as possible

--------------


   .. _here: https://github.com/MeteoSwiss-APN/HIR
