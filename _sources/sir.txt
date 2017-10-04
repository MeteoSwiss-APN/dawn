.. _SIR:

Stencil Intermediate Representation
###################################

Developing a single DSL that is able to support all numerical methods and computational patterns present in GFD models present a serious challenge due to the wide domain it needs to cover. In this project, we accept the reality that multiple scientific communities desire to develop their own DSL language, tailored for their model and needs. The Stencil Intermediate Representation (SIR) allows to define multiple high level DSLs in a lightweight manner by reusing most of the complex toolchain i.e the Dawn library. In addition, a standardized SIR has another major advantage: It allows to easily interact with third-party developers and hardware manufacturers. For instance, the SIR of the `COSMO <http://www.cosmo-model.org/>`_ atmospheric model, serialized to a mark-up Language like `JSON <https://en.wikipedia.org/wiki/JSON>`_, can be distributed to hardware vendors, which in turn have their proprietary, in-house compilers based on the SIR and can return plain C/C++ or CUDA code. This frees third-party developers from compiling the models and hopefully improves collaboration.

In this section we define the specification of the SIR.