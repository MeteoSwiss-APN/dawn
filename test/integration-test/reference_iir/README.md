## Updating References

To update the references if iir was changed simply write them to disk, starting from the in-memory representation, e.g.
```
  createXXXStencilIIRInMemory(XXX_stencil_memory);
  IIRSerializer::serialize("XXX_stencil.iir", XXX_stencil_memory, IIRSerializer::SK_Json);
```