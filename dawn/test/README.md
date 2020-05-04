# Dawn Tests

* `integration-test/`: Integration tests from SIR to IIR.
    - `from-sir/`: Lower from SIR to IIR and assert properties. TODO Rename this to `optimizer`
    - `serializer/`: SIR and IIR serialization and deserialization tests. TODO Rename this to `irs`.
    - `unstructured/`: Experimental playground for integration with the unstructured interface.

* `unit-test/`: Unit tests on dawn components.
    - `dawn/`: Unit tests for each component. TODO Separate into `optimizer`, `codegen`, and `irs`.
        - `AST/`: Tests for the AST. TODO Move to `irs`/.
        - `CodeGen/`: Code generation from IIR. TODO Move to `codegen/`.
        - `IIR/`: Tests of the IIR. TODO Move to `irs/`.
        - `Optimizer/`: Tests optimizer passes. TODO Move to `optimizer/`.
        - `SIR/`: SIR tests. TODO Move to `SIR/`.
        - `Support/`: Tests of the low-level support library. TODO Move to `irs/`.
        - `Validator/`: Validation tests. TODO Move to `optimizer/`.
    - `dawn4py-tests/`: Tests of the python interface (using pytest).
        - TODO `sir-generation/`: Tests of the`sir_util`.
        - TODO `compilation/`: Test compilation interface.
        - TODO `codegen/`: Test code generation interface.
    - `driver-includes/`: Tests properties of the driver includes and generated code.
    - `interface/`: Tests for components of the experimental unstructured interface(s).
