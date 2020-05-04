# Dawn Tests

* `integration-test/`: Integration tests from SIR to IIR.
    - `from-sir/`: Lower from SIR to IIR and assert properties. TODO Rename this to `optimizer`
    - `serializer/`: SIR and IIR serialization and deserialization tests. TODO Rename this to `irs`.
    - `unstructured/`: Experimental playground for integration with the unstructured interface.

* `unit-test/`: Unit tests on dawn components.
    - `dawn/`: Unit tests for each component. TODO Separate into `optimizer`, `codegen`, and `irs`.
    - `dawn4py-tests/`: Tests of the python interface (using pytest).
    - `driver-includes/`: Tests properties of the driver includes and generated code.
    - `interface/`: Tests for components of the experimental unstructured interface(s).
