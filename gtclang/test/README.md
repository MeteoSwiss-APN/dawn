# GTClang Tests

* `integration-test/`: End-to-end gtclang integration tests (mostly) using [tester.py](integration-test/tester.py) to assert diagnostic messages are printed to stdout and stderr while running.
    - `Accesses/`: Test field accesses. TODO This should be removed.
    - `CodeGen/`: Test code generation from DSL input. TODO This should be removed.
    - `Diagnostics/`: Assert frontend diagostic error messages. These tests should stay in gtclang.
    - `IIRSerializer/`: Dump IIR from DSL files and compare to reference files. TODO Remove this duplicate of dawn tests
    - `Regression/`: Test IIR generation to specific regression issues. TODO Move this into dawn
    - `SIR/`: Generate SIR from DSL code and compare to reference SIR files. These tests should stay in gtclang.

* `unit-test/`: Unit tests on dawn components.
    - `Frontend/`: Tests for the frontend library.
    - `Support/`: Tests of the gtclang support library.
    - `Unittest/`: Google Test integration. TODO Remove this.
