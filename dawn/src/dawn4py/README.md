Dawn4Py
=======

## Introduction

**Dawn4Py** provides Python binding of the Dawn toolchain. See the top level `README` for a detailed description.

## Installing *dawn4py*

We thoroughly recommend to create first a Python *virtualenv* for your project:

```bash
python -m venv .project_venv
source .project_venv/bin/activate
pip install wheel
pip install scikit-build
```
Then you can install *dawn4py* directly from the GitHub repository:

```bash
pip install dawn4py@git+https://github.com/MeteoSwiss-APN/dawn.git#subdirectory=dawn  # Add -v to see the compilation output
```

Alternatively, you can clone locally the Dawn repository and install the Python bindings from there. This is specially useful if you plan to edit Dawn sources, since you can test your changes easily without reinstalling the package. In this case, we recommend you to add the `-e` option to the `pip install` command to perform an *editable* installation:

```bash
git clone git@github.com:MeteoSwiss-APN/dawn.git
pip install -e ./dawn/dawn  # Add -v at the end to see the compilation output
```

Changes in Python sources are instantly available in your environment. Changes in C++ sources require a recompilation of the CPython extension, which you can do by running the `develop` command of the `setup.py` script:

```bash
cd ./dawn/dawn && python setup.py develop
```

## Development

Do not edit the `_dawn4py.cpp` file directly. Instead, run `dawn/scripts/refactor/make_pybind11_sources.py`.

## Examples

Take a look to the files in the `dawn/examples/python` folder.

## License

This project is licensed under the terms of the **MIT** license.

The full license can be found [here](https://opensource.org/licenses/MIT).

<!-- Links -->
[Documentation.Badge]: https://img.shields.io/badge/documentation-link-blue.svg
[MIT.License]: https://img.shields.io/badge/License-MIT-blue.svg
[Version.Badge]: https://badge.fury.io/gh/MeteoSwiss-APN%2Fdawn.svg
