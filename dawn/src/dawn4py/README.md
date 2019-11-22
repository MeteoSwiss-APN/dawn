Dawn4Py
=======

## Introduction

**Dawn4Py** is the Python binding of the Dawn toolchain. See the top level `README` for a detailed description.

## Installing *dawn4py*

We thoroughly recommend to create first a Python *virtualenv* for your project:

```bash
python -m venv .project_venv
source .project_venv/bin/activate
pip install wheel
```
Then you can install *dawn4py* directly from the GitHub repository:

```bash
pip install dawn4py@git+https://github.com/MeteoSwiss-APN/dawn.git#subdirectory=dawn  # Add -v to see the compilation output  
```

Alternatively, you can checkout locally the Dawn repository and install the Python bindings from there:

```bash
git clone git@github.com:MeteoSwiss-APN/dawn.git
pip install -e ./dawn/dawn  # Add -v at the end to see the compilation output  
```


## Examples

Take a look to the `dawn/examples/python` folder (only `new_copy_stencil.py` has been updated).

## License

This project is licensed under the terms of the **MIT** license.

The full license can be found [here](https://opensource.org/licenses/MIT).

<!-- Links -->
[Documentation.Badge]: https://img.shields.io/badge/documentation-link-blue.svg
[MIT.License]: https://img.shields.io/badge/License-MIT-blue.svg
[Version.Badge]: https://badge.fury.io/gh/MeteoSwiss-APN%2Fdawn.svg
