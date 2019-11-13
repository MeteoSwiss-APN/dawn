Dawn4Py
=======

## Introduction

**Dawn4Py** is the Python binding of the Dawn toolchain. See the top level `README` for a detailed description.

## Installing *dawn4py* in a virtualenv

```bash
git clone git@github.com:MeteoSwiss-APN/dawn.git
python -m venv .dawn_venv
source .dawn_venv/bin/activate
pip install wheel
pip install -e ./dawn/dawn  # Add -v to see the compilation output  
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
