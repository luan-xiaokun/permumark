# PermuMark

## Requirements & Build

### TL;DR

1. install libgmp using `sudo apt-get install libgmp-dev`
2. prepare a Python3.11 environment `conda create -n permumark python=3.11`
3. install SageMath `conda install sage=10.4`
4. build shared library `./build.sh`
5. install the package `pip install -e .`

### Non-Python Library

- gcc (or any other C compiler)
- [GMP](https://gmplib.org/) (e.g., through `sudo apt-get install libgmp-dev`)
- [SageMath](https://www.sagemath.org/) 10.4 (e.g., through `conda install sage=10.4`)

The ranking and unranking of derangement are implemented in C and exposed to Python through a shared library (DLL should also work).
To build the shared library, simply run `./build.sh` under the project directory, it will produce a shared library under `permumark/derangement`.

### Python Packages

PermuMark is developed on Python3.11, and it should also work for Python>=3.10.
The following dependencies can be installed via `pip install -e .`

- datasets==2.21.0
- torch==2.5.0
- scipy==1.14.1
- sympy==1.13.1
- transformers==4.46.3
