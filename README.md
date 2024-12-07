# PermuMark

## Requirements

### Non-Python Library

- gcc (or any other C compiler)
- [GMP](https://gmplib.org/) (e.g., through `sudo apt-get install libgmp-dev`)
- [SageMath](https://www.sagemath.org/) 10.4 (e.g., through `conda install sage=10.4 -c conda-forge`)

### Python Packages

PermuMark is developed on Python3.11, and it should also work for Python>=3.10.
The following dependencies can be installed via `pip install -e .`

- datasets==2.21.0
- pytorch==2.4.0
- scipy==1.13.1
- sympy==1.13.2
- transformers==4.43.2

## Build

The ranking and unranking of derangement are implemented in C and exposed to Python through a shared library (DLL should also work).
To build the shared library, simply run `./build.sh` under the project directory, it will produce a shared library under `permumark/derangement`.
