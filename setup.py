import glob
import os

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Define the C extension
derangement_extension = Extension(
    "permumark.derangement.derangement",
    sources=["permumark/derangement/derangement.c"],
    extra_compile_args=["-O3"],
)


class CustomBuildExt(build_ext):
    def run(self):
        super().run()
        so_pattern = os.path.join(
            self.build_lib,
            "permumark",
            "derangement",
            "derangement.*.so",
        )
        generated_sos = glob.glob(so_pattern)
        if not generated_sos:
            raise FileNotFoundError(
                f"No generated .so files found matching {so_pattern}"
            )
        generated_so = generated_sos[0]

        target_dir = os.path.join(self.build_lib, "permumark", "derangement")
        os.makedirs(target_dir, exist_ok=True)
        self.copy_file(generated_so, os.path.join(target_dir, "derangement.so"))
        os.remove(generated_so)


setup(
    name="permumark",
    version="0.1.0",
    packages=["permumark", "permumark.derangement"],
    ext_modules=[derangement_extension],
    cmdclass={"build_ext": CustomBuildExt},
    install_requires=[
        "datasets==2.21.0",
        "peft==0.12.0",
        "sagemath-standard==10.4",
        "scipy==1.14.1",
        "sympy==1.13.3",
        "tokenizers==0.19.1",
        "torch==2.4.0",
        "tqdm==4.67.1",
        "transformers==4.43.2",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
)
