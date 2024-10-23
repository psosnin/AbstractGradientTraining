"""Setup for pip package."""

from setuptools import find_packages
from setuptools import setup

setup(
    name="abstract_gradient_training",
    version="0.2",
    description="Abstract gradient training of neural networks",
    url="https://github.com/psosnin/abstract-gradient-training",
    author="Philip Sosnin",
    author_email="philipsosnin@gmail.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch >= 2.4.1",
        "torchvision >= 0.19",
        "pydantic >= 2.9",
        "gurobipy >= 10.0",
        "gurobipy-stubs >= 2.0",
        "numpy >= 2.1",
        "scipy >= 1.14",
        "parameterized >= 0.9",
        "pytest >= 8.3",
    ],
    extras_require={
        "examples": [
            "seaborn >= 0.13",
            "matplotlib >= 3.9",
            "medmnist >= 3.0",
            "uci_datasets @ git+https://github.com/treforevans/uci_datasets.git",
            "matplotlib-label-lines >= 0.7",
        ],
    },
    platforms=["any"],
)
