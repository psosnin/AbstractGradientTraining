"""Setup for pip package."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='abstract_gradient_training',
    version='0.1',
    description='Abstract gradient training of neural networks',
    url='https://github.com/psosnin/abstract-gradient-training',
    author='Philip Sosnin',
    author_email='philipsosnin@gmail.com',
    packages=find_packages(),
    install_requires=[
      'torch', 
      'tqdm',
      'torchvision',
      'pydantic'
    ],
    platforms=['any'],
)