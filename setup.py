import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent


setup(
    name='cwarehouse',
    version='0.1',
    description='Collaborative warehouse',
    packages= find_packages(),
    classifiers=[
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10.9",
    ],
    install_requires=[
        "numpy",
        "gym"
    ],
)
