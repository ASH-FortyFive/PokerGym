# setup.py
from setuptools import find_packages, setup

setup(
    name="pokergym",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.26.2",
        "numpy>=1.21.0"],
    author="Alexandre Symeonidis-Herzig",
    author_email="a.symeonidis.herzig@gmail.com",

)