# setup.py
from setuptools import find_packages, setup

setup(
    name="pokergym",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=1.2.0",
        "pettingzoo>=1.25.0",
        "numpy>=2.3.2",
        "deuces>=0.2.1",
        "tyro>=0.9",
        "termcolor>=3.1",
    ],
    extras_require={
        "dev": [
            "pytest",
        ],
    },
    author="Alexandre Symeonidis-Herzig",
    author_email="a.symeonidis.herzig@gmail.com",
)
