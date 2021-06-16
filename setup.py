import sys

from setuptools import find_packages
from skbuild.setuptools_wrap import setup

setup(
    name="cubecobra-recommender-generators",
    version="0.0.1",
    description="A faster more scalable version of the pure python keras Sequence.",
    license="AGPL",
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    cmake_install_dir="src/generated",
    include_package_data=True,
    build_type='RelWithDebInfo'
)
