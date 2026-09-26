from setuptools import find_packages, setup

setup(
    name="executive-comm-coach",
    version="0.3.0",
    packages=find_packages(include=["core", "core.*"]),
)

