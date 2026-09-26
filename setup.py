from setuptools import find_packages, setup

setup(
    name="executive-comm-coach",
    version="0.4.1",
    packages=find_packages(include=["core", "core.*"]),
)
