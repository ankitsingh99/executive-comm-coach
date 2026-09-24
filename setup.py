from setuptools import setup, find_packages

setup(
    name="executive-comm-coach",
    version="0.1.0",
    packages=find_packages(include=["core", "core.*"]),
)
