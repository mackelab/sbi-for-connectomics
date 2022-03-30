from setuptools import find_packages, setup

REQUIRED = [
    "sbi",
    "svgutils==0.3.1",
    "invoke",
    "jupyterlab",
    "matplotlib",
]

setup(
    name="consbi",
    python_requires=">=3.6.0",
    packages=find_packages(),
    install_requires=REQUIRED,
)