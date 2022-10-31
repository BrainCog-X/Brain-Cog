from setuptools import find_packages
from setuptools import setup

with open("./requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read()

setup(
    install_requires=install_requires,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    name='braincog',
    version='0.2.7.12',
    author='',
    python_requires='>=3.6'
)
