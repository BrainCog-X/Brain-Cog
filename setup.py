from setuptools import find_packages
from setuptools import setup

with open("./requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read()
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setup(
    install_requires=install_requires,
    packages=find_packages(),
    description="BrainCog is an open source spiking neural network based brain-inspired cognitive intelligence engine for Brain-inspired Artificial Intelligence and brain simulation. More information on braincog can be found on its homepage http://www.brain-cog.network/",
    long_description=long_description,  
    long_description_content_type="text/markdown",  
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
    version='0.2.7.18',
    author='braincog',
    python_requires='>=3.6'
)
