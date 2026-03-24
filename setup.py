from setuptools import setup, find_packages

setup(
    name="neutrino-osc-calculator",
    version="1.0.0",
    author="Aaryan Chaulagain, Anju Dhakal",
    author_email="aaryan1379@xavier.edu.np",
    description="A lightning-fast three-flavor neutrino oscillation calculator in constant-density matter.",
    long_description=open("README.md").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://zenodo.org/...", # Update with your DOI link later
    packages=find_packages(), # This automatically finds core/, solvers/, etc.
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires='>=3.8',
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0"
    ],
)