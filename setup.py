from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="evodoodle",
    version="1.0.0",
    author="Anusha Bishop",
    author_email="anusha.bishop@berkeley.edu",
    description="A doodling game for understanding evolution across space",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AnushaPB/evodoodle",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "seaborn",
        "matplotlib",
        "geonomics",
        "pygame",
        "scikit-learn",
    ],
)