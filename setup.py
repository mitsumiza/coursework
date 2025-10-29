from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="characteristic_polynomial",
    version="1.0.0",
    author="Ролик София",
    author_email="sofiya-rolik-2006@mail.ru",
    description="Вычисление характеристических многочленов графов целочисленной решетки",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mitsumiza/coursework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "charpoly-experiment=experiment:main",
            "charpoly-run-all=run_experiments:run_multiple_experiments",
        ],
    },
    keywords="graph-theory characteristic-polynomial lattice-graph spectral-graph-theory",
)
