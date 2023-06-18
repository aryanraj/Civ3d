# Structural Analysis Package for Civil Engineering Structures

The package contains a solver which could be used to perform analysis to solve structural analysis problems and be easily integrated with workflows of an organisation working on analysis and design of civil engineering structures.

# Aim of the project
The aim of the project is to help civil engineering professionals perform structural analysis as good as if not better than commercial software packages. Many organisations have a workflows where they make use of a commercial software for the analysis and then use home grown workflows involving spreadsheets to perform design. These commertial softwares are good for analysis and to some extent structural design. However, their binding interfaces with external softwares is unusable if not nonexistent. Due to their closed source nature, the end user is left at the mercy of the developers to get a feature integrated in the software.

# Requirements
This project requires following dependencies:
* [numpy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [Scipy](https://scipy.org/)
* [matplotlib](https://matplotlib.org/)
* [pythonocc-core](https://github.com/tpaviot/pythonocc-core)

# Installation Instruction
The dependencies can be installed using [conda](https://docs.conda.io/en/latest/) environments. Define the conda environment:
```
conda create --name=civ3d python=3.9
conda activate civ3d
```
Installing dependencies:
```
conda install numpy pandas scipy matplotlib
conda install -c conda-forge pythonocc-core=7.7.0
```
Finally add the package to the modules path so that it can be included everywhere
```
conda develop /path/to/root/of/package
```

# Examples
Currently there are two examples in the `Examples/` folder which shows different work cases for this project.

> **_NOTE:_** This README is incomplete, requires a wiki page to go into in depth information about the analysis modules.