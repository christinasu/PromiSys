# PromiSys
## Modeling of Promiscuous Ligand-Receptor Systems
PromiSys contains code for modeling signaling pathways with promiscuous ligand-receptor systems, focused on analysis of the bone morphogenetic protein (BMP) signaling pathway. As an example application, this model is used to study how promiscuous pathway architectures can enable combinations of ligands to specifically activate target cell types, defined by their receptor expression. This concept of "combinatorial addressing" is described in greater detail in the associated publication.

## Contents
* Code: The PromiSys module is provided in the Code\ directory. Installation instructions are given below.
* CombinatorialAddressing: The model is applied to explore the capacity of the BMP signaling pathway for implementing addressing of multiple cell types by different ligand combinations. The notebook "CombinatorialAddressing.ipynb" contains all code needed to reproduce the data analysis and figures of the associated paper. Data can be downloaded [here](https://dx.doi.org/10.22002/D1.1692). By default, the notebook assumes that data are in a Data\ directory (at the same level as the CombinatorialAddressing\ directory).

## Installation
Running the code requires Python and several widely used packages, such as NumPy, scikit-learn, Pandas, and Matplotlib. These packages can be obtained through standard installations of Anaconda or other Python distributions.

An additional dependency is Equilibrium Toolkit (EQTK), which can be installed using `pip` as `pip install --upgrade eqtk`. Further information on this package is provided [here](https://eqtk.github.io/).

To install PromiSys locally, run `pip install -e .` from the directory containing `setup.py`, or `[appropriate path]\Code` if file and folder names are unmodified.

## References
### Ligand-receptor promiscuity enables cellular addressing
Su, C.J., Murugan, A., Linton, J.M., Yeluri, A., Bois, J., Klumpe, H., Langley, M.A., Antebi, Y.E., Elowitz, M.B. *Cell Systems* 2022.
