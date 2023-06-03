# Parameter estimation and Updation of Steel Truss Bridge
In this example, system identification and model updation is performed on `RDSO45mTruss` model. Structural response is recorded by instrumenting an existing bridge, built with the `RDSO45mTruss` specification, with various strain gauges and displacement sensors. Recorded response of the bridge when loaded train is run at crawling speed and 70kmph speed are stored in `span30/5kmcr.txt` and `span30/70km.txt` files.

For parameter estimation and model updation multiple files are created:-
* `parameterEstimation.ipynb` is a notebook containing the calculations to estimate the modal parameters using cross-correlation of results.
* `modelUpdation.ipynb` is a notebook containing procedure to update the numerical model by identifying and updating appropriate structural parameters.
* `ModalParamterList.py` contains definition of `class ModalParameterList` which contains detailed procedure to optimize and estimate the modal parameters.
* `PyOMA.py` is a package written by Pasca et.al. for Operational Modal Analysis. This package is used to get mode shapes from the power spectral density of signals.
* `createImbalancedForceVector.py` is a preprocessing subroutine to generate loading pattern at different vehicle spacing. This is separated as generating the load vector is costly.
* `displayModeShapesOnStructure3D.py` displays the mode shape superimposed on `RDSO45mTruss` model based on the results stored in `.AllResults.json` file created by `parameterEstimation.ipynb` notebook.
* Files for cross transfer of data between scripts,
  * `DataExchange/.AllResults.json` file contains results from `parameterEstimation.ipynb` notebook and is required by `displayModeShapesOnStructure3D.py` to display the mode shapes. You can copy `sample.AllResults.json` to `.AllResults.json` to work without running `parameterEstimation.ipynb` notebook, and
  * `DataExchange/.ImbalancedActionVector150.npz` is a datafile containing the preprocessed load vector computed by `createImbalancedForceVector.py` for `MAX_LOADCASE=150`. `modelUpdation.ipynb` requires the file to update the model. You can copy `sample.ImbalancedActionVector150.npz` to `.ImbalancedActionVector150.npz` to work without running `createImbalancedForceVector.py`.
* Files containing results from a commercial software Midas Civil<sup>TM</sup>
  * `Midas/Beam Stress_150LCs.xls` for beam stress of ideal model,
  * `Midas/Beam Stress_AfterUpdation_150LCs.xls` for beam stress after model updation,
  * `Midas/Displacements(Global)_150LCs.xls` for displacement of ideal model,
  * `Midas/Displacements(Global)_AfterUpdation_150LCs.xls` for displacement after model updation,

## References
> Dag Pasquale Pasca, Angelo Aloisio, Marco Martino Rosso et al., PyOMA and PyOMA_GUI: A Python module and software for Operational Modal Analysis. SoftwareX (2022) 101216, [https://doi.org/10.1016/j.softx.2022.101216](https://doi.org/10.1016/j.softx.2022.101216).