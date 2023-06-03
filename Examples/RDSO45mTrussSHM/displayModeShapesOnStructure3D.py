from pathlib import Path
from typing import Union
import numpy as np
import numpy.typing as npt
from Examples.RDSO45mTruss.Structure3D import Structure3D
from Models import DOFClass, FixedBeam
from Views.simpleStructure import SimpleView
import json

def getResultsFromElements(elements:list[Union[DOFClass,FixedBeam]], loadCases:list[int]) -> npt.NDArray[np.float64]:
  res = []
  for element in elements:
    if type(element) is DOFClass:
      res.append(element.displacement(loadCases).flatten())
    elif type(element) is FixedBeam:
      res.append(element.getAxialStrainForLoadCases(loadCases).flatten())
  return np.array(res)

def addLoadToElements(elements:list[Union[DOFClass,FixedBeam]], loads:npt.NDArray[np.float64], loadCases:list[int]) -> None:
  for element, load in zip(elements, loads):
    if type(element) is DOFClass:
      element.addAction(load, loadCases)
    elif type(element) is FixedBeam:
      element.addPointLoad(0, load, element.L, loadCases)
      element.addPointLoad(0, -load, 0, loadCases)

def getLoadFromTargetModeShapes(elements:list[Union[DOFClass,FixedBeam]], targetModeShapes:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
  n = len(elements)
  allLoads = np.eye(n)
  loadCases = list(range(n))
  addLoadToElements(elements, allLoads, loadCases)
  DOFClass.analyse()
  Amatrix = getResultsFromElements(elements, loadCases)
  addLoadToElements(elements, -allLoads, loadCases)
  DOFClass.analyse()
  return np.linalg.solve(Amatrix, targetModeShapes)

if __name__ == "__main__":
  with open(Path(__file__).parent.joinpath('DataExchange/.AllResults.json')) as f:
    AllResults = json.load(f)
  # print("Natural Frequencies (rad/s):")
  # print([_["omega"] for _ in AllResults])
  # print("Natural Frequencies (Hz):")
  # print([_["omega"]/2/np.pi for _ in AllResults])
  # print("Damping Ratio:")
  # print([_["zeta"] for _ in AllResults])
  # print("Mode Shapes:")
  # print([_["modeShape"] for _ in AllResults])

  structure = Structure3D()
  elements:list[Union[DOFClass,FixedBeam]] = [
    structure.truss1.node_by_name("L4").DOF[2], #D5
    structure.truss2.node_by_name("L4").DOF[2], #D6
    structure.truss1.node_by_name("L4").DOF[1], #D7
    structure.truss1.diagonalBeams[0].childBeams[0], #S21
    structure.truss1.bottomChordBeams[0].childBeams[0], #S22
    structure.truss1.verticalBeams[0].childBeams[0], #S23
    structure.truss1.diagonalBeams[1].childBeams[0], #S24
    structure.truss1.bottomChordBeams[1].childBeams[1], #S25
    structure.truss1.bottomChordBeams[2].childBeams[0], #S26
    structure.truss1.diagonalBeams[-2].childBeams[0], #S27
    structure.truss1.diagonalBeams[-1].childBeams[0], #S28
    structure.truss2.diagonalBeams[0].childBeams[0], #S29
    structure.truss2.bottomChordBeams[0].childBeams[0], #S30
    structure.truss2.verticalBeams[0].childBeams[0], #S31
    structure.truss2.diagonalBeams[1].childBeams[0], #S32
    # structure.truss2.bottomChordBeams[1].childBeams[1], #S33
    structure.truss2.bottomChordBeams[2].childBeams[0], #S34
    structure.truss2.diagonalBeams[-2].childBeams[0], #S35
    structure.truss2.diagonalBeams[-1].childBeams[0], #S36
  ]

  targetModeShapes = np.array([_["modeShape"] for _ in AllResults]).T
  targetLoads = getLoadFromTargetModeShapes(elements, targetModeShapes)
  targetLoadCases = list(range(targetLoads.shape[1]))
  addLoadToElements(elements, targetLoads, targetLoadCases)
  DOFClass.analyse()
  ModeShapes = DOFClass.getDisplacementVector(targetLoadCases)

  ModeShapeTags = [f"Mode {i}: {_['omega']/2/np.pi:.2f}Hz" for i,_ in enumerate(AllResults)]
  view = SimpleView(structure.nodes, structure.beams, ModeShapes, ModeShapeTags)
  view.start()
