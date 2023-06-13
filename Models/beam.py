from __future__ import annotations
from typing import Union
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field, InitVar
from . import DOFClass, Node, BeamSection, FixedBeam
from . import utils

@dataclass
class Beam:
  nodes: list[Node]
  section: InitVar[BeamSection]
  beta: InitVar[float] = 0 # Angle in degrees
  A: Node = None
  B: Node = None
  constraintsA: npt.NDArray[np.bool_] = None
  constraintsB: npt.NDArray[np.bool_] = None
  endStiffnessA: npt.NDArray[np.float64] = field(default_factory=lambda:np.zeros((6,6)))
  endStiffnessB: npt.NDArray[np.float64] = field(default_factory=lambda:np.zeros((6,6)))
  childBeams: list[FixedBeam] = field(init=False, default_factory=list)
  endStiffnessFactorA: npt.NDArray[np.float64] = field(init=False, default_factory=lambda:np.zeros(6))
  endStiffnessFactorB: npt.NDArray[np.float64] = field(init=False, default_factory=lambda:np.zeros(6))

  def __post_init__(self, section:BeamSection, beta:float):
    self.constraintsA = utils.ensure1DNumpyArray(self.constraintsA, np.bool_, [1,1,1,1,1,1])
    self.constraintsB = utils.ensure1DNumpyArray(self.constraintsB, np.bool_, [1,1,1,1,1,1])
    if len(self.nodes) < 2:
      raise Exception("Supply minimum of 2 nodes")
    if self.A is None:
      self.A = self.nodes[0]
      self.nodes[0] = Node(
        coord=self.A.coord,
        axis=utils.getAxisFromTwoNodesAndBeta(self.A.coord, self.nodes[1].coord, beta=beta)
      )
    if self.B is None:
      self.B = self.nodes[-1]
      self.nodes[-1] = Node(
        coord=self.B.coord,
        axis=utils.getAxisFromTwoNodesAndBeta(self.nodes[-2].coord, self.B.coord, beta=beta)
      )
    if np.any(np.abs(utils.getAxisFromTwoNodesAndBeta(self.nodes[0].coord, self.nodes[1].coord, beta=beta) - self.nodes[0].axis) > 10 * np.finfo(np.float64).eps):
      raise Exception("Axis of first node should be same as first childBeam")
    if np.any(np.abs(utils.getAxisFromTwoNodesAndBeta(self.nodes[-2].coord, self.nodes[-1].coord, beta=beta) - self.nodes[-1].axis) > 10 * np.finfo(np.float64).eps):
      raise Exception("Axis of last node should be same as last childBeam")
    self.setEndConstrains(self.constraintsA, self.constraintsB)
    self.addEndStiffness(self.endStiffnessA, self.endStiffnessB)
    for nodei, nodej in zip(self.nodes[:-1], self.nodes[1:]):
      self.childBeams.append(FixedBeam(nodei, nodej, section, beta))

  def __del__(self):
    raise NotImplementedError(f"Deletion of {type(self).__name__} is not supported")

  @property
  def DOF(self) -> list[DOFClass]:
    return sum([_.DOF for _ in self.childBeams], start=[])

  def setEndConstrains(self, constraintsA:npt.NDArray[np.bool_]=None, constraintsB:npt.NDArray[np.bool_]=None):
    if not constraintsA is None:
      self.constraintsA = utils.ensure1DNumpyArray(constraintsA, np.bool_, [1,1,1,1,1,1])
      self.A.constrainChildNode(self.nodes[0], constraintsA)
    if not constraintsB is None:
      self.constraintsB = utils.ensure1DNumpyArray(constraintsB, np.bool_, [1,1,1,1,1,1])
      self.B.constrainChildNode(self.nodes[-1], constraintsB)

  def addEndStiffness(self, endStiffnessA:npt.NDArray[np.float64]=None, endStiffnessB:npt.NDArray[np.float64]=None):
    if endStiffnessA is None:
      endStiffnessA = np.zeros((6,6))
    if endStiffnessB is None:
      endStiffnessB = np.zeros((6,6))
    endStiffnessA, endStiffnessB = np.array(endStiffnessA), np.array(endStiffnessB)
    if endStiffnessA.ndim == 1:
      endStiffnessA = np.diag(endStiffnessA)
    if endStiffnessB.ndim == 1:
      endStiffnessB = np.diag(endStiffnessB)
    self.A.addChildNodeStiffness(self.nodes[0], endStiffnessA)
    self.B.addChildNodeStiffness(self.nodes[-1], endStiffnessB)
    self.endStiffnessA += endStiffnessA
    self.endStiffnessB += endStiffnessB

  def _addEndStiffnessFactor(self, endStiffnessFactorA:npt.NDArray[np.float64], endStiffnessFactorB:npt.NDArray[np.float64]):
    endSectionA = self.childBeams[0].section
    endStiffnessCoefficientA = np.array([
      endSectionA.E * endSectionA.Area,
      endSectionA.G * endSectionA.Area,
      endSectionA.G * endSectionA.Area,
      endSectionA.G * endSectionA.Ixx,
      endSectionA.E * endSectionA.Iyy,
      endSectionA.E * endSectionA.Izz,
      ])
    endSectionB = self.childBeams[-1].section
    endStiffnessCoefficientB = np.array([
      endSectionB.E * endSectionB.Area,
      endSectionB.G * endSectionB.Area,
      endSectionB.G * endSectionB.Area,
      endSectionB.G * endSectionB.Ixx,
      endSectionB.E * endSectionB.Iyy,
      endSectionB.E * endSectionB.Izz,
      ])
    self.addEndStiffness(endStiffnessFactorA * endStiffnessCoefficientA, endStiffnessFactorB * endStiffnessCoefficientB)
    self.endStiffnessFactorA += endStiffnessFactorA
    self.endStiffnessFactorB += endStiffnessFactorB

  def setEndStiffnessFactor(self, endStiffnessFactorA:npt.NDArray[np.float64]=None, endStiffnessFactorB:npt.NDArray[np.float64]=None):
    if endStiffnessFactorA is None:
      endStiffnessFactorA = np.zeros(6)
    if endStiffnessFactorB is None:
      endStiffnessFactorB = np.zeros(6)
    endStiffnessFactorA, endStiffnessFactorB = np.array(endStiffnessFactorA), np.array(endStiffnessFactorB)
    if endStiffnessFactorA.ndim != 1 or endStiffnessFactorB.ndim != 1:
      raise Exception("endStiffnessFactor should be 1-Dimensional")
    self._addEndStiffnessFactor(endStiffnessFactorA - self.endStiffnessFactorA, endStiffnessFactorB - self.endStiffnessFactorB)

  def addUDL(self, dir:int, val:float, loadCases:list[int]) -> None:
    for childBeam in self.childBeams:
      childBeam.addUDL(dir, val, loadCases)

  def addPointLoad(self, dir:int, val:float, dist:float, loadCases:list[int]) -> None:
    index = 0
    while index < len(self.childBeams) and dist > self.childBeams[index].L:
      dist -= self.childBeams[index].L
      index += 1
    if dist >= 0 and index < len(self.childBeams):
      self.childBeams[index].addPointLoad(dir, np.array([val]), np.array([dist]), loadCases)

  def addSelfWeight(self, dir:int, factor:float, loadCases:list[int]) -> None:
    for childBeam in self.childBeams:
      childBeam.addSelfWeight(dir, factor, loadCases)

  def addMassUDL(self, massUDL:Union[float, npt.NDArray[np.float64]]) -> None:
    for childBeam in self.childBeams:
      childBeam.addMassUDL(massUDL)

  def setAdditionalMassUDL(self, additionalMassUDL:Union[float, npt.NDArray[np.float64]]) -> None:
    for childBeam in self.childBeams:
      childBeam.setAdditionalMassUDL(additionalMassUDL)

  def setAdditionalMassFactor(self, selfWeightFactor:Union[float, npt.NDArray[np.float64]]) -> None:
    for childBeam in self.childBeams:
      childBeam.setAdditionalMassFactor(selfWeightFactor)

  @staticmethod
  def addPointLoadToBeamList(beamList:list[Beam], dir:int, val:float, dist:float, loadCases:list[int]) -> None:
    for index, (beam1, beam2) in enumerate(zip(beamList[:-1], beamList[1:])):
      if np.linalg.norm(beam1.nodes[-1].coord - beam2.nodes[0].coord) > 10 * np.finfo(np.float64).eps:
        raise Exception(f"The beam ends are discontinuous after beam at {index=}")
    index = 0
    while index < len(beamList) and dist > (beamLength := sum([_.L for _ in beamList[index].childBeams])):
      dist -= beamLength
      index += 1
    if dist > 0 and index < len(beamList):
      beamList[index].addPointLoad(dir, val, dist, loadCases)
