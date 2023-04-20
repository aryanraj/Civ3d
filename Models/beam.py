import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field, InitVar
from . import Node, BeamSection, FixedBeam
from . import utils

@dataclass
class Beam:
  nodes: list[Node]
  section: InitVar[BeamSection]
  beta: InitVar[float] = 0 # Angle in degrees
  A: Node = None
  B: Node = None
  constraintsA: InitVar[npt.NDArray[np.bool_]] = np.ones((6,),dtype=np.bool_)
  constraintsB: InitVar[npt.NDArray[np.bool_]] = np.ones((6,),dtype=np.bool_)
  endStiffnessA: npt.NDArray[np.float64] = field(default_factory=lambda:np.zeros((6,6)))
  endStiffnessB: npt.NDArray[np.float64] = field(default_factory=lambda:np.zeros((6,6)))
  childBeams: list[FixedBeam] = field(init=False, default_factory=list)

  def __post_init__(self, section:BeamSection, beta:float, constraintsA:npt.NDArray[np.bool_], constraintsB:npt.NDArray[np.bool_]):
    if len(self.nodes) < 2:
      raise Exception("Supply minimum of 2 nodes")
    if self.A is None:
      self.A = self.nodes[0]
      self.nodes[0] = Node(
        self.A.coord,
        np.zeros((6,), dtype=np.bool_),
        self.A.Kg,
        utils.getAxisFromTwoNodesAndBeta(self.A.coord, self.nodes[1].coord, beta=beta)
      )
    if self.B is None:
      self.B = self.nodes[-1]
      self.nodes[-1] = Node(
        self.B.coord,
        np.zeros((6,), dtype=np.bool_),
        self.A.Kg,
        utils.getAxisFromTwoNodesAndBeta(self.nodes[-2].coord, self.B.coord, beta=beta)
      )
    self.addEndConstrains(constraintsA, constraintsB)
    self.addEndStiffness(self.endStiffnessA, self.endStiffnessB)
    for nodei, nodej in zip(self.nodes[:-1], self.nodes[1:]):
      self.childBeams.append(FixedBeam(nodei, nodej, section, beta))

  def __del__(self):
    raise NotImplementedError(f"Deletion of {type(self).__name__} is not supported")

  def addEndConstrains(self, constraintsA:npt.NDArray[np.bool_], constraintsB:npt.NDArray[np.bool_]):
    self.A.constrainChildNode(self.nodes[0], constraintsA)
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

  def addUDL(self, dir: float, val: float) -> None:
    for childBeam in self.childBeams:
      childBeam.addUDL(dir, val)

  def addPointLoad(self, dir:int, val: float, dist: float) -> None:
    raise NotImplementedError("Future feature")

  def addSelfWeight(self, dir:int) -> None:
    for childBeam in self.childBeams:
      childBeam.addSelfWeight(dir)

  def addMassUDL(self, massPerLength:float) -> None:
    for childBeam in self.childBeams:
      childBeam.addMassUDL(massPerLength)
      