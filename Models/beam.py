import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field, InitVar
from . import Node, FixedBeam
from .utils import getAxisFromTwoNodesAndBeta

@dataclass
class Beam:
  nodes: list[Node]
  EA: InitVar[float] = 1
  EIy: InitVar[float] = 1
  EIz: InitVar[float] = 1
  GJx: InitVar[float] = 0
  beta: InitVar[float] = 0 # Angle in degrees
  A: Node = None
  B: Node = None
  isConstrainedA: npt.NDArray[np.bool_] = field(default_factory=lambda:np.ones((6,),dtype=np.bool_))
  isConstrainedB: npt.NDArray[np.bool_] = field(default_factory=lambda:np.ones((6,),dtype=np.bool_))
  endStiffnessA: npt.NDArray[np.float64] = field(default_factory=lambda:np.zeros((6,6)))
  endStiffnessB: npt.NDArray[np.float64] = field(default_factory=lambda:np.zeros((6,6)))
  childBeams: list[FixedBeam] = field(init=False, default_factory=list)

  def __post_init__(self, *args, beta):
    if len(self.nodes) < 2:
      raise Exception("Supply minimum of 2 nodes")
    if type(self.isConstrainedA) is list:
      self.isConstrainedA = np.array(self.isConstrainedA, dtype=np.bool_)
    if type(self.isConstrainedB) is list:
      self.isConstrainedB = np.array(self.isConstrainedB, dtype=np.bool_)
    if self.A is None:
      self.A = self.nodes[0]
      self.nodes[0] = Node(
        self.A.coord,
        np.zeros((6,), dtype=np.bool_),
        self.A.Kg,
        getAxisFromTwoNodesAndBeta(self.A.coord, self.nodes[1].coord, beta))
    if self.B is None:
      self.B = self.nodes[-1]
      self.nodes[-1] = Node(
        self.B.coord,
        np.zeros((6,), dtype=np.bool_),
        self.A.Kg,
        getAxisFromTwoNodesAndBeta(self.nodes[-2].coord, self.B.coord, beta))
    self.A.constrainChildNode(self.nodes[0], self.isConstrainedA)
    self.B.constrainChildNode(self.nodes[-1], self.isConstrainedB)
    self.A.addChildNodeStiffness(self.nodes[0], self.endStiffnessA)
    self.B.addChildNodeStiffness(self.nodes[-1], self.endStiffnessB)
    for nodei, nodej in zip(self.nodes[:-1], self.nodes[1:]):
      self.childBeams.append(FixedBeam(nodei, nodej, *args, beta))

  def __del__(self):
    raise NotImplementedError(f"Deletion of {type(self).__name__} is not supported")

  def addConstrains(self, constraints:npt.NDArray[np.bool_]):
    raise NotImplementedError("Future feature")

  def addSimpleEndStiffness(self, endStiffnessA:npt.NDArray[np.float64]=None, endStiffnessB:npt.NDArray[np.float64]=None):
    if endStiffnessA is None:
      endStiffnessA = np.zeros((6,6))
    if endStiffnessB is None:
      endStiffnessB = np.zeros((6,6))
    if type(endStiffnessA) is list or endStiffnessA.ndim == 1:
      endStiffnessA = np.diag(endStiffnessA)
    if type(endStiffnessB) is list or endStiffnessB.ndim == 1:
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