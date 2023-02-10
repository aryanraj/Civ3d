import numpy as np
import numpy.typing as npt
from . import Node, FixedBeam
from dataclasses import dataclass, field

@dataclass
class Beam(FixedBeam):
  isConstrainedA: npt.NDArray[np.bool_] = field(default_factory=lambda:np.ones((6,),dtype=np.bool_))
  isConstrainedB: npt.NDArray[np.bool_] = field(default_factory=lambda:np.ones((6,),dtype=np.bool_))
  endStiffnessA: npt.NDArray[np.float64] = field(default_factory=lambda:np.zeros((6,6)))
  endStiffnessB: npt.NDArray[np.float64] = field(default_factory=lambda:np.zeros((6,6)))
  A: Node = None
  B: Node = None

  def __post_init__(self):
    if type(self.isConstrainedA) is list:
      self.isConstrainedA = np.array(self.isConstrainedA, dtype=np.bool_)
    if type(self.isConstrainedB) is list:
      self.isConstrainedB = np.array(self.isConstrainedB, dtype=np.bool_)
    if self.A is None:
      self.A = self.i
      self.i = self.A.copy()
      self.i.addRestraint(np.zeros((6,), dtype=np.bool_))
    if self.B is None:
      self.B = self.j
      self.j = self.B.copy()
      self.j.addRestraint(np.zeros((6,), dtype=np.bool_))
    self.A.constrainChildNode(self.i, self.isConstrainedA)
    self.B.constrainChildNode(self.j, self.isConstrainedB)
    self.A.addChildNodeStiffness(self.i, self.endStiffnessA)
    self.B.addChildNodeStiffness(self.j, self.endStiffnessB)
    super().__post_init__()

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
    self.A.addChildNodeStiffness(self.i, endStiffnessA)
    self.B.addChildNodeStiffness(self.j, endStiffnessB)
    self.endStiffnessA += endStiffnessA
    self.endStiffnessB += endStiffnessB
