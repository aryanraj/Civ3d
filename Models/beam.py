import numpy as np
import numpy.typing as npt
from . import Node, FixedBeam
from dataclasses import dataclass, field

@dataclass
class Beam(FixedBeam):
  constraints: npt.NDArray[np.bool_] = field(default_factory=lambda:np.ones((12,1),dtype=np.bool_))
  A: Node = None
  B: Node = None

  def __post_init__(self):
    if type(self.constraints) is list:
      self.constraints = np.array(self.constraints, dtype=np.bool_)
    if self.A is None:
      self.A = self.i
      self.i = self.A.copy()
      self.i.addRestraint(np.zeros((6,), dtype=np.bool_))
    if self.B is None:
      self.B = self.j
      self.j = self.B.copy()
      self.j.addRestraint(np.zeros((6,), dtype=np.bool_))
    self.A.constrainChildNode(self.i, self.constraints[:6])
    self.B.constrainChildNode(self.j, self.constraints[6:])
    super().__post_init__()

  def __del__(self):
    raise NotImplementedError(f"Deletion of {type(self).__name__} is not supported")

  def addConstrains(self, constraints:npt.NDArray[np.bool_]):
    raise NotImplementedError("Future feature")